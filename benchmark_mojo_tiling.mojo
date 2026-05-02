from std.math import ceildiv
from std.sys import argv

from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from std.testing import assert_almost_equal
from std.utils.index import Index

from std.utils.index import IndexList


comptime Layout2D = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)


def matmul_kernel[
    BLOCKSIZE: Int = 16,
    SIMD_WIDTH: Int = 4,
](
    c: LayoutTensor[DType.bfloat16, Layout2D, MutAnyOrigin],
    a: LayoutTensor[DType.bfloat16, Layout2D, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, Layout2D, MutAnyOrigin],
):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    var tid = thread_idx.y * block_dim.x + thread_idx.x

    M = c.dim(0)
    N = c.dim(1)
    K = a.dim(1)

    var a_tile_s = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BLOCKSIZE, BLOCKSIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var b_tile_s = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BLOCKSIZE, BLOCKSIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var c_val: Float32 = 0

    for i in range(ceildiv(K, BLOCKSIZE)):
        a_tile = a.tile[BLOCKSIZE, BLOCKSIZE](block_idx.y, i)
        b_tile = b.tile[BLOCKSIZE, BLOCKSIZE](i, block_idx.x)
        if thread_idx.y < a_tile.dim(0) and thread_idx.x < a_tile.dim(1):
            a_tile_s[thread_idx.y, thread_idx.x] = a_tile[
                thread_idx.y, thread_idx.x
            ]
        else:
            a_tile_s[thread_idx.y, thread_idx.x] = Scalar[DType.bfloat16](0)
        if thread_idx.y < b_tile.dim(0) and thread_idx.x < b_tile.dim(1):
            b_tile_s[thread_idx.y, thread_idx.x] = b_tile[
                thread_idx.y, thread_idx.x
            ]
        else:
            b_tile_s[thread_idx.y, thread_idx.x] = Scalar[DType.bfloat16](0)
        barrier()

        for k in range(BLOCKSIZE):
            var a_val = rebind[Float32](
                a_tile_s[thread_idx.y, k].cast[DType.float32]()
            )
            var b_val = rebind[Float32](
                b_tile_s[k, thread_idx.x].cast[DType.float32]()
            )
            c_val += a_val * b_val

        barrier()

    if row < M and col < N:
        c[row, col] = c_val.cast[DType.bfloat16]()


def benchmark_kernel(M: Int, N: Int, K: Int, ctx: DeviceContext) raises:
    print(M, "x", N, "x", K)

    var a_layout = RuntimeLayout[Layout2D].row_major(Index(M, K))
    var b_layout = RuntimeLayout[Layout2D].row_major(Index(K, N))
    var c_layout = RuntimeLayout[Layout2D].row_major(Index(M, N))

    var d_a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var d_b = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    var d_c = ctx.enqueue_create_buffer[DType.bfloat16](M * N)

    var h_a = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var h_b = ctx.enqueue_create_host_buffer[DType.bfloat16](K * N)
    var h_c = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    for i in range(M * K):
        h_a[i] = Scalar[DType.bfloat16](1.0)
    for i in range(K * N):
        h_b[i] = Scalar[DType.bfloat16](2.0)

    h_a.enqueue_copy_to(d_a)
    h_b.enqueue_copy_to(d_b)
    ctx.synchronize()

    a = LayoutTensor[DType.bfloat16, Layout2D, MutAnyOrigin](d_a, a_layout)
    b = LayoutTensor[DType.bfloat16, Layout2D, MutAnyOrigin](d_b, b_layout)
    c = LayoutTensor[DType.bfloat16, Layout2D, MutAnyOrigin](d_c, c_layout)

    comptime BLOCKSIZE = 16
    comptime kernel = matmul_kernel[BLOCKSIZE=BLOCKSIZE]
    ctx.enqueue_function[kernel, kernel](
        c,
        a,
        b,
        grid_dim=(ceildiv(N, BLOCKSIZE), ceildiv(M, BLOCKSIZE)),
        block_dim=(BLOCKSIZE, BLOCKSIZE),
    )

    ctx.synchronize()

    comptime num_runs = 10
    comptime num_warmup = 2

    @always_inline
    @parameter
    def run_kernel(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel, kernel](
            c,
            a,
            b,
            grid_dim=(ceildiv(N, BLOCKSIZE), ceildiv(M, BLOCKSIZE)),
            block_dim=(BLOCKSIZE, BLOCKSIZE),
        )

    for _ in range(num_warmup):
        run_kernel(ctx)
    ctx.synchronize()
    print("finished warmup")

    var nstime = Float64(ctx.execution_time[run_kernel](num_runs)) / num_runs
    var sectime = nstime * 1e-9
    var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12

    print("  Average time: ", sectime * 1000, " ms")
    print("  Performance: ", TFlop / sectime, " TFLOPS")
    print()

    h_c.enqueue_copy_from(d_c)
    ctx.synchronize()
    for i in range(M):
        assert_almost_equal(
            h_c[i].cast[DType.float32](),
            Scalar[DType.float32](2.0 * Float32(K)),
            "Mismatch at index "
            + String(i)
            + ": "
            + String(h_c[i].cast[DType.float32]()),
        )


def main() raises:
    args = argv()
    var size = Int(args[1] if len(args) > 1 else "4096")
    M = size
    N = size
    K = size
    with DeviceContext() as ctx:
        benchmark_kernel(M, N, K, ctx)
        return
