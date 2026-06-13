from std.math import ceildiv
from std.sys import argv

from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from std.testing import assert_almost_equal
from std.utils.index import Index


comptime Layout2DRow = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)


def matmul_kernel[
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 16,
    TM: Int = 4,
    TN: Int = 4,
](
    c: LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin],
    a: LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin],
    b: LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin],
):
    var tid = thread_idx.y * block_dim.x + thread_idx.x

    M = c.dim(0)
    N = c.dim(1)
    K = a.dim(1)

    var a_s = LayoutTensor[
        DType.float16,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var b_s = LayoutTensor[
        DType.float16,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var c_r = LayoutTensor[
        DType.float32,
        Layout.row_major(TM, TN),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()

    comptime for i in range(TM):
        comptime for j in range(TN):
            c_r[i, j] = 0

    comptime thread_layout = Layout.row_major(BM / TM, BN / TN)

    for i in range(ceildiv(K, BK)):
        a_fragment = (
            a.tile[BM, BK](block_idx.y, i)
            .vectorize[1, 2]()
            .distribute[thread_layout](tid)
        )
        a_fragment_s = a_s.vectorize[1, 2]().distribute[thread_layout](tid)
        b_fragment = (
            b.tile[BK, BN](i, block_idx.x)
            .vectorize[1, 2]()
            .distribute[thread_layout](tid)
        )
        b_fragment_s = b_s.vectorize[1, 2]().distribute[thread_layout](tid)

        a_fragment_s.copy_from(a_fragment)
        b_fragment_s.copy_from(b_fragment)

        barrier()

        comptime for k in range(BK):
            a_r = SIMD[DType.float32, TM](0)
            b_r = SIMD[DType.float32, TN](0)

            comptime for row in range(TM):
                a_r[row] = rebind[Float32](
                    a_s[thread_idx.y * TM + row, k].cast[DType.float32]()
                )

            comptime for col in range(TN):
                b_r[col] = rebind[Float32](
                    b_s[k, thread_idx.x * TN + col].cast[DType.float32]()
                )

            comptime for row in range(TM):
                comptime for col in range(TN):
                    c_r[row, col] += a_r[row] * b_r[col]

        barrier()

    c_tile = c.tile[BM, BN](block_idx.y, block_idx.x).tile[TM, TN](
        thread_idx.y, thread_idx.x
    )
    comptime for row in range(TM):
        if row < c_tile.dim(0):
            var simd_data = SIMD[DType.float16, TN](0)
            comptime for col in range(TN):
                simd_data[col] = c_r[row, col].cast[DType.float16]()[0]
            if c_tile.dim(1) == TN:
                c_tile.aligned_store[TN](row, 0, simd_data)
            else:
                comptime for col in range(TN):
                    if col < c_tile.dim(1):
                        c_tile[row, col] = simd_data[col]
        # comptime for col in range(TN):
        #     if row < c_tile.dim(0) and col < c_tile.dim(1):
        #         c_tile[row, col] = c_r[row, col].cast[DType.float16]()


def benchmark_kernel(
    M: Int, N: Int, K: Int, num_runs: Int, num_warmup: Int, ctx: DeviceContext
) raises:
    print(M, "x", N, "x", K)

    var a_layout = RuntimeLayout[Layout2DRow].row_major(Index(M, K))
    var b_layout = RuntimeLayout[Layout2DRow].row_major(Index(K, N))
    var c_layout = RuntimeLayout[Layout2DRow].row_major(Index(M, N))

    var d_a = ctx.enqueue_create_buffer[DType.float16](M * K)
    var d_b = ctx.enqueue_create_buffer[DType.float16](K * N)
    var d_c = ctx.enqueue_create_buffer[DType.float16](M * N)

    var h_a = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var h_b = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    var h_c = ctx.enqueue_create_host_buffer[DType.float16](M * N)
    for i in range(M * K):
        h_a[i] = Scalar[DType.float16](1.0)
    for i in range(K * N):
        h_b[i] = Scalar[DType.float16](2.0)

    h_a.enqueue_copy_to(d_a)
    h_b.enqueue_copy_to(d_b)
    ctx.synchronize()

    a = LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin](d_a, a_layout)
    b = LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin](d_b, b_layout)
    c = LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin](d_c, c_layout)

    comptime BM = 64
    comptime BN = 64
    comptime BK = 16
    comptime TM = 4
    comptime TN = 8
    comptime kernel = matmul_kernel[BM=BM, BN=BN, BK=BK, TM=TM, TN=TN]
    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(BN / TN, BM / TM),
    )

    ctx.synchronize()

    @always_inline
    @parameter
    def run_kernel(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BN / TN, BM / TM),
        )

    for _ in range(num_warmup):
        run_kernel(ctx)
    ctx.synchronize()
    print("finished warmup")

    var nstime = Float64(ctx.execution_time[run_kernel](num_runs)) / Float64(
        num_runs
    )
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
    var num_runs = Int(args[2] if len(args) > 2 else "4")
    var num_warmup = Int(args[3] if len(args) > 3 else "2")
    M = size
    N = size
    K = size
    with DeviceContext() as ctx:
        benchmark_kernel(
            M, N, K, num_runs=num_runs, num_warmup=num_warmup, ctx=ctx
        )
        return
