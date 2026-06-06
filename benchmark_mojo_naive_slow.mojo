from std.math import ceildiv
from std.sys import argv

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from std.testing import assert_almost_equal
from std.utils.index import Index


comptime Layout2DRow = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
comptime Layout2DCol = Layout.col_major(UNKNOWN_VALUE, UNKNOWN_VALUE)


def matmul_kernel[
    BLOCKSIZE: Int = 32,
](
    c: LayoutTensor[DType.bfloat16, Layout2DRow, MutAnyOrigin],
    a: LayoutTensor[DType.bfloat16, Layout2DRow, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, Layout2DCol, MutAnyOrigin],
):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x

    M = c.dim(0)
    N = c.dim(1)
    K = a.dim(1)

    if row < M and col < N:
        # Still accumulate in float32 for precision
        var acc: Float32 = 0

        for k in range(K):
            var a_val = rebind[Float32](a[row, k].cast[DType.float32]())
            var b_val = rebind[Float32](b[k, col].cast[DType.float32]())
            acc += a_val * b_val

        c[row, col] = acc.cast[DType.bfloat16]()


def benchmark_kernel(
    M: Int, N: Int, K: Int, num_runs: Int, num_warmup: Int, ctx: DeviceContext
) raises:
    print(M, "x", N, "x", K)

    var a_layout = RuntimeLayout[Layout2DRow].row_major(Index(M, K))
    var b_layout = RuntimeLayout[Layout2DCol].col_major(Index(K, N))
    var c_layout = RuntimeLayout[Layout2DRow].row_major(Index(M, N))

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

    a = LayoutTensor[DType.bfloat16, Layout2DRow, MutAnyOrigin](d_a, a_layout)
    b = LayoutTensor[DType.bfloat16, Layout2DCol, MutAnyOrigin](d_b, b_layout)
    c = LayoutTensor[DType.bfloat16, Layout2DRow, MutAnyOrigin](d_c, c_layout)

    comptime BLOCKSIZE = 16
    comptime kernel = matmul_kernel[BLOCKSIZE=BLOCKSIZE]
    # Use 1D thread block for memory coalescing

    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        grid_dim=(ceildiv(N, BLOCKSIZE), ceildiv(M, BLOCKSIZE)),
        block_dim=(BLOCKSIZE, BLOCKSIZE),
    )

    ctx.synchronize()

    @always_inline
    @parameter
    def run_kernel(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
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

    var nstime = Float64(ctx.execution_time[run_kernel](num_runs)) / Float64(
        num_runs
    )
    var sectime = nstime * 1e-9
    var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12

    print("  Average time: ", sectime * 1000, " ms")
    print("  Performance: ", TFlop / sectime, " TFLOPS")
    print()


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
