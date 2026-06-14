from std.math import ceildiv
from std.sys import argv

from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from layout.layout_tensor import (
    copy_dram_to_sram,
    copy_dram_to_sram_async,
)
from std.testing import assert_almost_equal
from std.utils.index import Index


comptime Layout2DRow = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)


@always_inline
def load_tile[
    num_threads: Int,
](
    T: UnsafePointer[Float16, _],
    lda: Int,
    maxRow: Int,
    maxCol: Int,
    T_s: UnsafePointer[
        mut=True,
        Scalar[DType.float16],
        _,
        address_space=AddressSpace.SHARED,
    ],
    ldas: Int,
    height: Int,
    width: Int,
):
    """Load tile from global memory to shared memory.

    Args:
        T: Pointer to global memory tile.
        lda: Leading dimension of T (stride).
        maxRow: Maximum valid row to load.
        maxCol: Maximum valid column to load.
        T_s: Pointer to shared memory tile.
        ldas: Leading dimension of T_s (stride).
        height: Height of tile to load.
        width: Width of tile to load.
    """
    var num_rows_per_tile = num_threads // width
    var num_subtiles = height // num_rows_per_tile

    var tx = thread_idx.x

    for subtile in range(num_subtiles):
        var row, col = divmod(tx, width)
        row += subtile * num_rows_per_tile

        if row < maxRow and col < maxCol:
            T_s[row * ldas + col] = Scalar[DType.float16](T[row * lda + col])
        else:
            T_s[row * ldas + col] = Scalar[DType.float16](0.0)


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

    # M = c.dim(0)
    # N = c.dim(1)
    K = a.dim(1)

    var a_s = LayoutTensor[
        DType.float16,
        Layout.row_major(BM, BK + 1),
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

    comptime NUM_THREADS = BM / TM * BN / TN
    comptime thread_layout = Layout.row_major(BM / TM, BN / TN)
    comptime load_a_layout = Layout.row_major(NUM_THREADS / BK, BK)
    comptime load_b_layout = Layout.row_major(BK, NUM_THREADS / BK)

    for i in range(ceildiv(K, BK)):
        # Copy with distribute function
        # a_fragment = (
        #     a.tile[BM, BK](block_idx.y, i)
        #     .vectorize[1, 2]()
        #     .distribute[thread_layout](tid)
        # )
        # a_fragment_s = a_s.vectorize[1, 2]().distribute[thread_layout](tid)
        # b_fragment = (
        #     b.tile[BK, BN](i, block_idx.x)
        #     .vectorize[1, 2]()
        #     .distribute[thread_layout](tid)
        # )
        # b_fragment_s = b_s.vectorize[1, 2]().distribute[thread_layout](tid)

        # a_fragment_s.copy_from(a_fragment)
        # b_fragment_s.copy_from(b_fragment)

        # Copy with copy_dram_to_sram function
        # copy_dram_to_sram[thread_layout=load_a_layout, block_dim_count=2](
        #     dst=a_s,
        #     src=a.tile[BM, BK](block_idx.y, i),
        # )

        # copy_dram_to_sram[thread_layout=load_b_layout, block_dim_count=2](
        #     dst=b_s,
        #     src=b.tile[BK, BN](i, block_idx.x),
        # )

        # Copy with manual tiling
        comptime a_sub_tiles_per_thread = ceildiv((BM * BK), NUM_THREADS)
        comptime A_TK = 2
        comptime A_TM = a_sub_tiles_per_thread / A_TK

        a_row, a_col = divmod(tid, BK / A_TK)
        a_tile = a.tile[BM, BK](block_idx.y, i).tile[A_TM, A_TK](a_row, a_col)
        # a_tile_s = a_s.tile[A_TM, A_TK](a_row, a_col)

        comptime for row in range(A_TM):
            if row < a_tile.dim(0):
                a_s_row = a_row * A_TM + row
                a_s_col = a_col * A_TK
                if a_tile.dim(1) == A_TK:
                    a_s.aligned_store[A_TK](
                        a_s_row,
                        a_s_col,
                        a_tile.aligned_load[A_TK](row, 0),
                    )
                else:
                    comptime for col in range(A_TK):
                        if col < a_tile.dim(1):
                            a_s[a_s_row, a_s_col + col] = a_tile[row, col]
                        else:
                            a_s[a_s_row, a_s_col + col] = 0

        comptime b_sub_tiles_per_thread = ceildiv((BK * BN), NUM_THREADS)
        comptime B_TN = 2
        comptime B_TK = b_sub_tiles_per_thread / B_TN

        b_row, b_col = divmod(tid, BN / B_TN)
        b_tile = b.tile[BK, BN](i, block_idx.x).tile[B_TK, B_TN](b_row, b_col)
        # b_tile_s = b_s.tile[B_TK, B_TN](b_row, b_col)

        comptime for row in range(B_TK):
            if row < b_tile.dim(0):
                b_s_row = a_row * B_TK + row
                b_s_col = a_col * B_TN
                if b_tile.dim(1) == B_TN:
                    b_s.aligned_store[B_TN](
                        b_s_row,
                        b_s_col,
                        b_tile.aligned_load[B_TN](row, 0),
                    )
                else:
                    comptime for col in range(B_TN):
                        if col < b_tile.dim(1):
                            b_s[b_s_row, b_s_col + col] = b_tile[row, col]
                        else:
                            b_s[b_s_row, b_s_col + col] = 0

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


def benchmark_kernel(
    M: Int, N: Int, K: Int, num_runs: Int, num_warmup: Int, ctx: DeviceContext
) raises:
    print(M, "x", N, "x", K)
    comptime BM = 64
    comptime BN = 64
    comptime BK = 16
    comptime TM = 4
    comptime TN = 8

    rounded_m = BM * ceildiv(M, BM)
    rounded_n = BN * ceildiv(N, BN)
    rounded_k = BK * ceildiv(K, BK)

    var a_layout = RuntimeLayout[Layout2DRow].row_major(
        Index(rounded_m, rounded_k)
    )
    var b_layout = RuntimeLayout[Layout2DRow].row_major(
        Index(rounded_k, rounded_n)
    )
    var c_layout = RuntimeLayout[Layout2DRow].row_major(
        Index(rounded_m, rounded_n)
    )

    var d_a = ctx.enqueue_create_buffer[DType.float16](rounded_m * rounded_k)
    var d_b = ctx.enqueue_create_buffer[DType.float16](rounded_k * rounded_n)
    var d_c = ctx.enqueue_create_buffer[DType.float16](rounded_m * rounded_n)

    var h_a = ctx.enqueue_create_host_buffer[DType.float16](
        rounded_m * rounded_k
    )
    var h_b = ctx.enqueue_create_host_buffer[DType.float16](
        rounded_k * rounded_n
    )
    var h_c = ctx.enqueue_create_host_buffer[DType.float16](
        rounded_m * rounded_n
    )
    for i in range(rounded_m):
        for j in range(rounded_k):
            if i < M and j < K:
                h_a[i * rounded_k + j] = Scalar[DType.float16](1.0)
    for i in range(rounded_k):
        for j in range(rounded_n):
            if i < K and j < N:
                h_b[i * rounded_n + j] = Scalar[DType.float16](2.0)

    h_a.enqueue_copy_to(d_a)
    h_b.enqueue_copy_to(d_b)
    ctx.synchronize()

    a = LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin](d_a, a_layout)
    b = LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin](d_b, b_layout)
    c = LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin](d_c, c_layout)

    comptime kernel = matmul_kernel[BM=BM, BN=BN, BK=BK, TM=TM, TN=TN]
    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        grid_dim=(ceildiv(rounded_n, BN), ceildiv(rounded_m, BM)),
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
            grid_dim=(ceildiv(rounded_n, BN), ceildiv(rounded_m, BM)),
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
