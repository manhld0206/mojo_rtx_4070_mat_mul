from std.math import ceildiv
from std.sys import argv

from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from std.testing import assert_almost_equal
from std.utils.index import Index


comptime Layout2DRow = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)


@always_inline
def _load_tile[
    NUM_THREADS: Int,
    PER_THREAD_COL: Int,
    TILE_ROW: Int,
    TILE_COL: Int,
](
    dram_tile: LayoutTensor[mut=False, DType.float16, ...],
    sram_tile: LayoutTensor[
        mut=True,
        DType.float16,
        address_space=AddressSpace.SHARED,
        element_layout=dram_tile.element_layout,
        ...,
    ],
):
    """Load tile from global memory to shared memory."""
    tid = thread_idx.y * block_dim.x + thread_idx.x
    comptime PER_THREAD_ROW = ceildiv(
        TILE_ROW * TILE_COL, NUM_THREADS
    ) / PER_THREAD_COL

    assert TILE_COL % PER_THREAD_COL == 0
    assert TILE_ROW % PER_THREAD_ROW == 0

    comptime load_layout = Layout.row_major(
        TILE_ROW / PER_THREAD_ROW, TILE_COL / PER_THREAD_COL
    )
    sram_fragment = (
        sram_tile.tile[TILE_ROW, TILE_COL](0, 0)
        .vectorize[1, PER_THREAD_COL]()
        .distribute[load_layout](tid)
    )
    dram_fragment = dram_tile.vectorize[1, PER_THREAD_COL]().distribute[
        load_layout
    ](tid)
    sram_fragment.copy_from_async(dram_fragment)


@always_inline
def _mm[
    BK: Int, TM: Int, TN: Int
](
    a_s: LayoutTensor[mut=False, DType.float16, ...],
    b_s: LayoutTensor[mut=False, DType.float16, ...],
    c_r: LayoutTensor[mut=True, DType.float32, ...],
):
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


@always_inline
def _write_tile[
    TM: Int,
    TN: Int,
](
    dram_tile: LayoutTensor[mut=True, DType.float16, ...],
    local_tile: LayoutTensor[
        mut=True,
        DType.float32,
        address_space=AddressSpace.LOCAL,
        element_layout=dram_tile.element_layout,
        ...,
    ],
):
    comptime for row in range(TM):
        if row < dram_tile.dim(0):
            var simd_data = SIMD[DType.float16, TN](0)
            comptime for col in range(TN):
                simd_data[col] = local_tile[row, col].cast[DType.float16]()[0]
            if dram_tile.dim(1) == TN:
                dram_tile.aligned_store[TN](row, 0, simd_data)
            else:
                comptime for col in range(TN):
                    if col < dram_tile.dim(1):
                        dram_tile[row, col] = simd_data[col]


def matmul_kernel[
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 16,
    TM: Int = 4,
    TN: Int = 8,
](
    c: LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin],
    a: LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin],
    b: LayoutTensor[DType.float16, Layout2DRow, MutAnyOrigin],
):
    var tid = thread_idx.y * block_dim.x + thread_idx.x

    # M = c.dim(0)
    # N = c.dim(1)
    K = a.dim(1)

    var a_s_curr = LayoutTensor[
        DType.float16,
        Layout.row_major(BM, BK + 2),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var a_s_next = LayoutTensor[
        DType.float16,
        Layout.row_major(BM, BK + 2),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var b_s_curr = LayoutTensor[
        DType.float16,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var b_s_next = LayoutTensor[
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

    # Prefetch first
    _load_tile[
        NUM_THREADS=NUM_THREADS,
        PER_THREAD_COL=2,
        TILE_ROW=BM,
        TILE_COL=BK,
    ](
        dram_tile=a.tile[BM, BK](block_idx.y, 0),
        sram_tile=a_s_curr,
    )

    _load_tile[
        NUM_THREADS=NUM_THREADS,
        PER_THREAD_COL=4,
        TILE_ROW=BK,
        TILE_COL=BN,
    ](
        dram_tile=b.tile[BK, BN](0, block_idx.x),
        sram_tile=b_s_curr,
    )

    async_copy_wait_all()
    barrier()

    for i in range(ceildiv(K, BK) - 1):
        _mm[BK, TM, TN](a_s_curr, b_s_curr, c_r)

        _load_tile[
            NUM_THREADS=NUM_THREADS,
            PER_THREAD_COL=2,
            TILE_ROW=BM,
            TILE_COL=BK,
        ](
            dram_tile=a.tile[BM, BK](block_idx.y, i + 1),
            sram_tile=a_s_next,
        )

        _load_tile[
            NUM_THREADS=NUM_THREADS,
            PER_THREAD_COL=4,
            TILE_ROW=BK,
            TILE_COL=BN,
        ](
            dram_tile=b.tile[BK, BN](i + 1, block_idx.x),
            sram_tile=b_s_next,
        )

        async_copy_wait_all()
        barrier()

        a_s_curr, a_s_next = a_s_next, a_s_curr
        b_s_curr, b_s_next = b_s_next, b_s_curr

    _mm[BK, TM, TN](a_s_curr, b_s_curr, c_r)

    _write_tile[TM, TN](
        dram_tile=c.tile[BM, BN](block_idx.y, block_idx.x).tile[TM, TN](
            thread_idx.y, thread_idx.x
        ),
        local_tile=c_r,
    )


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
