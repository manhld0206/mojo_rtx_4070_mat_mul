# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

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
    BLOCKSIZE: Int = 32,
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

    comptime thread_layout = Layout.row_major(BLOCKSIZE, BLOCKSIZE // SIMD_WIDTH)

    var c_val: Float32 = 0

    for i in range(ceildiv(K, BLOCKSIZE)):
        a_tile = a.tile[BLOCKSIZE, BLOCKSIZE](block_idx.y, i)
        a_tile_segment = a_tile.vectorize[1, SIMD_WIDTH]().distribute[
            thread_layout
        ](tid)
        a_shared_segment = a_tile_s.vectorize[1, SIMD_WIDTH]().distribute[
            thread_layout
        ](tid)
        a_shared_segment.copy_from_async(a_tile_segment)
        b_tile = b.tile[BLOCKSIZE, BLOCKSIZE](i, block_idx.x)
        b_tile_segment = b_tile.vectorize[1, SIMD_WIDTH]().distribute[
            thread_layout
        ](tid)
        b_shared_segment = b_tile_s.vectorize[1, SIMD_WIDTH]().distribute[
            thread_layout
        ](tid)
        b_shared_segment.copy_from_async(b_tile_segment)

        async_copy_wait_all()

        for k in range(a_tile_s.dim(1)):
            var a_val = rebind[Float32](
                a_tile_s[thread_idx.y, k].cast[DType.float32]()
            )
            var b_val = rebind[Float32](
                b_tile_s[k, thread_idx.x].cast[DType.float32]()
            )
            c_val += a_val * b_val

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

    comptime kernel = matmul_kernel[BLOCKSIZE=BLOCKSIZE]
    # Use 1D thread block for memory coalescing
    comptime BLOCKSIZE = 32

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
