#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // Required for __half data type
#include <cublas_v2.h>

int main(int argc, char *argv[])
{
    // Default matrix size (M=N=K) and iterations
    int matrix_size = 4096;
    int ITERATIONS = 100;

    // Parse command line arguments if provided
    if (argc > 1)
        matrix_size = std::atoi(argv[1]);
    if (argc > 2)
        ITERATIONS = std::atoi(argv[2]);

    if (matrix_size <= 0 || ITERATIONS <= 0)
    {
        std::cerr << "Error: Matrix size and iterations must be greater than 0." << std::endl;
        std::cerr << "Usage: ./benchmark_cublas_fp16 [SIZE] [ITERATIONS]" << std::endl;
        return 1;
    }

    int M = matrix_size;
    int N = matrix_size;
    int K = matrix_size;

    std::cout << "==========================================" << std::endl;
    std::cout << "Benchmarking cuBLAS mixed-precision (FP16/FP32)" << std::endl;
    std::cout << "Size (M=N=K): " << matrix_size << " x " << matrix_size << std::endl;
    std::cout << "Iterations:   " << ITERATIONS << std::endl;
    std::cout << "==========================================" << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // 1. Enable Tensor Cores
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Calculate memory requirements using __half (2 bytes per element)
    size_t size_A = (size_t)M * K * sizeof(__half);
    size_t size_B = (size_t)K * N * sizeof(__half);
    size_t size_C = (size_t)M * N * sizeof(__half);

    // 2. Allocate and initialize Host (CPU) Memory using __float2half
    std::vector<__half> h_A((size_t)M * K, __float2half(1.0f));
    std::vector<__half> h_B((size_t)K * N, __float2half(2.0f));
    std::vector<__half> h_C((size_t)M * N, __float2half(0.0f));

    // Allocate Device (GPU) Memory
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy Initialized Data from Host to Device
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // 3. Alpha and Beta must match the COMPUTE type (FP32), not the memory type (FP16)
    float alpha = 1.0f;
    float beta = 0.0f;

    // 4. Warm-up run using cublasGemmEx
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha,
                 d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();

    // Setup CUDA Events for high-precision timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark Loop
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; ++i)
    {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &alpha,
                     d_B, CUDA_R_16F, N,
                     d_A, CUDA_R_16F, K,
                     &beta,
                     d_C, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate Time and Performance
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_ms = elapsed_ms / ITERATIONS;

    // Floating point operations: 2 * M * N * K
    double tflops = (2.0 * (double)M * (double)N * (double)K) / (avg_ms * 1e9);

    std::cout << "cuBLAS Avg Time:    " << avg_ms << " ms" << std::endl;
    std::cout << "cuBLAS Performance: " << tflops << " TFLOPS" << std::endl;

    // ==========================================
    // CORRECTNESS VERIFICATION
    // ==========================================

    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    float expected_value = 1.0f * 2.0f * K;
    bool is_correct = true;
    float max_error = 0.0f;

    // 5. Dynamic tolerance based on FP16 precision limits
    float tolerance = std::max(1.0f, expected_value * 0.005f);

    for (size_t i = 0; i < (size_t)M * N; ++i)
    {
        // Convert __half back to float for CPU-side verification
        float val = __half2float(h_C[i]);
        float error = std::abs(val - expected_value);
        if (error > max_error)
        {
            max_error = error;
        }

        if (error > tolerance)
        {
            std::cerr << "\n[!] Verification FAILED at index " << i << "!" << std::endl;
            std::cerr << "Expected: " << expected_value << ", Got: " << val << std::endl;
            is_correct = false;
            break;
        }
    }

    if (is_correct)
    {
        std::cout << "Verification PASSED! (Max error: " << max_error << ")" << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    return 0;
}
