#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char* argv[]) {
    // Default matrix size (M=N=K) and iterations
    int matrix_size = 4096;
    int ITERATIONS = 100;

    // Parse command line arguments if provided
    if (argc > 1) matrix_size = std::atoi(argv[1]);
    if (argc > 2) ITERATIONS = std::atoi(argv[2]);

    // Basic input validation
    if (matrix_size <= 0 || ITERATIONS <= 0) {
        std::cerr << "Error: Matrix size and iterations must be greater than 0." << std::endl;
        std::cerr << "Usage: ./benchmark_cublas [SIZE] [ITERATIONS]" << std::endl;
        return 1;
    }

    // Set M, N, and K to the unified size
    int M = matrix_size;
    int N = matrix_size;
    int K = matrix_size;

    std::cout << "==========================================" << std::endl;
    std::cout << "Benchmarking cuBLAS SGEMM (Square Matrix)" << std::endl;
    std::cout << "Size (M=N=K): " << matrix_size << " x " << matrix_size << std::endl;
    std::cout << "Iterations:   " << ITERATIONS << std::endl;
    std::cout << "==========================================" << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Calculate memory requirements (using size_t to prevent overflow on massive matrices)
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    // 1. Allocate Host (CPU) Memory
    std::vector<float> h_A((size_t)M * K, 1.0f);
    std::vector<float> h_B((size_t)K * N, 2.0f);
    std::vector<float> h_C((size_t)M * N, 0.0f);

    // 2. Allocate Device (GPU) Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 3. Copy Initialized Data from Host to Device
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;

    // 4. Warm-up run (CRITICAL: Removes lazy-loading context overhead)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    // 5. Setup CUDA Events for high-precision timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 6. Benchmark Loop
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                    &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 7. Calculate Time and Performance
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_ms = elapsed_ms / ITERATIONS;

    // Floating point operations: 2 * M * N * K
    // Cast to double to prevent overflow before division
    double tflops = (2.0 * (double)M * (double)N * (double)K) / (avg_ms * 1e9);

    std::cout << "cuBLAS Avg Time:    " << avg_ms << " ms" << std::endl;
    std::cout << "cuBLAS Performance: " << tflops << " TFLOPS" << std::endl;

    // ==========================================
    // 8. CORRECTNESS VERIFICATION
    // ==========================================
    
    // Copy the final result from Device back to Host
    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    float expected_value = 1.0f * 2.0f * K; 
    bool is_correct = true;
    float max_error = 0.0f;

    // Verify all elements (using size_t for index)
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        float error = std::abs(h_C[i] - expected_value);
        if (error > max_error) {
            max_error = error;
        }
        
        if (error > 1e-3) {
            std::cerr << "\n[!] Verification FAILED at index " << i << "!" << std::endl;
            std::cerr << "Expected: " << expected_value << ", Got: " << h_C[i] << std::endl;
            is_correct = false;
            break; 
        }
    }

    if (is_correct) {
        std::cout << "Verification PASSED! (Max error: " << max_error << ")" << std::endl;
    }

    // 9. Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cublasDestroy(handle);

    return 0;
}
