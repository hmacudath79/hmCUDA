#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

#define TILE_WIDTH 32

// Error checking helper
void checkCuda(cudaError_t result, const char* func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

#ifdef USE_CUBLAS
void checkCublas(cublasStatus_t result, const char* func) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error in " << func << std::endl;
        exit(1);
    }
}
#endif

// Naive Kernel
__global__ void matrixMulNaive(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled Kernel
__global__ void matrixMulTiled(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (row < M && (m * TILE_WIDTH + tx) < K)
            tile_A[ty][tx] = A[row * K + (m * TILE_WIDTH + tx)];
        else
            tile_A[ty][tx] = 0.0f;

        if ((m * TILE_WIDTH + ty) < K && col < N)
            tile_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        else
            tile_B[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void verify(const float* ref, const float* res, int M, int N) {
    double max_diff = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs(ref[i] - res[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max diff: " << max_diff << " ... ";
    if (max_diff < 1e-4) std::cout << "PASS" << std::endl;
    else std::cout << "FAIL" << std::endl;
}

int main() {
    int M = 2048;
    int K = 4096;
    int N = 1024;
    int iterations = 10;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    std::cout << "Matrix Dimensions: M=" << M << ", K=" << K << ", N=" << N << std::endl;

    // Host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(M * N);
    std::vector<float> h_C_res(M * N);

    // Initialize
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, bytes_A), "Alloc A");
    checkCuda(cudaMalloc(&d_B, bytes_B), "Alloc B");
    checkCuda(cudaMalloc(&d_C, bytes_C), "Alloc C");

    checkCuda(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice), "Copy A");
    checkCuda(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice), "Copy B");

    // Verify data was copied correctly
    std::cout << "\n=== Memcpy Verification ===" << std::endl;
    std::vector<float> verify_A(10);
    checkCuda(cudaMemcpy(verify_A.data(), d_A, 10 * sizeof(float), cudaMemcpyDeviceToHost), "Verify A");
    std::cout << "First 10 values of A:" << std::endl;
    std::cout << "  Host:   ";
    for (int i = 0; i < 10; i++) std::cout << h_A[i] << " ";
    std::cout << std::endl;
    std::cout << "  Device: ";
    for (int i = 0; i < 10; i++) std::cout << verify_A[i] << " ";
    std::cout << std::endl;
    bool match = true;
    for (int i = 0; i < 10; i++) {
        if (h_A[i] != verify_A[i]) {
            match = false;
            break;
        }
    }
    std::cout << "  Status: " << (match ? "MATCH ✓" : "MISMATCH ✗") << std::endl;
    std::cout << "=========================" << std::endl;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "EventCreate");
    checkCuda(cudaEventCreate(&stop), "EventCreate");

    float milliseconds = 0;
    double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    std::cout << "Operations: " << ops << std::endl;
    
#ifdef USE_CUBLAS
    // 1. cuBLAS (Reference)
    std::cout << "\nRunning cuBLAS..." << std::endl;
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");
    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    checkCuda(cudaDeviceSynchronize(), "Sync");

    checkCuda(cudaEventRecord(start), "Record Start");
    for (int i = 0; i < iterations; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    checkCuda(cudaEventRecord(stop), "Record Stop");
    checkCuda(cudaEventSynchronize(stop), "Sync Stop");
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Time");
    
    std::cout << "Avg Time: " << milliseconds / iterations << " ms" << std::endl;
    std::cout << "Performance: " << (ops * 1e-9) / (milliseconds / iterations / 1000.0f) << " Gflop/s" << std::endl;
    
    checkCuda(cudaMemcpy(h_C_ref.data(), d_C, bytes_C, cudaMemcpyDeviceToHost), "Copy C");

#else
    std::cout << "\ncuBLAS disabled. Computing reference on CPU (this may take a moment)..." << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
#endif

    // 2. Naive
    std::cout << "\nRunning Naive..." << std::endl;
    checkCuda(cudaMemset(d_C, 0, bytes_C), "Memset C");
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matrixMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, K, N); // Warmup
    checkCuda(cudaDeviceSynchronize(), "Sync");

    checkCuda(cudaEventRecord(start), "Record Start");
    for (int i = 0; i < iterations; ++i) {
        matrixMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    checkCuda(cudaEventRecord(stop), "Record Stop");
    checkCuda(cudaEventSynchronize(stop), "Sync Stop");
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Time");

    std::cout << "Avg Time: " << milliseconds / iterations << " ms" << std::endl;
    std::cout << "Performance: " << (ops * 1e-9) / (milliseconds / iterations / 1000.0f) << " Gflop/s" << std::endl;

    checkCuda(cudaMemcpy(h_C_res.data(), d_C, bytes_C, cudaMemcpyDeviceToHost), "Copy C");
    verify(h_C_ref.data(), h_C_res.data(), M, N);

    // 3. Tiled
    std::cout << "\nRunning Tiled..." << std::endl;
    checkCuda(cudaMemset(d_C, 0, bytes_C), "Memset C");
    dim3 blockTiled(TILE_WIDTH, TILE_WIDTH);
    dim3 gridTiled((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMulTiled<<<gridTiled, blockTiled>>>(d_A, d_B, d_C, M, K, N); // Warmup
    checkCuda(cudaDeviceSynchronize(), "Sync");

    checkCuda(cudaEventRecord(start), "Record Start");
    for (int i = 0; i < iterations; ++i) {
        matrixMulTiled<<<gridTiled, blockTiled>>>(d_A, d_B, d_C, M, K, N);
    }
    checkCuda(cudaEventRecord(stop), "Record Stop");
    checkCuda(cudaEventSynchronize(stop), "Sync Stop");
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Time");

    std::cout << "Avg Time: " << milliseconds / iterations << " ms" << std::endl;
    std::cout << "Performance: " << (ops * 1e-9) / (milliseconds / iterations / 1000.0f) << " Gflop/s" << std::endl;

    checkCuda(cudaMemcpy(h_C_res.data(), d_C, bytes_C, cudaMemcpyDeviceToHost), "Copy C");
    verify(h_C_ref.data(), h_C_res.data(), M, N);

    // Cleanup
#ifdef USE_CUBLAS
    checkCublas(cublasDestroy(handle), "cublasDestroy");
#endif
    checkCuda(cudaFree(d_A), "Free A");
    checkCuda(cudaFree(d_B), "Free B");
    checkCuda(cudaFree(d_C), "Free C");
    checkCuda(cudaEventDestroy(start), "EventDestroy");
    checkCuda(cudaEventDestroy(stop), "EventDestroy");

    return 0;
}
