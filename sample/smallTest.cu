#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

void checkCuda(cudaError_t result, const char* func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

// Simple kernel: C = A + B (element-wise)
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Matrix multiply: C[M x N] = A[M x K] * B[K x N]
__global__ void matMul(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void printArray(const char* label, const float* arr, int n) {
    printf("  %s: ", label);
    for (int i = 0; i < n; i++) printf("%8.3f ", arr[i]);
    printf("\n");
}

void printMatrix(const char* label, const float* mat, int rows, int cols) {
    printf("  %s:\n", label);
    for (int r = 0; r < rows; r++) {
        printf("    [");
        for (int c = 0; c < cols; c++) {
            printf("%8.3f", mat[r * cols + c]);
            if (c < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

// ============================================================
// Test 1: memcpy round-trip with known values
// ============================================================
bool test_memcpy_roundtrip() {
    printf("\n========== TEST 1: Memcpy Round-Trip ==========\n");
    const int N = 16;
    float h_src[N], h_dst[N];

    // Fill with known pattern: 1.0, 2.0, ..., 16.0
    for (int i = 0; i < N; i++) h_src[i] = (float)(i + 1);

    printf("  Source: ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_src[i]);
    printf("\n");

    float *d_buf;
    checkCuda(cudaMalloc(&d_buf, N * sizeof(float)), "Malloc");
    checkCuda(cudaMemcpy(d_buf, h_src, N * sizeof(float), cudaMemcpyHostToDevice), "H2D");
    memset(h_dst, 0, N * sizeof(float));
    checkCuda(cudaMemcpy(h_dst, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

    printf("  Result: ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_dst[i]);
    printf("\n");

    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (h_src[i] != h_dst[i]) {
            printf("  MISMATCH at [%d]: expected %.3f, got %.3f (raw: 0x%08x)\n",
                   i, h_src[i], h_dst[i], *(unsigned int*)&h_dst[i]);
            pass = false;
        }
    }
    checkCuda(cudaFree(d_buf), "Free");
    printf("  => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================
// Test 2: memcpy with larger buffer (crosses chunk boundary?)
// ============================================================
bool test_memcpy_large() {
    printf("\n========== TEST 2: Memcpy Large (1MB) ==========\n");
    const int N = 256 * 1024; // 1MB of floats
    float *h_src = new float[N];
    float *h_dst = new float[N];

    for (int i = 0; i < N; i++) h_src[i] = (float)(i % 1000) + 0.5f;

    float *d_buf;
    checkCuda(cudaMalloc(&d_buf, N * sizeof(float)), "Malloc");
    checkCuda(cudaMemcpy(d_buf, h_src, N * sizeof(float), cudaMemcpyHostToDevice), "H2D");
    memset(h_dst, 0, N * sizeof(float));
    checkCuda(cudaMemcpy(h_dst, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

    // Check first 8, last 8, and chunk boundaries
    bool pass = true;
    int check_positions[] = {
        0, 1, 2, 3, 4, 5, 6, 7,                          // first 8
        N-8, N-7, N-6, N-5, N-4, N-3, N-2, N-1,          // last 8
        65536, 65537,                                       // 256KB boundary (chunk 1->2)
        131072, 131073,                                     // 512KB boundary (chunk 2->3)
        196608, 196609,                                     // 768KB boundary (chunk 3->4)
    };
    int num_checks = sizeof(check_positions) / sizeof(check_positions[0]);

    printf("  Checking %d positions:\n", num_checks);
    for (int c = 0; c < num_checks; c++) {
        int i = check_positions[c];
        if (h_src[i] != h_dst[i]) {
            printf("    [%6d] expected %10.3f, got %10.3f (raw: 0x%08x) MISMATCH\n",
                   i, h_src[i], h_dst[i], *(unsigned int*)&h_dst[i]);
            pass = false;
        } else {
            printf("    [%6d] %10.3f OK\n", i, h_dst[i]);
        }
    }

    // Full scan for mismatches
    int mismatch_count = 0;
    int first_mismatch = -1;
    for (int i = 0; i < N; i++) {
        if (h_src[i] != h_dst[i]) {
            if (first_mismatch < 0) first_mismatch = i;
            mismatch_count++;
        }
    }
    if (mismatch_count > 0)
        printf("  Total mismatches: %d / %d (first at index %d)\n", mismatch_count, N, first_mismatch);
    else
        printf("  All %d values match!\n", N);

    checkCuda(cudaFree(d_buf), "Free");
    delete[] h_src;
    delete[] h_dst;
    printf("  => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================
// Test 3: Vector add (simple kernel)
// ============================================================
bool test_vector_add() {
    printf("\n========== TEST 3: Vector Add (kernel) ==========\n");
    const int N = 8;
    float h_A[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_B[N] = {10, 20, 30, 40, 50, 60, 70, 80};
    float h_C[N] = {0};
    float expected[N] = {11, 22, 33, 44, 55, 66, 77, 88};

    printArray("A", h_A, N);
    printArray("B", h_B, N);

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, N * sizeof(float)), "Malloc A");
    checkCuda(cudaMalloc(&d_B, N * sizeof(float)), "Malloc B");
    checkCuda(cudaMalloc(&d_C, N * sizeof(float)), "Malloc C");

    checkCuda(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);
    checkCuda(cudaDeviceSynchronize(), "Sync");

    checkCuda(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    printArray("C", h_C, N);
    printArray("Exp", expected, N);

    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != expected[i]) {
            printf("  MISMATCH at [%d]: expected %.1f, got %.1f\n", i, expected[i], h_C[i]);
            pass = false;
        }
    }

    checkCuda(cudaFree(d_A), "Free A");
    checkCuda(cudaFree(d_B), "Free B");
    checkCuda(cudaFree(d_C), "Free C");
    printf("  => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================
// Test 4: Small matrix multiply (4x3 * 3x2 = 4x2)
// ============================================================
bool test_matmul_small() {
    printf("\n========== TEST 4: MatMul 4x3 * 3x2 ==========\n");
    const int M = 4, K = 3, N = 2;

    // A = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    float h_A[M * K] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
    // B = [[1,2],[3,4],[5,6]]
    float h_B[K * N] = {1,2, 3,4, 5,6};
    float h_C[M * N] = {0};

    // Expected: CPU computation
    float expected[M * N];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += h_A[i * K + k] * h_B[k * N + j];
            expected[i * N + j] = sum;
        }

    printMatrix("A (4x3)", h_A, M, K);
    printMatrix("B (3x2)", h_B, K, N);
    printMatrix("Expected C", expected, M, N);

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, M * K * sizeof(float)), "Malloc A");
    checkCuda(cudaMalloc(&d_B, K * N * sizeof(float)), "Malloc B");
    checkCuda(cudaMalloc(&d_C, M * N * sizeof(float)), "Malloc C");

    checkCuda(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
    checkCuda(cudaMemset(d_C, 0, M * N * sizeof(float)), "Memset C");

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matMul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    checkCuda(cudaDeviceSynchronize(), "Sync");

    checkCuda(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    printMatrix("GPU C", h_C, M, N);

    bool pass = true;
    for (int i = 0; i < M * N; i++) {
        if (std::abs(h_C[i] - expected[i]) > 1e-3f) {
            printf("  MISMATCH at [%d]: expected %.3f, got %.3f\n", i, expected[i], h_C[i]);
            pass = false;
        }
    }

    checkCuda(cudaFree(d_A), "Free A");
    checkCuda(cudaFree(d_B), "Free B");
    checkCuda(cudaFree(d_C), "Free C");
    printf("  => %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================
int main() {
    printf("===================================\n");
    printf("  hmCUDA Small Correctness Tests\n");
    printf("===================================\n");

    int passed = 0, total = 0;

    total++; if (test_memcpy_roundtrip()) passed++;
    total++; if (test_memcpy_large()) passed++;
    total++; if (test_vector_add()) passed++;
    total++; if (test_matmul_small()) passed++;

    printf("\n===================================\n");
    printf("  Results: %d / %d passed\n", passed, total);
    printf("===================================\n");

    return (passed == total) ? 0 : 1;
}
