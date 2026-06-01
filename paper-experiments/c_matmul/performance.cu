#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %s:%d\n", __FILE__, __LINE__); \
        return -1; \
    } \
} while(0)

// Convert SM version to CUDA cores per SM
int cores_per_sm(int major, int minor) {
    switch ((major << 4) + minor) {
        case 0x30: return 192; // Kepler
        case 0x32: return 192;
        case 0x35: return 192;
        case 0x37: return 192;
        case 0x50: return 128; // Maxwell
        case 0x52: return 128;
        case 0x53: return 128;
        case 0x60: return 64;  // Pascal
        case 0x61: return 128;
        case 0x62: return 128;
        case 0x70: return 64;  // Volta
        case 0x72: return 64;
        case 0x75: return 64;  // Turing
        case 0x80: return 64;  // Ampere
        case 0x86: return 128;
        case 0x89: return 128; // Ada
        case 0x90: return 128; // Hopper
        default: return 64;    // fallback
    }
}

int main() {
    // -----------------------------
    // Matrix dimensions
    // -----------------------------
    const int m = 4096;
    const int n = 4096;
    const int k = 4096;

    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    // -----------------------------
    // Host memory
    // -----------------------------
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    for (int i = 0; i < m*k; i++) h_A[i] = 1.0f;
    for (int i = 0; i < k*n; i++) h_B[i] = 1.0f;

    // -----------------------------
    // Device memory
    // -----------------------------
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // -----------------------------
    // cuBLAS handle
    // -----------------------------
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Use only normal CUDA cores (no Tensor Cores)
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    float alpha = 1.0f;
    float beta  = 0.0f;

    // -----------------------------
    // Warm-up
    // -----------------------------
    for (int i = 0; i < 5; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, d_A, m,
                    d_B, k,
                    &beta, d_C, m);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // -----------------------------
    // Timing
    // -----------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 10;

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, d_A, m,
                    d_B, k,
                    &beta, d_C, m);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / iters;

    // -----------------------------
    // Achieved GFLOP/s
    // -----------------------------
    double flops = 2.0 * (double)m * n * k;
    double achieved_gflops = (flops / 1e9) / (avg_ms / 1e3);

    // -----------------------------
    // Compute theoretical peak (CUDA cores)
    // -----------------------------
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    int sm_count = prop.multiProcessorCount;
    int cores_perSM = cores_per_sm(prop.major, prop.minor);
    int total_cuda_cores = sm_count * cores_perSM;
    double clock_ghz = prop.clockRate * 1e-6; // kHz → GHz

    double peak_gflops = 2.0 * total_cuda_cores * clock_ghz;

    // -----------------------------
    // Fraction of peak
    // -----------------------------
    double fraction_peak = achieved_gflops / peak_gflops * 100.0;

    // -----------------------------
    // Output
    // -----------------------------
    printf("Maximum FP32 GEMM on normal CUDA cores\n");
    printf("GPU: %s\n", prop.name);
    printf("SM version: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d, CUDA cores: %d, Clock: %.2f GHz\n", sm_count, total_cuda_cores, clock_ghz);
    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Average GEMM time: %.3f ms\n", avg_ms);
    printf("Achieved GFLOP/s: %.2f\n", achieved_gflops);
    printf("Theoretical FP32 peak: %.2f GFLOP/s\n", peak_gflops);
    printf("Fraction of peak: %.2f %%\n", fraction_peak);

    // -----------------------------
    // Cleanup
    // -----------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

