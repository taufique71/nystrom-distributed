//#include <numeric>
//#include <omp.h>
//#include <iostream>
//#include <vector>
//#include <cassert>

//#ifdef USE_CUBLAS
//#include <cublas_v2.h>
//#include <cuda_runtime.h>
//#define CUDA_CHECK(call) { \
    //cudaError_t err = call; \
    //if (err != cudaSuccess) { \
        //std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        //exit(err); \
    //} \
//}
//#else
//#include <mkl.h>
//#endif

////#include "procgrid.h"
//#include "prng.h"
//using namespace std;

//int main(int argc, char* argv[]) {

	//double t0, t1, t2, t3;
	
    //int cblas_m = 10000;
    //int cblas_k = 1000;
    //int cblas_n = 5000;
    //double cblas_alpha = 1.0;
    //double cblas_beta = 0.0;
    //double* cblas_a = new double[cblas_m * cblas_k];
    //int cblas_lda = 10000;
    //double* cblas_b = new double[cblas_k * cblas_n];
    //int cblas_ldb = 1000;
    //double* cblas_c = new double[cblas_m * cblas_n]; 
    //int cblas_ldc = 10000;
	
	////std::memset(cblas_a, 1.0, sizeof(double) * cblas_m * cblas_k);
	////std::memset(cblas_b, 2.0, sizeof(double) * cblas_k * cblas_n);
	//for (int i = 0; i < cblas_m * cblas_k; i++) cblas_a[i] = (double)(i);
	//for (int i = 0; i < cblas_k * cblas_n; i++) cblas_b[i] = (double)(i);
	////std::memset(cblas_c, 0, sizeof(double) * cblas_m * cblas_n);
	////for (int i = 0; i < cblas_m * cblas_n; i++) cblas_c = (double)(i);

//#ifdef USE_CUBLAS
	//double tMemMove = 0;
	//double tDgemm = 0;

	////t0 = MPI_Wtime();
	//double *d_A, *d_B, *d_C;
    //cudaError_t err;
	//err = cudaMalloc(&d_A, sizeof(double) * cblas_lda * cblas_k);
	//cudaMalloc(&d_B, sizeof(double) * cblas_ldb * cblas_n);
	//cudaMalloc(&d_C, sizeof(double) * cblas_ldc * cblas_n);
	//cudaMemcpy(d_A, cblas_a, sizeof(double) * cblas_lda * cblas_k, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_B, cblas_b, sizeof(double) * cblas_ldb * cblas_n, cudaMemcpyHostToDevice);
	//cudaMemset(d_C, 0, sizeof(double) * cblas_ldc * cblas_n);
	//cublasHandle_t handle;
	//cublasCreate(&handle);
	//cublasOperation_t transA = CUBLAS_OP_N;
	//cublasOperation_t transB = CUBLAS_OP_N;
	////t1 = MPI_Wtime();
	//tMemMove += (t1-t0);

    //int cublasVersion;
    //cublasGetVersion(handle, &cublasVersion);
    //printf("cuBLAS version: %d\n", cublasVersion);

	////t0 = MPI_Wtime();
	//cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
				//&cblas_alpha, d_A, cblas_lda, d_B, cblas_ldb,
				//&cblas_beta, d_C, cblas_ldc);
	////t1 = MPI_Wtime();
	//tDgemm += (t1-t0);

	////t0 = MPI_Wtime();
	//cudaMemcpy(cblas_c, d_C, sizeof(double) * cblas_ldc * cblas_n, cudaMemcpyDeviceToHost);
	//cublasDestroy(handle);
	//cudaFree(d_A);
	//cudaFree(d_B);
	//cudaFree(d_C);
	////t1 = MPI_Wtime();
	//tMemMove += (t1-t0);

		//printf("Time for local multiply host-device mem movement: %lf sec\n", tMemMove);
		//printf("Time for local multiply: %lf sec\n", tDgemm);
//#else

	//MKLVersion Version;
 
    //mkl_get_version(&Version);
 
 
    //printf("Major version:           %d\n",Version.MajorVersion);
    //printf("Minor version:           %d\n",Version.MinorVersion);
    //printf("Update version:          %d\n",Version.UpdateVersion);
    //printf("Product status:          %s\n",Version.ProductStatus);
    //printf("Build:                   %s\n",Version.Build);
    //printf("Platform:                %s\n",Version.Platform);
    //printf("Processor optimization:  %s\n",Version.Processor);
    //printf("================================================================\n");
    //printf("\n");
                                     
    ////t0 = MPI_Wtime();

    //cblas_dgemm(
        //CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
        //CblasNoTrans, // A matrix is not transpose. `transa` param of MKL cblas call.
        //CblasNoTrans, // B matrix is not transpose. `transb` param of MKL cblas call.
        //cblas_m, // Number of rows of A or C. `m` param of MKL cblas call.
        //cblas_n, // Number of cols of B or C. `n` param of MKL cblas call.
        //cblas_k, // Inner dimension - number of columns of A or number of rows of B. `k` param of MKL cblas call.
        //cblas_alpha, // Scalar `alpha` param of MKL cblas call.
        //cblas_a, // Data buffer of A. `a` param of MKL cblas call.
        //cblas_lda, // Leading dimension of A. `lda` param of MKL cblas call.
        //cblas_b, // Data buffer of B. `b` param of MKL cblas call.
        //cblas_ldb, // Leading dimension of B. `ldb` param of MKL cblas call.
        //cblas_beta, // Scalar `beta` param of MKL cblas call.
        //cblas_c, // Data buffer of C. `c` param of MKL cblas call.
        //cblas_ldc // Leading dimension of C. `ldc` param of MKL cblas call.
    //);

    ////t1 = MPI_Wtime();
        //printf("Time for local multiply: %lf sec\n", t1-t0);
//#endif

    //return 0;
//}

//// Suraj
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

