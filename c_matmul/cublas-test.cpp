#include <numeric>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cassert>

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
}
#else
#include <mkl.h>
#endif

//#include "procgrid.h"
#include "prng.h"
using namespace std;

int main(int argc, char* argv[]) {

	double t0, t1, t2, t3;
	
    int cblas_m = 10000;
    int cblas_k = 1000;
    int cblas_n = 5000;
    double cblas_alpha = 1.0;
    double cblas_beta = 0.0;
    double* cblas_a = new double[cblas_m * cblas_k];
    int cblas_lda = 10000;
    double* cblas_b = new double[cblas_k * cblas_n];
    int cblas_ldb = 1000;
    double* cblas_c = new double[cblas_m * cblas_n]; 
    int cblas_ldc = 10000;
	
	//std::memset(cblas_a, 1.0, sizeof(double) * cblas_m * cblas_k);
	//std::memset(cblas_b, 2.0, sizeof(double) * cblas_k * cblas_n);
	for (int i = 0; i < cblas_m * cblas_k; i++) cblas_a[i] = (double)(i);
	for (int i = 0; i < cblas_k * cblas_n; i++) cblas_b[i] = (double)(i);
	std::memset(cblas_c, 0, sizeof(double) * cblas_m * cblas_n);
	//for (int i = 0; i < cblas_m * cblas_n; i++) cblas_c = (double)(i);

#ifdef USE_CUBLAS
	double tMemMove = 0;
	double tDgemm = 0;

	//t0 = MPI_Wtime();
	double *d_A, *d_B, *d_C;
    cudaError_t err;
	err = cudaMalloc(&d_A, sizeof(double) * cblas_lda * cblas_k);
	cudaMalloc(&d_B, sizeof(double) * cblas_ldb * cblas_n);
	cudaMalloc(&d_C, sizeof(double) * cblas_ldc * cblas_n);
	cudaMemcpy(d_A, cblas_a, sizeof(double) * cblas_lda * cblas_k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, cblas_b, sizeof(double) * cblas_ldb * cblas_n, cudaMemcpyHostToDevice);
	cudaMemset(d_C, 0, sizeof(double) * cblas_ldc * cblas_n);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasOperation_t transA = CUBLAS_OP_N;
	cublasOperation_t transB = CUBLAS_OP_N;
	//t1 = MPI_Wtime();
	tMemMove += (t1-t0);

	//t0 = MPI_Wtime();
	cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
				&cblas_alpha, d_A, cblas_lda, d_B, cblas_ldb,
				&cblas_beta, d_C, cblas_ldc);
	//t1 = MPI_Wtime();
	tDgemm += (t1-t0);

	//t0 = MPI_Wtime();
	cudaMemcpy(cblas_c, d_C, sizeof(double) * cblas_ldc * cblas_n, cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	//t1 = MPI_Wtime();
	tMemMove += (t1-t0);

		printf("Time for local multiply host-device mem movement: %lf sec\n", tMemMove);
		printf("Time for local multiply: %lf sec\n", tDgemm);
#else
                                     
    //t0 = MPI_Wtime();

    cblas_dgemm(
        CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
        CblasNoTrans, // A matrix is not transpose. `transa` param of MKL cblas call.
        CblasNoTrans, // B matrix is not transpose. `transb` param of MKL cblas call.
        cblas_m, // Number of rows of A or C. `m` param of MKL cblas call.
        cblas_n, // Number of cols of B or C. `n` param of MKL cblas call.
        cblas_k, // Inner dimension - number of columns of A or number of rows of B. `k` param of MKL cblas call.
        cblas_alpha, // Scalar `alpha` param of MKL cblas call.
        cblas_a, // Data buffer of A. `a` param of MKL cblas call.
        cblas_lda, // Leading dimension of A. `lda` param of MKL cblas call.
        cblas_b, // Data buffer of B. `b` param of MKL cblas call.
        cblas_ldb, // Leading dimension of B. `ldb` param of MKL cblas call.
        cblas_beta, // Scalar `beta` param of MKL cblas call.
        cblas_c, // Data buffer of C. `c` param of MKL cblas call.
        cblas_ldc // Leading dimension of C. `ldc` param of MKL cblas call.
    );

    //t1 = MPI_Wtime();
        printf("Time for local multiply: %lf sec\n", t1-t0);
#endif

    return 0;
}
