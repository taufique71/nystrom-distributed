#ifndef NYSTROM_H
#define NYSTROM_H

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include "matrix.h"

#ifdef USE_CUBLAS
	#include <cublas_v2.h>
	#include <cuda_runtime.h>
    #include <curand_kernel.h>
    #include <curand.h>

    #define CUDA_CHECK(call) { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(err); \
        } \
    }

    // CURAND API error checking
    #define CURAND_CHECK(err)                                                      \
    do {                                                                         \
        curandStatus_t err_ = (err);                                               \
        if (err_ != CURAND_STATUS_SUCCESS) {                                       \
            std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
            throw std::runtime_error("curand error");                                \
        }                                                                          \
    } while (0)
#else
	#include <mkl.h>
#endif

#include "procgrid.h"
#include "prng.h"
#include "utils.h"

//void nystrom_1d_noredist_1d(ParMat &A, int r, ParMat &Y, ParMat &Z){
    //double t0, t1, t2, t3;
    //int nproc, myrank;
    //MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    //MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    //ProcGrid grid1 = A.grid;
    //ProcGrid grid2 = Y.grid;

//#ifdef USE_CUBLAS
	//cublasHandle_t handle;
	//cublasCreate(&handle);
//#endif

    ////ParMat Y(A.nRowGlobal, r, grid, 'C');
    //if(myrank == 0) printf("matmul1 in %dx%dx%d grid\n", A.grid.nProcRow, A.grid.nProcCol, A.grid.nProcFib);
    
    //double* Omega = NULL;
    //{
        //t0 = MPI_Wtime();

        //Omega = new double[A.nColGlobal * r]; // Allocate for received B matrix
        //Xoroshiro128Plus prng(123456789, 987654321); // Defined in prng.cpp

        //for (size_t i = 0; i < A.nColGlobal * r; ++i) {
            //Omega[i] = prng.nextDouble();
        //}

        //t1 = MPI_Wtime();

        //if(myrank == 0){
            //printf("Time to generate Omega: %lf sec\n", t1-t0);
        //}
    //}

    //{
        //auto cblas_m = A.nRowLocal;
        //auto cblas_k = A.nColGlobal;
        //auto cblas_n = r;
        //auto cblas_alpha = 1.0;
        //auto cblas_beta = 0.0;
        //auto cblas_a = A.localMat;
        //auto cblas_lda = A.nRowLocal;
        //auto cblas_b = Omega;
        //auto cblas_ldb = A.nColGlobal;
        //auto cblas_c = Y.localMat; 
        //auto cblas_ldc = A.nRowLocal; 
                                         
        //t0 = MPI_Wtime();

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

        //t1 = MPI_Wtime();
        //if(myrank == 0){
            //printf("Time for first dgemm: %lf sec\n", t1-t0);
        //}
    //}

    //if(myrank == 0) printf("matmul2 in %dx%dx%d grid\n", Y.grid.nProcRow, Y.grid.nProcCol, Y.grid.nProcFib);

    //// grid1 and grid2 is same, so does not matter if grid1 and grid2 is used. Using grid2 because the name is relevant for matmul2
    //double* contribZ = new double[r*r];
    //int OmegaTColOffset = grid2.rowRank * (A.nColGlobal / grid2.nProcRow); // How many columns of BT needs to be moved forward
    //int OmegaTColCount = (grid2.rankInColWorld < grid2.nProcRow-1) ? (A.nColGlobal / grid2.nProcRow) : (A.nColGlobal - OmegaTColOffset) ;

    //{
        //auto cblas_m = r;
        //auto cblas_k = OmegaTColCount; // Number of columns of Omega-transpose to be used
        //auto cblas_n = Y.nColLocal;
        //auto cblas_alpha = 1.0;
        //auto cblas_beta = 0.0;
        //auto cblas_a = Omega + (A.nRowGlobal / grid2.nProcRow) * grid2.rowRank; // Move forward these many entries of Omega
        //auto cblas_lda = A.nColGlobal;
        //auto cblas_b = Y.localMat;
        //auto cblas_ldb = Y.nRowLocal;
        //auto cblas_c = contribZ; 
        //auto cblas_ldc = r; 
                                         
        //t0 = MPI_Wtime();

        //cblas_dgemm(
            //CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
            //CblasTrans, // A matrix is transpose. `transa` param of MKL cblas call.
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

        //t1 = MPI_Wtime();
        //if(myrank == 0){
            //printf("Time for second dgemm: %lf sec\n", t1-t0);
        //}
    //}

    ////ParMat Z(r, r, grid, 'B'); // B face for column split distrib of Z
    //{
        //int commSize = 0;
        //MPI_Comm_size(grid2.colWorld, &commSize); // colWorld - because we chose B face for distribution of Z

        //// Reduce scatter of Z along process grid column
        //int nColToRecv = Z.nColLocal; // Data structure is already prepared, just collect necessary information
        //int* nColToSend = new int[commSize]; // Change to commSize for column-wise collection
        
        //t0 = MPI_Wtime();

        //// Gather the number of columns from all processes in the column grid
        //MPI_Allgather(&nColToRecv, 1, MPI_INT, nColToSend, 1, MPI_INT, grid2.colWorld); // Update communicator

        //// Calculate the number of values to scatter based on the number of rows in Z
        //int* nValToSend = new int[commSize];
        //for (int i = 0; i < commSize; i++) {
            //nValToSend[i] = nColToSend[i] * Z.nRowLocal; // Update to use Z matrix
        //}

        //// Prepare displacements for the data to be scattered
        //int* sendDispls = new int[commSize + 1];
        //sendDispls[0] = 0;
        //std::partial_sum(nValToSend, nValToSend + commSize, sendDispls + 1);

        //// Scatter and reduce the relevant pieces of C
        //MPI_Reduce_scatter(contribZ, Z.localMat, nValToSend, MPI_DOUBLE, MPI_SUM, grid2.colWorld);

        //t1 = MPI_Wtime();
        //if(myrank == 0){
            //printf("Time to scatter and reduce Z: %lf sec\n", t1-t0);
        //}

        //// Clean up allocated memory
        //delete[] nColToSend;
        //delete[] nValToSend;
        //delete[] sendDispls;
    //}

    //delete[] Omega;
    ////delete[] multC;
    //delete[] contribZ;

    //return;

//}

void nystrom_1d_redist_1d(ParMat &A, int r, ParMat &Y, ParMat &Z){
    double t0, t1, t2, t3;
    double tGenOmega1=0.0, tDataMove1=0.0, tDgemm1=0.0, tAll2All=0.0, tUnpack=0.0;
    double tGenOmega2=0.0, tDataMove2=0.0, tDgemm2=0.0;

	int nproc, myrank;
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int n = A.nRowGlobal;

	ProcGrid grid1 = A.grid;
	ProcGrid grid2 = Y.grid;

#ifdef USE_CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
#endif

	// Compute temporary Y on grid1, then redistribute for matmul2 on grid2
	ParMat Ytemp(A.nRowGlobal, r, grid1, 'C');

	//if(myrank == 0) printf("matmul1 in %dx%dx%d grid\n", A.grid.nProcRow, A.grid.nProcCol, A.grid.nProcFib);
	
    double* Omega = NULL;
    {
        t0 = MPI_Wtime();
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&Omega), sizeof(double) * (n * r) ) );
        curandGenerator_t gen = NULL;
        curandRngType_t rng = CURAND_RNG_PSEUDO_XORWOW; 
        curandOrdering_t order = CURAND_ORDERING_PSEUDO_SEEDED;
        const unsigned long long offset = 0ULL;
        const unsigned long long seed = 1234ULL;

        CURAND_CHECK(curandCreateGenerator(&gen, rng));
        CURAND_CHECK(curandSetGeneratorOffset(gen, offset));
        CURAND_CHECK(curandSetGeneratorOrdering(gen, order));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
        CURAND_CHECK(curandGenerateUniformDouble(gen, Omega, n * r ));
#else
        Omega = new double[n * r]; // Allocate for received B matrix
        Xoroshiro128Plus prng(123456789, 987654321); // Defined in prng.cpp

        for (size_t i = 0; i < n * r; ++i) {
            Omega[i] = prng.nextDouble();
        }
#endif
        t1 = MPI_Wtime();
        tGenOmega1 = t1-t0;
    }

	{
		auto cblas_m = A.nRowLocal; // Number of rows of local A
		auto cblas_k = A.nColLocal; // Number of columns of local A. Local and global are expected to be same due to row distrib
		auto cblas_n = r; // Number of columns of Omega
		auto cblas_alpha = 1.0;
		auto cblas_beta = 0.0;
		auto cblas_a = A.localMat; // Local A
		auto cblas_lda = A.nRowLocal; // Stride length to iterate over the columns of local A
		auto cblas_b = Omega; // Entire Omega
		auto cblas_ldb = A.nColLocal; // Stride length to iterate over the columns of Omega
		auto cblas_c = Ytemp.localMat; // Local Ytemp
		auto cblas_ldc = A.nRowLocal; // Stride length to iterate over the columns of local A
										 
        t0 = MPI_Wtime();
#ifdef USE_CUBLAS
        cublasOperation_t transA = CUBLAS_OP_N;
        cublasOperation_t transB = CUBLAS_OP_N;

        cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
                    &cblas_alpha, cblas_a, cblas_lda, cblas_b, cblas_ldb,
                    &cblas_beta, cblas_c, cblas_ldc);
#else

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

#endif
        t1 = MPI_Wtime();
        tDgemm1 = (t1-t0);
	}

	//if(myrank == 0) printf("matmul2 in %dx%dx%d grid\n", Y.grid.nProcRow, Y.grid.nProcCol, Y.grid.nProcFib);
	{
		t0 = MPI_Wtime();

		t2 = MPI_Wtime();

		int commSize = 0;
		MPI_Comm_size(grid2.fibWorld, &commSize); // Row world because, column distribution

		// TODO: Change according to Scalapack distribution
		// Y has row-wise distribution. Now change to columns-wise distribution on second grid
		int nColPerProc = Ytemp.nColLocal / commSize; // Target number of columns by process other than the last process
		int *nColToSend = new int[commSize]; // Number of columns to send to each process
		for (int p = 0; p < commSize; p++){
			if ( p < (commSize - 1) ) nColToSend[p] = nColPerProc;
			else nColToSend[p] = Ytemp.nColGlobal - nColPerProc * p;
		}

		int *nValToSend = new int[commSize]; // Number of values to send to each process
		for (int p = 0; p < commSize; p++){
			nValToSend[p] = nColToSend[p] * Ytemp.nRowLocal;
		}

		int* sendDispls = new int[commSize + 1];
		sendDispls[0] = 0;
		std::partial_sum(nValToSend, nValToSend + commSize, sendDispls + 1);

		int nRowPerProc = Ytemp.nRowGlobal / commSize; // Number of rows by process other than the last process
		int *nRowToRecv = new int[commSize]; // Number of rows to receive from each process
		for (int p = 0; p < commSize; p++){
			if ( p < (commSize - 1) ) nRowToRecv[p] = nRowPerProc;
			else nRowToRecv[p] = Ytemp.nRowGlobal - nRowPerProc * p;
		}

		int *nValToRecv = new int[commSize]; // Number of values to receive from each process
		for (int p = 0; p < commSize; p++){
			nValToRecv[p] = nRowToRecv[p] * nColToSend[grid2.rankInFibWorld];
		}

		int* recvDispls = new int[commSize + 1];
		recvDispls[0] = 0;
		std::partial_sum(nValToRecv, nValToRecv + commSize, recvDispls + 1);
		
		double* recvBuff = NULL;
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&recvBuff), sizeof(double) * (recvDispls[commSize]) ) );
#else
		recvBuff = new double[recvDispls[commSize]];
#endif
		t3 = MPI_Wtime();
		double tBuffPrep = t3-t2;
		
		// Alltoallv
		//t0 = MPI_Wtime();
		MPI_Alltoallv(Ytemp.localMat, nValToSend, sendDispls, MPI_DOUBLE,
				   recvBuff, nValToRecv, recvDispls, MPI_DOUBLE,
				   grid2.fibWorld);
		t1 = MPI_Wtime();
		tAll2All = t1-t0;

		// Unpacking
		t2 = MPI_Wtime();
		for(int c = 0; c < nColToSend[grid2.rankInFibWorld]; c++){
			size_t offset = 0;
			for(int p = 0; p < commSize; p++){
#ifdef USE_CUBLAS
                CUDA_CHECK( cudaMemcpy(
                        Y.localMat + c * Y.nRowLocal + offset, 
                        recvBuff + recvDispls[grid2.rankInFibWorld] + c * nRowToRecv[p], 
                        sizeof(double)*nRowToRecv[p], 
                        cudaMemcpyDeviceToDevice) );
#else
				memcpy(Y.localMat + c * Y.nRowLocal + offset, 
						recvBuff + recvDispls[grid2.rankInFibWorld] + c * nRowToRecv[p], 
						sizeof(double)*nRowToRecv[p]
				);
#endif
				offset += nRowToRecv[p];
			}
		}
		t3 = MPI_Wtime();
		tUnpack = t3-t2;

		delete[] nColToSend;
		delete[] nValToSend;
		delete[] sendDispls;
		delete[] nRowToRecv;
		delete[] nValToRecv;
		delete[] recvDispls;
#ifdef USE_CUBLAS
		CUDA_CHECK(cudaFree(recvBuff));
#else
		delete[] recvBuff;
#endif

		//t1 = MPI_Wtime();
	}

	{
		auto cblas_m = r; // Number of rows of transposed Omega
		auto cblas_k = Y.nRowLocal; // Number of columns of transposed Omega / number of rows of local Y after redistribution
		auto cblas_n = Z.nColLocal; // Number of columns of local Y / number of columns of local Z
		auto cblas_alpha = 1.0;
		auto cblas_beta = 0.0;
		auto cblas_a = Omega; // Entire Omega
		auto cblas_lda = Y.nRowLocal; // Stride length to iterate over columns of transposed Omega
									  // which is equal to the number of rows of local Y after redistribution
		auto cblas_b = Y.localMat; // Local Y
		auto cblas_ldb = Y.nRowLocal; // Stride length to iterate over columns of local Y after redistribution
		auto cblas_c = Z.localMat; // Local Z
		auto cblas_ldc = r; // Stride length to iterate over columns of local Z
										 
		t0 = MPI_Wtime();

#ifdef USE_CUBLAS
        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;

        cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
                    &cblas_alpha, cblas_a, cblas_lda, cblas_b, cblas_ldb,
                    &cblas_beta, cblas_c, cblas_ldc);

#else
		cblas_dgemm(
			CblasColMajor, // Column major order. `layout` parameter of MKL cblas call.
			CblasTrans, // A matrix is transpose. `transa` param of MKL cblas call.
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
#endif

		t1 = MPI_Wtime();
		tDgemm2 = t1-t0;
	}

#ifdef USE_CUBLAS
    CUDA_CHECK(cudaFree(Omega));
	cublasDestroy(handle);
#else
	delete[] Omega;
#endif

    double tGenOmega1_max=0.0, tDataMove1_max=0.0, tDgemm1_max=0.0, tUnpack_max=0.0, tAll2All_max=0.0;
    double tGenOmega1_min=0.0, tDataMove1_min=0.0, tDgemm1_min=0.0, tUnpack_min=0.0, tAll2All_min=0.0;
    double tGenOmega2_max=0.0, tDataMove2_max=0.0, tDgemm2_max=0.0;
    double tGenOmega2_min=0.0, tDataMove2_min=0.0, tDgemm2_min=0.0;
	
	MPI_Allreduce(&tGenOmega1, &tGenOmega1_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tGenOmega1, &tGenOmega1_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tDataMove1, &tDataMove1_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tDataMove1, &tDataMove1_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tDgemm1, &tDgemm1_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tDgemm1, &tDgemm1_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tAll2All, &tAll2All_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tAll2All, &tAll2All_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tUnpack, &tUnpack_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tUnpack, &tUnpack_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tDgemm2, &tDgemm2_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tDgemm2, &tDgemm2_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

	if(myrank == 0){
        printf("Time to generate Omega: %lf sec\n", tGenOmega1_max);
        printf("Time for first dgemm: %lf sec\n", tDgemm1_max);
        printf("Time for all2all: %lf sec\n", tAll2All_max);
        printf("Time to unpack: %lf sec\n", tUnpack_max);
        printf("Time for second dgemm: %lf sec\n", tDgemm2_max);
	}

	return;
}

void nystrom_2d_redist_1d_redundant(ParMat &A, int r, ParMat &Y, ParMat &Z){
    double t0, t1, t2, t3;
    double tGenOmega1=0.0, tDataMove1=0.0, tDgemm1=0.0, tPack1=0.0, tReduceScatter1=0.0;
    double tGenOmega2=0.0, tDataMove2=0.0, tDgemm2=0.0, tReduceScatter2=0.0;
    int nproc, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    ProcGrid grid1 = A.grid;
    ProcGrid grid2 = Y.grid;

#ifdef USE_CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
#endif
    
    int n = A.nColGlobal;

    //if(myrank == 0) printf("matmul1 in %dx%dx%d grid\n", grid1.nProcRow, grid1.nProcCol, grid1.nProcFib);
    
    double* Omega = NULL;
    {
        t0 = MPI_Wtime();
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&Omega), sizeof(double) * (n * r) ) );
        curandGenerator_t gen = NULL;
        curandRngType_t rng = CURAND_RNG_PSEUDO_XORWOW; 
        curandOrdering_t order = CURAND_ORDERING_PSEUDO_SEEDED;
        const unsigned long long offset = 0ULL;
        const unsigned long long seed = 1234ULL;

        CURAND_CHECK(curandCreateGenerator(&gen, rng));
        CURAND_CHECK(curandSetGeneratorOffset(gen, offset));
        CURAND_CHECK(curandSetGeneratorOrdering(gen, order));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
        CURAND_CHECK(curandGenerateUniformDouble(gen, Omega, n * r ));
#else
        Omega = new double[n * r]; // Allocate for received B matrix
        Xoroshiro128Plus prng(123456789, 987654321); // Defined in prng.cpp

        for (size_t i = 0; i < n * r; ++i) {
            Omega[i] = prng.nextDouble();
        }
#endif
        t1 = MPI_Wtime();
        tGenOmega1 = t1-t0;
    }

    // Each process would generate [A.nRowLocal x r] partial output
    double* Ytemp = NULL;
    {
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&Ytemp), sizeof(double) * (A.nRowLocal * r) ) );
#else
        Ytemp = new double[A.nRowLocal * r];
#endif
        
        auto cblas_m = A.nRowLocal;
        auto cblas_k = A.nColLocal;
        auto cblas_n = r;
        auto cblas_alpha = 1.0;
        auto cblas_beta = 0.0;
        auto cblas_a = A.localMat;
        auto cblas_lda = A.nRowLocal; // Stride size to access the entry at the same row of the next column of the local copy of A
                                      // It's obvious that it would be the number of rows of the local copy of A
        auto cblas_b = Omega + A.colDispls[A.colRank]; // Starting location of Omega
                                                       // Would change depending on the col rank of the A matrix
        auto cblas_ldb = n; // Stride size to access the entry at the same row of the next column of Omega
                            // Because we are generating a copy of entire Omega at every process
                            // It would be `n`
        auto cblas_c = Ytemp; 
        auto cblas_ldc = cblas_lda; // Number of rows of the local copy of the partial result
                                    // Same as the number of rows of the local copy of A

        t0 = MPI_Wtime();
#ifdef USE_CUBLAS
        cublasOperation_t transA = CUBLAS_OP_N;
        cublasOperation_t transB = CUBLAS_OP_N;

        cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
                    &cblas_alpha, cblas_a, cblas_lda, cblas_b, cblas_ldb,
                    &cblas_beta, cblas_c, cblas_ldc);
#else

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

#endif
        t1 = MPI_Wtime();
        tDgemm1 = (t1-t0);
    }

    {
        t0 = MPI_Wtime();
        int* sendrows = new int[A.nProcCol];
        int* sendrowsDispls = new int[A.nProcCol + 1];
        findSplits( A.nRowLocal, A.nProcCol, sendrows ); // Defined in utils.h
        sendrowsDispls[0] = 0;
        std::partial_sum(sendrows, sendrows+A.nProcCol, sendrowsDispls+1); 

        int* sendcnt = new int[A.nProcCol];
        int* sdispls = new int[A.nProcCol + 1];
        int* recvcnt = new int[A.nProcCol];
        int* rdispls = new int[A.nProcCol + 1];
        double* sendbuff;
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&sendbuff), sizeof(double) * (A.nRowLocal * r) ) );
#else
        sendbuff = new double[A.nRowLocal * r];
#endif
        for(int i = 0; i < A.nProcCol; i++){
            sendcnt[i] = sendrows[i] * r;
            recvcnt[i] = sendrows[A.colRank] * r;
        }
        sdispls[0] = 0;
        std::partial_sum(sendcnt, sendcnt+A.nProcCol, sdispls+1); 
        rdispls[0] = 0;
        std::partial_sum(recvcnt, recvcnt+A.nProcCol, rdispls+1); 
        for (int i = 0; i < r; i++){
            for(int p = 0; p < A.nProcCol; p++){
#ifdef USE_CUBLAS
                CUDA_CHECK( cudaMemcpy(
                        sendbuff + sdispls[p] + (sendrows[p] * i), 
                        Ytemp + (A.nRowLocal * i) + sendrowsDispls[p], 
                        sizeof(double) * (sendrows[p]), 
                        cudaMemcpyDeviceToDevice) );
#else
                memcpy(sendbuff + sdispls[p] + (sendrows[p] * i), 
                        Ytemp + (A.nRowLocal * i) + sendrowsDispls[p], 
                        sizeof(double) * (sendrows[p]) );
#endif
            }
        }
        t1 = MPI_Wtime();
        tPack1 = (t1-t0);

        t0 = MPI_Wtime();
        MPI_Reduce_scatter(sendbuff, Y.localMat, recvcnt, MPI_DOUBLE, MPI_SUM, grid2.colWorld);
        t1 = MPI_Wtime();
        tReduceScatter1 = (t1-t0);

#ifdef USE_CUBLAS
        CUDA_CHECK(cudaFree(sendbuff));
#else
        delete [] sendbuff;
#endif

        delete [] sendrows;
        delete [] sendrowsDispls;
        delete [] sendcnt;
        delete [] sdispls;
        delete [] recvcnt;
        delete [] rdispls;
    }

    double* Ztemp = NULL;
    {
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&Ztemp), sizeof(double) * (r * r) ) );
#else
        Ztemp = new double[r * r];
#endif

        auto cblas_m = r;
        auto cblas_k = Y.nRowLocal;
        auto cblas_n = r;
        auto cblas_alpha = 1.0;
        auto cblas_beta = 0.0;
        auto cblas_a = Omega + Y.rowDispls[Y.rowRank]; // Starting position of the first column
                                                       // Because Omega.T would be used effectively, and Omega is in col major
                                                       // We need to move ahead in the first column as many rows as it needs to get to
                                                       // the row relevant for local Y
                                                       // Would change depending on the row rank of Y matrix
        auto cblas_lda = n; // Stride size to access the entry at the same row of the next column of Omega
                            // Because we store entire Omega, this value would be n as Omega is nxr matrix
        auto cblas_b = Y.localMat; // Starting location of local Y
        auto cblas_ldb = Y.nRowLocal; // Stride size to access the entry at the same row of the next column of local Y
        auto cblas_c = Ztemp; 
        auto cblas_ldc = r; // Number of rows of the local copy of the partial result
                            // Same as the number of rows of Omega.T which is r

        t0 = MPI_Wtime();
#ifdef USE_CUBLAS
        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;

        cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
                    &cblas_alpha, cblas_a, cblas_lda, cblas_b, cblas_ldb,
                    &cblas_beta, cblas_c, cblas_ldc);
#else

        cblas_dgemm(
            CblasColMajor, // Column major order. `Layout` parameter of MKL cblas call.
            CblasTrans, // A matrix is transpose. `transa` param of MKL cblas call.
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

#endif
        t1 = MPI_Wtime();
        tDgemm2 = (t1-t0);
    }

    {
        // Reduce and scatter partial results along the grid fiber in the context of Z
        int* recvcnt = new int [Z.nProcFib];
        MPI_Allgather(&Z.nColLocal, 1, MPI_INT, recvcnt, 1, MPI_INT, Z.fibWorld);
        std::transform(recvcnt, recvcnt + Z.nProcFib, recvcnt, [r](double colcnt) { return (colcnt) * r; });
        int sum = std::accumulate(recvcnt, recvcnt + Z.nProcFib, 0);
        assert(sum == r*r);

        t0 = MPI_Wtime();
        MPI_Reduce_scatter(Ztemp, Z.localMat, recvcnt, MPI_DOUBLE, MPI_SUM, Z.fibWorld);
        t1 = MPI_Wtime();
        tReduceScatter2 = (t1-t0);

        delete [] recvcnt;
    }


#ifdef USE_CUBLAS
    CUDA_CHECK(cudaFree(Omega));
    CUDA_CHECK(cudaFree(Ytemp));
    CUDA_CHECK(cudaFree(Ztemp));
	cublasDestroy(handle);
#else
    delete[] Omega;
    delete[] Ytemp;
    delete[] Ztemp;
#endif

    double tGenOmega1_max=0.0, tDataMove1_max=0.0, tDgemm1_max=0.0, tPack1_max=0.0, tReduceScatter1_max=0.0;
    double tGenOmega1_min=0.0, tDataMove1_min=0.0, tDgemm1_min=0.0, tPack1_min=0.0, tReduceScatter1_min=0.0;
    double tGenOmega2_max=0.0, tDataMove2_max=0.0, tDgemm2_max=0.0, tReduceScatter2_max=0.0;
    double tGenOmega2_min=0.0, tDataMove2_min=0.0, tDgemm2_min=0.0, tReduceScatter2_min=0.0;
	
	MPI_Allreduce(&tGenOmega1, &tGenOmega1_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tGenOmega1, &tGenOmega1_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tDataMove1, &tDataMove1_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tDataMove1, &tDataMove1_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tDgemm1, &tDgemm1_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tDgemm1, &tDgemm1_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tPack1, &tPack1_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tPack1, &tPack1_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tReduceScatter1, &tReduceScatter1_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tReduceScatter1, &tReduceScatter1_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tDgemm2, &tDgemm2_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tDgemm2, &tDgemm2_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&tReduceScatter2, &tReduceScatter2_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&tReduceScatter2, &tReduceScatter2_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

	if(myrank == 0){
        printf("Time to generate Omega: %lf sec\n", tGenOmega1_max);
        printf("Time for first dgemm: %lf sec\n", tDgemm1_max);
        printf("Time to pack for reduce-scatter: %lf sec\n", tPack1_max);
        printf("Time for first reduce-scatter: %lf sec\n", tReduceScatter1_max);
        printf("Time for second dgemm: %lf sec\n", tDgemm2_max);
        printf("Time for second reduce-scatter: %lf sec\n", tReduceScatter2_max);
	}

    return;
}


#endif

