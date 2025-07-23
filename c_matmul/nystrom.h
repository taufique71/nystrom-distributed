#ifndef NYSTROM_H
#define NYSTROM_H

//#include <mpi.h>
//#include <iostream>
//#include <vector>
//#include <cassert>
//#include <numeric>
//#include <algorithm>
//#include <cstring>
//#include <omp.h>
//#include <mkl.h>
//#include "procgrid.h"
//#include "matrix.h"
//#include "prng.h"

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <omp.h>
#if defined(USE_CUBLAS)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#else
#include <mkl.h>
#endif
#include "procgrid.h"
#include "matrix.h"
#include "prng.h"

void nystrom_1d_noredist_1d(ParMat &A, int r, ParMat &Y, ParMat &Z){
    double t0, t1, t2, t3;
    int nproc, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    ProcGrid grid1 = A.grid;
    ProcGrid grid2 = Y.grid;
    //ParMat Y(A.nRowGlobal, r, grid, 'C');
    if(myrank == 0) printf("matmul1 in %dx%dx%d grid\n", A.grid.nProcRow, A.grid.nProcCol, A.grid.nProcFib);
    
    double* Omega = NULL;
    {
        t0 = MPI_Wtime();

        Omega = new double[A.nColGlobal * r]; // Allocate for received B matrix
        Xoroshiro128Plus prng(123456789, 987654321); // Defined in prng.cpp

        for (size_t i = 0; i < A.nColGlobal * r; ++i) {
            Omega[i] = prng.nextDouble();
        }

        t1 = MPI_Wtime();

        if(myrank == 0){
            printf("Time to generate Omega: %lf sec\n", t1-t0);
        }
    }

    {
        auto cblas_m = A.nRowLocal;
        auto cblas_k = A.nColGlobal;
        auto cblas_n = r;
        auto cblas_alpha = 1.0;
        auto cblas_beta = 0.0;
        auto cblas_a = A.localMat;
        auto cblas_lda = A.nRowLocal;
        auto cblas_b = Omega;
        auto cblas_ldb = A.nColGlobal;
        auto cblas_c = Y.localMat; 
        auto cblas_ldc = A.nRowLocal; 

#if defined(USE_CUBLAS)
		double tMemMove = 0;
		double tDgemm = 0;
		t0 = MPI_Wtime();
		double *d_A, *d_B, *d_C;
		cudaMalloc((void**)&d_A, sizeof(double) * cblas_lda * cblas_k);
		cudaMalloc((void**)&d_B, sizeof(double) * cblas_ldb * cblas_n);
		cudaMalloc((void**)&d_C, sizeof(double) * cblas_ldc * cblas_n);
		cudaMemcpy(d_A, cblas_a, sizeof(double) * cblas_lda * cblas_k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, cblas_b, sizeof(double) * cblas_ldb * cblas_n, cudaMemcpyHostToDevice);
		cudaMemset(d_C, 0, sizeof(double) * cblas_ldc * cblas_n);
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasOperation_t transA = CUBLAS_OP_N;
		cublasOperation_t transB = CUBLAS_OP_N;
		t1 = MPI_Wtime();
		tMemMove += (t1-t0);
		

		t0 = MPI_Wtime();
		cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
					&cblas_alpha, d_A, cblas_lda, d_B, cblas_ldb,
					&cblas_beta, d_C, cblas_ldc);
		t1 = MPI_Wtime();
		tDgemm += (t1-t0);

		t0 = MPI_Wtime();
		cudaMemcpy(cblas_c, d_C, sizeof(double) * cblas_ldc * cblas_n, cudaMemcpyDeviceToHost);
		cublasDestroy(handle);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		t1 = MPI_Wtime();
		tMemMove += (t1-t0);

        if(myrank == 0){
            printf("Time for first dgemm host-device mem movement: %lf sec\n", tMemMove);
            printf("Time for first dgemm: %lf sec\n", tDgemm);
        }
#else
        t0 = MPI_Wtime();

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

        t1 = MPI_Wtime();
        if(myrank == 0){
            printf("Time for first dgemm: %lf sec\n", t1-t0);
        }
#endif
    }

    if(myrank == 0) printf("matmul2 in %dx%dx%d grid\n", Y.grid.nProcRow, Y.grid.nProcCol, Y.grid.nProcFib);

    // grid1 and grid2 is same, so does not matter if grid1 and grid2 is used. Using grid2 because the name is relevant for matmul2
    double* contribZ = new double[r*r];
    int OmegaTColOffset = grid2.rowRank * (A.nColGlobal / grid2.nProcRow); // How many columns of BT needs to be moved forward
    int OmegaTColCount = (grid2.rankInColWorld < grid2.nProcRow-1) ? (A.nColGlobal / grid2.nProcRow) : (A.nColGlobal - OmegaTColOffset) ;

    {
        auto cblas_m = r;
        auto cblas_k = OmegaTColCount; // Number of columns of Omega-transpose to be used
        auto cblas_n = Y.nColLocal;
        auto cblas_alpha = 1.0;
        auto cblas_beta = 0.0;
        auto cblas_a = Omega + (A.nRowGlobal / grid2.nProcRow) * grid2.rowRank; // Move forward these many entries of Omega
        auto cblas_lda = A.nColGlobal;
        auto cblas_b = Y.localMat;
        auto cblas_ldb = Y.nRowLocal;
        auto cblas_c = contribZ; 
        auto cblas_ldc = r; 

#if defined(USE_CUBLAS)
		double tMemMove = 0;
		double tDgemm = 0;
		t0 = MPI_Wtime();
		double *d_A, *d_B, *d_C;
		cudaMalloc((void**)&d_A, sizeof(double) * cblas_lda * cblas_k);
		cudaMalloc((void**)&d_B, sizeof(double) * cblas_ldb * cblas_n);
		cudaMalloc((void**)&d_C, sizeof(double) * cblas_ldc * cblas_n);
		cudaMemcpy(d_A, cblas_a, sizeof(double) * cblas_lda * cblas_k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, cblas_b, sizeof(double) * cblas_ldb * cblas_n, cudaMemcpyHostToDevice);
		cudaMemset(d_C, 0, sizeof(double) * cblas_ldc * cblas_n);
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasOperation_t transA = CUBLAS_OP_N;
		cublasOperation_t transB = CUBLAS_OP_N;
		t1 = MPI_Wtime();
		tMemMove += (t1-t0);
		

		t0 = MPI_Wtime();
		cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
					&cblas_alpha, d_A, cblas_lda, d_B, cblas_ldb,
					&cblas_beta, d_C, cblas_ldc);
		t1 = MPI_Wtime();
		tDgemm += (t1-t0);

		t0 = MPI_Wtime();
		cudaMemcpy(cblas_c, d_C, sizeof(double) * cblas_ldc * cblas_n, cudaMemcpyDeviceToHost);
		cublasDestroy(handle);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		t1 = MPI_Wtime();
		tMemMove += (t1-t0);

        if(myrank == 0){
            printf("Time for second dgemm host-device mem movement: %lf sec\n", tMemMove);
            printf("Time for second dgemm: %lf sec\n", tDgemm);
        }
#else
        t0 = MPI_Wtime();

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

        t1 = MPI_Wtime();
        if(myrank == 0){
            printf("Time for second dgemm: %lf sec\n", t1-t0);
        }
#endif
    }

    //ParMat Z(r, r, grid, 'B'); // B face for column split distrib of Z
    {
        int commSize = 0;
        MPI_Comm_size(grid2.colWorld, &commSize); // colWorld - because we chose B face for distribution of Z

        // Reduce scatter of Z along process grid column
        int nColToRecv = Z.nColLocal; // Data structure is already prepared, just collect necessary information
        int* nColToSend = new int[commSize]; // Change to commSize for column-wise collection
        
        t0 = MPI_Wtime();

        // Gather the number of columns from all processes in the column grid
        MPI_Allgather(&nColToRecv, 1, MPI_INT, nColToSend, 1, MPI_INT, grid2.colWorld); // Update communicator

        // Calculate the number of values to scatter based on the number of rows in Z
        int* nValToSend = new int[commSize];
        for (int i = 0; i < commSize; i++) {
            nValToSend[i] = nColToSend[i] * Z.nRowLocal; // Update to use Z matrix
        }

        // Prepare displacements for the data to be scattered
        int* sendDispls = new int[commSize + 1];
        sendDispls[0] = 0;
        std::partial_sum(nValToSend, nValToSend + commSize, sendDispls + 1);

        // Scatter and reduce the relevant pieces of C
        MPI_Reduce_scatter(contribZ, Z.localMat, nValToSend, MPI_DOUBLE, MPI_SUM, grid2.colWorld);

        t1 = MPI_Wtime();
        if(myrank == 0){
            printf("Time to scatter and reduce Z: %lf sec\n", t1-t0);
        }

        // Clean up allocated memory
        delete[] nColToSend;
        delete[] nValToSend;
        delete[] sendDispls;
    }

    delete[] Omega;
    //delete[] multC;
    delete[] contribZ;

    return;

}

void nystrom_1d_redist_1d(ParMat &A, int r, ParMat &Y, ParMat &Z){
    double t0, t1, t2, t3;
    int nproc, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    ProcGrid grid1 = A.grid;
    ProcGrid grid2 = Y.grid;

    // Compute temporary Y on grid1, then redistribute for matmul2 on grid2
    ParMat Ytemp(A.nRowGlobal, r, grid1, 'C');

    if(myrank == 0) printf("matmul1 in %dx%dx%d grid\n", A.grid.nProcRow, A.grid.nProcCol, A.grid.nProcFib);
    
    double* Omega = NULL;
    
    {
        t0 = MPI_Wtime();

        Omega = new double[A.nColGlobal * r]; // Allocate for received B matrix
        Xoroshiro128Plus prng(123456789, 987654321); // Defined in prng.cpp

        for (size_t i = 0; i < A.nColGlobal * r; ++i) {
            Omega[i] = prng.nextDouble();
        }

        t1 = MPI_Wtime();

        if(myrank == 0){
            printf("Time to generate Omega: %lf sec\n", t1-t0);
        }
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

#if defined(USE_CUBLAS)
		double tMemMove = 0;
		double tDgemm = 0;
		t0 = MPI_Wtime();
		double *d_A, *d_B, *d_C;
		cudaMalloc((void**)&d_A, sizeof(double) * cblas_lda * cblas_k);
		cudaMalloc((void**)&d_B, sizeof(double) * cblas_ldb * cblas_n);
		cudaMalloc((void**)&d_C, sizeof(double) * cblas_ldc * cblas_n);
		cudaMemcpy(d_A, cblas_a, sizeof(double) * cblas_lda * cblas_k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, cblas_b, sizeof(double) * cblas_ldb * cblas_n, cudaMemcpyHostToDevice);
		cudaMemset(d_C, 0, sizeof(double) * cblas_ldc * cblas_n);
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasOperation_t transA = CUBLAS_OP_N;
		cublasOperation_t transB = CUBLAS_OP_N;
		t1 = MPI_Wtime();
		tMemMove += (t1-t0);
		

		t0 = MPI_Wtime();
		cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
					&cblas_alpha, d_A, cblas_lda, d_B, cblas_ldb,
					&cblas_beta, d_C, cblas_ldc);
		t1 = MPI_Wtime();
		tDgemm += (t1-t0);

		t0 = MPI_Wtime();
		cudaMemcpy(cblas_c, d_C, sizeof(double) * cblas_ldc * cblas_n, cudaMemcpyDeviceToHost);
		cublasDestroy(handle);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		t1 = MPI_Wtime();
		tMemMove += (t1-t0);

        if(myrank == 0){
            printf("Time for first dgemm host-device mem movement: %lf sec\n", tMemMove);
            printf("Time for first dgemm: %lf sec\n", tDgemm);
        }
#else
                                         
        t0 = MPI_Wtime();

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

        t1 = MPI_Wtime();
        if(myrank == 0){
            printf("Time for first dgemm: %lf sec\n", t1-t0);
        }
#endif
    }

    if(myrank == 0) printf("matmul2 in %dx%dx%d grid\n", Y.grid.nProcRow, Y.grid.nProcCol, Y.grid.nProcFib);
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

        double * recvBuff = new double[recvDispls[commSize]];
        t3 = MPI_Wtime();
        double tBuffPrep = t3-t2;
        
        // Alltoallv
        t2 = MPI_Wtime();
        MPI_Alltoallv(Ytemp.localMat, nValToSend, sendDispls, MPI_DOUBLE,
                   recvBuff, nValToRecv, recvDispls, MPI_DOUBLE,
                   grid2.fibWorld);
        t3 = MPI_Wtime();
        double tAlltoallv = t3-t2;

		// Unpacking
        t2 = MPI_Wtime();
        for(int c = 0; c < nColToSend[grid2.rankInFibWorld]; c++){
            size_t offset = 0;
            for(int p = 0; p < commSize; p++){
                memcpy(Y.localMat + c * Y.nRowLocal + offset, 
                        recvBuff + recvDispls[grid2.rankInFibWorld] + c * nRowToRecv[p], 
                        sizeof(double)*nRowToRecv[p]
                );
                offset += nRowToRecv[p];
            }
        }
        t3 = MPI_Wtime();
        double tUnpack = t3-t2;

        delete[] nColToSend;
        delete[] nValToSend;
        delete[] sendDispls;
        delete[] nRowToRecv;
        delete[] nValToRecv;
        delete[] recvDispls;
        delete[] recvBuff;

        t1 = MPI_Wtime();

        if(myrank == 0){
            printf("Time to redistribute Y: %lf sec\n", t1-t0);
            printf("\tTime to prepare buffer for alltoallv: %lf sec\n", tBuffPrep);
            printf("\tTime to do alltoallv: %lf sec\n", tAlltoallv);
            printf("\tTime to unpack: %lf sec\n", tUnpack);
        }
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

#if defined(USE_CUBLAS)
		double tMemMove = 0;
		double tDgemm = 0;
		t0 = MPI_Wtime();
		double *d_A, *d_B, *d_C;
		cudaMalloc((void**)&d_A, sizeof(double) * cblas_lda * cblas_k);
		cudaMalloc((void**)&d_B, sizeof(double) * cblas_ldb * cblas_n);
		cudaMalloc((void**)&d_C, sizeof(double) * cblas_ldc * cblas_n);
		cudaMemcpy(d_A, cblas_a, sizeof(double) * cblas_lda * cblas_k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, cblas_b, sizeof(double) * cblas_ldb * cblas_n, cudaMemcpyHostToDevice);
		cudaMemset(d_C, 0, sizeof(double) * cblas_ldc * cblas_n);
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasOperation_t transA = CUBLAS_OP_N;
		cublasOperation_t transB = CUBLAS_OP_N;
		t1 = MPI_Wtime();
		tMemMove += (t1-t0);
		

		t0 = MPI_Wtime();
		cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
					&cblas_alpha, d_A, cblas_lda, d_B, cblas_ldb,
					&cblas_beta, d_C, cblas_ldc);
		t1 = MPI_Wtime();
		tDgemm += (t1-t0);

		t0 = MPI_Wtime();
		cudaMemcpy(cblas_c, d_C, sizeof(double) * cblas_ldc * cblas_n, cudaMemcpyDeviceToHost);
		cublasDestroy(handle);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		t1 = MPI_Wtime();
		tMemMove += (t1-t0);

        if(myrank == 0){
            printf("Time for second dgemm host-device mem movement: %lf sec\n", tMemMove);
            printf("Time for second dgemm: %lf sec\n", tDgemm);
        }
#else
                                         
        t0 = MPI_Wtime();

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

        t1 = MPI_Wtime();
        if(myrank == 0){
            printf("Time for second dgemm: %lf sec\n", t1-t0);
        }
#endif
    }

    delete[] Omega;

    return;
}


#endif
