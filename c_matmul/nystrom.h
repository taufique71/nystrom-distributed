#ifndef NYSTROM_H
#define NYSTROM_H

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <mkl.h>
#include "procgrid.h"
#include "matrix.h"
#include "prng.h"

void nystrom_1d_noredist_1d(ParMat &A, int r){
    double t0, t1, t2, t3;
    int nproc, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    ProcGrid grid = A.grid; // Same grid
    ParMat C(A.nRowGlobal, r, grid, 'C');
    
    double* recvA = A.localMat;
    double* recvB = NULL;
    double* multC = new double[ A.nRowGlobal * r];
    
    //printf("Here\n");
    {
        t0 = MPI_Wtime();

        recvB = new double[A.nColGlobal * r]; // Allocate for received B matrix
        Xoroshiro128Plus prng(123456789, 987654321); // Defined in prng.cpp

        for (size_t i = 0; i < A.nColGlobal * r; ++i) {
            recvB[i] = prng.nextDouble();
        }

        t1 = MPI_Wtime();

        if(myrank == 0){
            printf("Time to generate B: %lf sec\n", t1-t0);
        }
    }

    {
        auto cblas_m = A.nRowLocal;
        auto cblas_k = A.nColGlobal;
        auto cblas_n = r;
        auto cblas_alpha = 1.0;
        auto cblas_beta = 0.0;
        auto cblas_a = recvA;
        auto cblas_lda = cblas_m;
        auto cblas_b = recvB;
        auto cblas_ldb = cblas_k;
        auto cblas_c = multC; 
        auto cblas_ldc = cblas_m; // Number of rows of the matrix
                                         
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
            printf("Time for first local multiply: %lf sec\n", t1-t0);
        }
    }

    double* contribZ = NULL;
    int BtColOffset = grid.rowRank * (A.nColGlobal / grid.nProcRow); // How many columns of BT needs to be moved forward
    int BtColCount = (grid.rankInColWorld < grid.nProcRow-1) ? (A.nColGlobal / grid.nProcRow) : (A.nColGlobal - BtColOffset) ;
    contribZ = new double[r * r];

    {
        auto cblas_m = r;
        auto cblas_k = C.nRowLocal;
        auto cblas_n = C.nColLocal; // r
        auto cblas_alpha = 1.0;
        auto cblas_beta = 0.0;
        auto cblas_a = recvB + BtColOffset * r; // Move forward these many entries of B
        auto cblas_lda = cblas_k;
        auto cblas_b = C.localMat;
        auto cblas_ldb = cblas_k;
        auto cblas_c = contribZ; 
        auto cblas_ldc = cblas_m; // Number of rows of the matrix
                                         
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
            printf("Time for second local multiply: %lf sec\n", t1-t0);
        }
    }

    ParMat Z(r, r, grid, 'B'); // B face for column split distrib of Z
    {
        int commSize = 0;
        MPI_Comm_size(grid.colWorld, &commSize); // colWorld - because we chose B face for distribution of Z

        // Reduce scatter of Z along process grid column
        int nColToRecv = Z.nColLocal; // Data structure is already prepared, just collect necessary information
        int* nColToSend = new int[commSize]; // Change to commSize for column-wise collection
        
        t0 = MPI_Wtime();

        // Gather the number of columns from all processes in the column grid
        MPI_Allgather(&nColToRecv, 1, MPI_INT, nColToSend, 1, MPI_INT, grid.colWorld); // Update communicator

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
        MPI_Reduce_scatter(contribZ, Z.localMat, nValToSend, MPI_DOUBLE, MPI_SUM, grid.colWorld);

        t1 = MPI_Wtime();
        if(myrank == 0){
            printf("Time to scatter and reduce Z: %lf sec\n", t1-t0);
        }

        // Clean up allocated memory
        delete[] nColToSend;
        delete[] nValToSend;
        delete[] sendDispls;
    }

    delete[] recvB;
    delete[] multC;
    delete[] contribZ;

    return;

}

void nystrom_1d_redist_1d(ParMat &A, int r){
    return;
}


#endif
