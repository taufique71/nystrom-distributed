#ifndef MATRIX_H
#define MATRIX_H

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <omp.h>

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

class ParMat {
public:
    ParMat(int m, int n, ProcGrid& grid, char frontFace)
        : nRowGlobal(m), nColGlobal(n), grid(grid), frontFace(frontFace) {
        //initialize();
        if (frontFace == 'A') {
            // Distribute matrix rows
            nRowLocal = nRowGlobal / grid.nProcRow;
            localRowStart = nRowLocal * grid.rankInColWorld;
            if (grid.rankInColWorld == grid.nProcRow - 1) {
                nRowLocal = nRowGlobal - nRowLocal * grid.rankInColWorld;
            }

            // Distribute matrix columns
            localColStart = nColGlobal / grid.nProcCol * grid.rankInRowWorld;
            nColLocal = nColGlobal / grid.nProcCol;
            if (grid.rankInRowWorld == grid.nProcCol - 1) {
                nColLocal = nColGlobal - nColGlobal / grid.nProcCol * grid.rankInRowWorld;
            }

            // Further distribute matrix columns to fibers
            localColStart += nColLocal / grid.nProcFib * grid.rankInFibWorld;
            if (grid.rankInFibWorld < grid.nProcFib - 1) {
                nColLocal /= grid.nProcFib;
            } else {
                nColLocal -= nColLocal / grid.nProcFib * grid.rankInFibWorld;
            }
        }
        else if (frontFace == 'B') {
            // Similar logic for frontFace 'B'
            
            // Distribute matrix rows
            nRowLocal = nRowGlobal / grid.nProcCol;
            localRowStart = nRowLocal * grid.rankInRowWorld;
            if (grid.rankInRowWorld == grid.nProcCol - 1) {
                nRowLocal = nRowGlobal - nRowLocal * grid.rankInRowWorld;
            }

            // Distribute matrix columns
            localColStart = nColGlobal / grid.nProcFib * grid.rankInFibWorld;
            nColLocal = nColGlobal / grid.nProcFib;
            if (grid.rankInFibWorld == grid.nProcFib - 1) {
                nColLocal = nColGlobal - nColGlobal / grid.nProcFib * grid.rankInFibWorld;
            }

            localColStart += nColLocal / grid.nProcRow * grid.rankInColWorld;
            if (grid.rankInColWorld < grid.nProcRow - 1) {
                nColLocal /= grid.nProcRow;
            } else {
                nColLocal -= nColLocal / grid.nProcRow * grid.rankInColWorld;
            }
        }
        else if (frontFace == 'C') {
            // Distribute matrix rows
            nRowLocal = nRowGlobal / grid.nProcRow;
            localRowStart = nRowLocal * grid.rankInColWorld;
            if (grid.rankInColWorld == grid.nProcRow - 1) {
                nRowLocal = nRowGlobal - nRowLocal * grid.rankInColWorld;
            }

    		// Distribute matrix columns
    		localColStart = nColGlobal / grid.nProcFib * grid.rankInFibWorld;
    		nColLocal = nColGlobal / grid.nProcFib;
    		if (grid.rankInFibWorld == grid.nProcFib - 1) {
    			nColLocal = nColGlobal - nColGlobal / grid.nProcFib * grid.rankInFibWorld;
    		}

    		// Further distribute matrix columns to grid fibers
    		localColStart += nColLocal / grid.nProcCol * grid.rankInRowWorld;
    		if (grid.rankInRowWorld < grid.nProcCol - 1) {
    			nColLocal /= grid.nProcCol;
    		} else {
    			nColLocal -= nColLocal / grid.nProcCol * grid.rankInRowWorld;
    		}
    	}
        //std::cout << nRowLocal << "x" << nColLocal << ", " << sizeof (double) << std::endl;
        //localMat.resize(nRowLocal*nColLocal);
        //localMat = (double *) malloc(nRowLocal * nColLocal * sizeof(double));
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&localMat), sizeof(double) * (nRowLocal * nColLocal)) );
#else
        localMat = new double[nRowLocal * nColLocal];
#endif
    }

    ~ ParMat(){
#ifdef USE_CUBLAS
        CUDA_CHECK(cudaFree(localMat));
#else
        delete[] localMat;
#endif
    }

    void generate() {
        //localMat.resize(nRowLocal, std::vector<double>(nColLocal, 0.0));
#ifdef USE_CUBLAS
        // Generate in host memory and then copy to device memory
        // Alternative is to write a CUDA kernel to generate in GPU, which will be adopted if needed
        double *localMat_h = new double[nRowLocal * nColLocal];
    	for (int idxColLocal = 0; idxColLocal < nColLocal; ++idxColLocal) {
    		for (int idxRowLocal = 0; idxRowLocal < nRowLocal; ++idxRowLocal) {
                int idxRowGlobal = localRowStart + idxRowLocal;
                int idxColGlobal = localColStart + idxColLocal;
    			int idx = idxColLocal * nRowLocal + idxRowLocal;
                localMat_h[idx] = (double)(idxRowGlobal * nColGlobal + idxColGlobal);
            }
        }
        CUDA_CHECK(cudaMemcpy(localMat, localMat_h, sizeof(double) * (nRowLocal * nColLocal), cudaMemcpyHostToDevice));
        delete[] localMat_h;
#else
    	for (int idxColLocal = 0; idxColLocal < nColLocal; ++idxColLocal) {
    		for (int idxRowLocal = 0; idxRowLocal < nRowLocal; ++idxRowLocal) {
                int idxRowGlobal = localRowStart + idxRowLocal;
                int idxColGlobal = localColStart + idxColLocal;
    			int idx = idxColLocal * nRowLocal + idxRowLocal;
                localMat[idx] = (double)(idxRowGlobal * nColGlobal + idxColGlobal);
            }
        }
#endif
    }

    void printLocalMatrix() const {
    	std::cout << "Local Matrix (Process " << grid.myrank << " [" << nRowLocal << "x" << nColLocal << "]" <<  "):\n";
    	//for (const auto& row : localMat) {
    		//for (const auto& val : row) {
    			//std::cout << val << " ";
    		//}
    		//std::cout << "\n";
    	//}
    }

    int nRowGlobal;
    int nColGlobal;
    int nRowLocal;
    int nColLocal;
    ProcGrid& grid;
    char frontFace;
    int localRowStart;
    int localColStart;
    double *localMat;
};

/*
 * Written by: Taufique Hussain (hussaint@wfu.edu)
 * General matrix matrix multiplication
 * */
ParMat matmul(ParMat& A, ParMat& B){
    double t0, t1, t2, t3;
    int nproc, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    assert(A.nColGlobal == B.nRowGlobal);
    ProcGrid grid = A.grid; // Same grid
    ParMat C(A.nRowGlobal, B.nColGlobal, grid, 'C');
    
    double* recvA = NULL;
    double* recvB = NULL;
    double* multC = NULL;
    int nColRecvA, nColRecvB;

#ifdef USE_CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
#endif

    {
        int commSize = 0;
        MPI_Comm_size(grid.fibWorld, &commSize);

        int nColToSend = A.nColLocal;
        int* nColToRecv = new int[commSize];
        
        t0 = MPI_Wtime();

        MPI_Allgather(&nColToSend, 1, MPI_INT, nColToRecv, 1, MPI_INT, grid.fibWorld);

        //std::cout << myrank << ": ";
        //for(int i = 0; i < commSize; i++){
            //std::cout << nColToRecv[i] << " " ;
        //}
        //std::cout << std::endl;

        int* nValToRecv = new int[commSize];
        nColRecvA = 0;
        for(int i = 0; i < commSize; i++) {
            nValToRecv[i] = nColToRecv[i] * A.nRowLocal;
            nColRecvA = nColRecvA + nColToRecv[i];
        }

        int* recvDispls = new int[commSize + 1];
        recvDispls[0] = 0;
        std::partial_sum(nValToRecv, nValToRecv + commSize, recvDispls+1);
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&recvA), sizeof(double) * (recvDispls[commSize]) ) );
#else
        recvA = new double[recvDispls[commSize]];
#endif

        MPI_Allgatherv(A.localMat, A.nRowLocal*A.nColLocal, MPI_DOUBLE, recvA, nValToRecv, recvDispls, MPI_DOUBLE, grid.fibWorld);

        t1 = MPI_Wtime();


        delete[] nColToRecv;
        delete[] nValToRecv;
        delete[] recvDispls;

        if(myrank == 0){
            printf("Time to gather A: %lf sec\n", t1-t0);
        }
    }


    {
        int commSize = 0;
        MPI_Comm_size(grid.colWorld, &commSize);

        // Allgather the columns of B accross process grid columns 
        int nColToSend = B.nColLocal;
        int* nColToRecv = new int[commSize]; 

        t0 = MPI_Wtime();

        // Gather the number of columns from all processes in the grid columns
        MPI_Allgather(&nColToSend, 1, MPI_INT, nColToRecv, 1, MPI_INT, grid.colWorld);

        // Calculate the number of values to receive based on the number of rows in B
        int* nValToRecv = new int[commSize];
        nColRecvB = 0;
        for (int i = 0; i < commSize; i++) {
            nValToRecv[i] = nColToRecv[i] * B.nRowLocal; 
            nColRecvB = nColRecvB + nColToRecv[i];
        }

        // Prepare displacements for the received data
        int* recvDispls = new int[commSize + 1];
        recvDispls[0] = 0;
        std::partial_sum(nValToRecv, nValToRecv + commSize, recvDispls + 1);

#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&recvB), sizeof(double) * (recvDispls[commSize]) ) );
#else
        recvB = new double[recvDispls[commSize]];
#endif
       
        // Perform the Allgatherv operation to collect B matrix columns
        MPI_Allgatherv(B.localMat, B.nRowLocal * B.nColLocal, MPI_DOUBLE, recvB, nValToRecv, recvDispls, MPI_DOUBLE, grid.colWorld);

        t1 = MPI_Wtime();

        // Clean up allocated memory
        delete[] nColToRecv;
        delete[] nValToRecv;
        delete[] recvDispls;

        if(myrank == 0){
            printf("Time to gather B: %lf sec\n", t1-t0);
        }
    }

    //std::cout << myrank << ": " << nColRecvA << " - " << B.nRowLocal << std::endl;

#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&multC), sizeof(double) * (A.nRowLocal * nColRecvB) ) );
#else
        multC = new double[A.nRowLocal * nColRecvB];
#endif

    int cblas_m = A.nRowLocal;
    int cblas_k = nColRecvA;
    int cblas_n = nColRecvB;
    double cblas_alpha = 1.0;
    double cblas_beta = 0.0;
    double* cblas_a = recvA;
    int cblas_lda = A.nRowLocal;
    double* cblas_b = recvB;
    int cblas_ldb = B.nRowLocal;
    double* cblas_c = multC; 
    int cblas_ldc = cblas_lda; // Number of rows of the matrix

#ifdef USE_CUBLAS
	double tMemMove = 0;
	double tDgemm = 0;
	t0 = MPI_Wtime();
	cublasOperation_t transA = CUBLAS_OP_N;
	cublasOperation_t transB = CUBLAS_OP_N;
	t1 = MPI_Wtime();
	tMemMove += (t1-t0);
	

	t0 = MPI_Wtime();
    cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
                &cblas_alpha, cblas_a, cblas_lda, cblas_b, cblas_ldb,
                &cblas_beta, cblas_c, cblas_ldc);
	t1 = MPI_Wtime();
	tDgemm += (t1-t0);

	t0 = MPI_Wtime();
	t1 = MPI_Wtime();
	tMemMove += (t1-t0);

	if(myrank == 0){
		printf("Time for host-device mem movement: %lf sec\n", tMemMove);
		printf("Time for local multiply: %lf sec\n", tDgemm);
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
        printf("Time for local multiply: %lf sec\n", t1-t0);
    }
#endif

    {
        int commSize = 0;
        MPI_Comm_size(grid.rowWorld, &commSize);

        // Reduce scatter of C along process grid column
        int nColToRecv = C.nColLocal; // Data structure is already prepared, just collect necessary information
        int* nColToSend = new int[commSize]; // Change to commSize for column-wise collection
        
        t0 = MPI_Wtime();

        // Gather the number of columns from all processes in the column grid
        MPI_Allgather(&nColToRecv, 1, MPI_INT, nColToSend, 1, MPI_INT, grid.rowWorld); 

        // Calculate the number of values to scatter based on the number of rows in C
        int* nValToSend = new int[commSize];
        for (int i = 0; i < commSize; i++) {
            nValToSend[i] = nColToSend[i] * C.nRowLocal; // Update to use C matrix
        }

        // Prepare displacements for the data to be scattered
        int* sendDispls = new int[commSize + 1];
        sendDispls[0] = 0;
        std::partial_sum(nValToSend, nValToSend + commSize, sendDispls + 1);

        // Scatter and reduce the relevant pieces of C
        MPI_Reduce_scatter(multC, C.localMat, nValToSend, MPI_DOUBLE, MPI_SUM, grid.rowWorld);

        t1 = MPI_Wtime();
        if(myrank == 0){
            printf("Time to scatter and reduce C: %lf sec\n", t1-t0);
        }

        // Clean up allocated memory
        delete[] nColToSend;
        delete[] nValToSend;
        delete[] sendDispls;
    }
    
#ifdef USE_CUBLAS
    CUDA_CHECK(cudaFree(recvA));
    CUDA_CHECK(cudaFree(recvB));
    CUDA_CHECK(cudaFree(multC));
	cublasDestroy(handle);
#else
    delete[] recvA;
    delete[] recvB;
    delete[] multC;
#endif

    return C;
}

/*
 * Written by: Taufique Hussain (hussaint@wfu.edu)
 * Nystrom style first multiplication
 * Y = A x Omega where Omega is a tall and skinny random matrix.
 * All matrices distributed in 1D process grid
 * Omega is generated redundantly in all process
 * */
ParMat matmul1_gen(ParMat& A, ParMat& B, std::string generator){
    double t0, t1, t2, t3;
    int nproc, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    assert(A.nColGlobal == B.nRowGlobal);
    ProcGrid grid = A.grid; // Same grid
    ParMat C(A.nRowGlobal, B.nColGlobal, grid, 'C');
    
    double* recvA = A.localMat;
    double* recvB = NULL;
    double* multC = C.localMat;

#ifdef USE_CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
#endif

    {
        t0 = MPI_Wtime();
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&recvB), sizeof(double) * (B.nRowLocal * B.nColGlobal) ) );
        curandGenerator_t gen = NULL;
        curandRngType_t rng = CURAND_RNG_PSEUDO_XORWOW; 
        curandOrdering_t order = CURAND_ORDERING_PSEUDO_SEEDED;
        const unsigned long long offset = 0ULL;
        const unsigned long long seed = 1234ULL;

        CURAND_CHECK(curandCreateGenerator(&gen, rng));
        CURAND_CHECK(curandSetGeneratorOffset(gen, offset));
        CURAND_CHECK(curandSetGeneratorOrdering(gen, order));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
        CURAND_CHECK(curandGenerateUniformDouble(gen, recvB, B.nRowGlobal * B.nColGlobal ));
#else
        recvB = new double[B.nRowGlobal * B.nColGlobal]; // Allocate for received B matrix
        Xoroshiro128Plus prng(123456789, 987654321); // Defined in prng.cpp

        for (size_t i = 0; i < B.nRowGlobal * B.nColGlobal; ++i) {
            recvB[i] = prng.nextDouble();
        }
#endif
        t1 = MPI_Wtime();

        if(myrank == 0){
            printf("Time to generate B: %lf sec\n", t1-t0);
        }
    }

    auto cblas_m = A.nRowLocal;
    auto cblas_k = B.nRowGlobal;
    auto cblas_n = B.nColGlobal;
    auto cblas_alpha = 1.0;
    auto cblas_beta = 0.0;
    auto cblas_a = recvA;
    auto cblas_lda = A.nRowLocal;
    auto cblas_b = recvB;
    auto cblas_ldb = B.nRowGlobal;
    auto cblas_c = multC; 
    auto cblas_ldc = cblas_lda; // Number of rows of the matrix

#ifdef USE_CUBLAS
	double tMemMove = 0;
	double tDgemm = 0;
	t0 = MPI_Wtime();
	cublasOperation_t transA = CUBLAS_OP_N;
	cublasOperation_t transB = CUBLAS_OP_N;
	t1 = MPI_Wtime();
	tMemMove += (t1-t0);
	

	t0 = MPI_Wtime();
    cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
                &cblas_alpha, cblas_a, cblas_lda, cblas_b, cblas_ldb,
                &cblas_beta, cblas_c, cblas_ldc);
	t1 = MPI_Wtime();
	tDgemm += (t1-t0);

	t0 = MPI_Wtime();
	t1 = MPI_Wtime();
	tMemMove += (t1-t0);

	if(myrank == 0){
		printf("Time for host-device mem movement: %lf sec\n", tMemMove);
		printf("Time for local multiply: %lf sec\n", tDgemm);
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
        printf("Time for local multiply: %lf sec\n", t1-t0);
    }
#endif

#ifdef USE_CUBLAS
    CUDA_CHECK(cudaFree(recvB));
	cublasDestroy(handle);
#else
    delete[] recvB;
#endif

    return C;
}

/*
 * Written by: Taufique Hussain (hussaint@wfu.edu)
 * Nystrom style first multiplication
 * Y = A x Omega where Omega is a tall and skinny random matrix.
 * All matrices distributed in 1D process grid
 * A piece of Omega is randomly generated in all process and communicated
 * */
ParMat matmul1_comm(ParMat& A, ParMat& B, std::string generator){
    double t0, t1, t2, t3;
    int nproc, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    assert(A.nColGlobal == B.nRowGlobal);
    ProcGrid grid = A.grid; // Same grid
    ParMat C(A.nRowGlobal, B.nColGlobal, grid, 'C');
    
    double* recvA = A.localMat;
    double* recvB = NULL;
    double* multC = C.localMat;
    int nColRecvB;

#ifdef USE_CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
#endif

    {
        t0 = MPI_Wtime();
#ifdef USE_CUBLAS
        curandGenerator_t gen = NULL;
        curandRngType_t rng = CURAND_RNG_PSEUDO_XORWOW; 
        curandOrdering_t order = CURAND_ORDERING_PSEUDO_SEEDED;
        const unsigned long long offset = 0ULL;
        //const unsigned long long seed = 1234ULL;
        const unsigned long long seed = static_cast<unsigned long long>(myrank);

        CURAND_CHECK(curandCreateGenerator(&gen, rng));
        CURAND_CHECK(curandSetGeneratorOffset(gen, offset));
        CURAND_CHECK(curandSetGeneratorOrdering(gen, order));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
        CURAND_CHECK(curandGenerateUniformDouble(gen, B.localMat, (B.nRowLocal * B.nColLocal)));
#else
        //Xoroshiro128Plus prng(123456789, 987654321); // Defined in prng.cpp
        Xoroshiro128Plus prng(myrank, myrank); // Defined in prng.cpp

        for (size_t i = 0; i < B.nRowLocal * B.nColLocal; ++i) {
            B.localMat[i] = prng.nextDouble();
        }
#endif
        t1 = MPI_Wtime();

        if(myrank == 0){
            printf("Time to generate B: %lf sec\n", t1-t0);
        }
    }

    {
        int commSize = 0;
        MPI_Comm_size(grid.colWorld, &commSize);

        // Allgather the columns of B accross process grid columns 
        int nColToSend = B.nColLocal;
        int* nColToRecv = new int[commSize]; 

        t0 = MPI_Wtime();

        // Gather the number of columns from all processes in the grid columns
        MPI_Allgather(&nColToSend, 1, MPI_INT, nColToRecv, 1, MPI_INT, grid.colWorld);

        // Calculate the number of values to receive based on the number of rows in B
        int* nValToRecv = new int[commSize];
        nColRecvB = 0;
        for (int i = 0; i < commSize; i++) {
            nValToRecv[i] = nColToRecv[i] * B.nRowLocal; 
            nColRecvB = nColRecvB + nColToRecv[i];
        }

        // Prepare displacements for the received data
        int* recvDispls = new int[commSize + 1];
        recvDispls[0] = 0;
        std::partial_sum(nValToRecv, nValToRecv + commSize, recvDispls + 1);
#ifdef USE_CUBLAS
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&recvB), sizeof(double) * (recvDispls[commSize]) ) );
#else
        recvB = new double[recvDispls[commSize]]; // Allocate for received B matrix
#endif

        // Perform the Allgatherv operation to collect B matrix columns
        MPI_Allgatherv(B.localMat, B.nRowLocal * B.nColLocal, MPI_DOUBLE, recvB, nValToRecv, recvDispls, MPI_DOUBLE, grid.colWorld);

        t1 = MPI_Wtime();

        // Clean up allocated memory
        delete[] nColToRecv;
        delete[] nValToRecv;
        delete[] recvDispls;

        if(myrank == 0){
            printf("Time to gather B: %lf sec\n", t1-t0);
        }
    }

    auto cblas_m = A.nRowLocal;
    auto cblas_k = A.nColLocal;
    auto cblas_n = nColRecvB;
    auto cblas_alpha = 1.0;
    auto cblas_beta = 0.0;
    auto cblas_a = recvA;
    auto cblas_lda = A.nRowLocal;
    auto cblas_b = recvB;
    auto cblas_ldb = B.nRowGlobal;
    auto cblas_c = multC; 
    auto cblas_ldc = cblas_lda; // Number of rows of the matrix

#ifdef USE_CUBLAS
	double tMemMove = 0;
	double tDgemm = 0;
	t0 = MPI_Wtime();
	cublasOperation_t transA = CUBLAS_OP_N;
	cublasOperation_t transB = CUBLAS_OP_N;
	t1 = MPI_Wtime();
	tMemMove += (t1-t0);
	
	t0 = MPI_Wtime();
    cublasDgemm(handle, transA, transB, cblas_m, cblas_n, cblas_k,
                &cblas_alpha, cblas_a, cblas_lda, cblas_b, cblas_ldb,
                &cblas_beta, cblas_c, cblas_ldc);
	t1 = MPI_Wtime();
	tDgemm += (t1-t0);

	t0 = MPI_Wtime();
	t1 = MPI_Wtime();
	tMemMove += (t1-t0);

	if(myrank == 0){
		printf("Time for host-device mem movement: %lf sec\n", tMemMove);
		printf("Time for local multiply: %lf sec\n", tDgemm);
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
        printf("Time for local multiply: %lf sec\n", t1-t0);
    }
#endif

#ifdef USE_CUBLAS
    CUDA_CHECK(cudaFree(recvB));
	cublasDestroy(handle);
#else
    delete[] recvB;
#endif

    return C;
}


#endif
