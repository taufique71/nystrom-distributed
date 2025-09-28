#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <cstring>
#include "utils.h"

#ifdef USE_CUBLAS
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

// CUDA API error checking
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
}

// curand API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)

//__global__ void fill_double_kernel(double *data, size_t N, double value) {
    //size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx < N) data[idx] = value;
//}

#endif



int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get MPI rank and size
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;
#ifdef USE_CUBLAS
	int deviceCount;
	cudaDeviceProp prop;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("myrank %d: No CUDA-capable GPU detected.\n", myrank);
    } else {
        printf("myrank %d: %d CUDA-capable GPU(s) detected.\n", myrank, deviceCount);
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    // Reduce-scatter test
    
    int m = 5000, n = 5000;
    int * distrib = new int[nprocs];
    findSplits(n, nprocs, distrib);
	//printf("distrib myrank %d: ", myrank);
	//for(int i=0; i < nprocs; i++) {
		//if(i < nprocs-1) printf("%d ", distrib[i]);
		//else printf("%d\n", distrib[i]);
	//}

    double * sendbuf_h = new double[m * n];
    std::fill( sendbuf_h, sendbuf_h + (m * n), static_cast<double>(myrank+1) );
	double * recvbuf = NULL;
	double * sendbuf = NULL;
	int* recvcnt = new int[nprocs];
	for (int i = 0; i < nprocs; i++) recvcnt[i] = distrib[i] * m;
#ifdef USE_CUBLAS
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sendbuf), sizeof(double) * (m*n) ));
    CUDA_CHECK(cudaMemcpy(sendbuf, sendbuf_h, sizeof(double) * (m*n), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&recvbuf), sizeof(double) * (m * distrib[myrank]) ));
#else
	sendbuf = new double[ m * n ];
	memcpy(sendbuf, sendbuf_h, sizeof(double) * m * n);
	recvbuf = new double[ m * distrib[myrank] ];
#endif
	
	t0 = MPI_Wtime();
	MPI_Reduce_scatter(sendbuf,
                       recvbuf,
                       recvcnt,
                       MPI_DOUBLE,
                       MPI_SUM,
                       MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	double tReduceScatter = t1-t0;
	double tReduceScatter_max;
	MPI_Allreduce(&tReduceScatter, &tReduceScatter_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if(myrank == 0) printf("Time for reduce-scatter of %d elements: %lf\n", m*n, tReduceScatter_max);

#ifdef USE_CUBLAS
    CUDA_CHECK(cudaFree(sendbuf));
    CUDA_CHECK(cudaFree(recvbuf));
#else
	delete [] sendbuf;
	delete [] recvbuf;
#endif
	delete [] recvcnt;
	delete [] distrib;
	delete [] sendbuf_h;

    /*
    // Allgather test
    int *ranks = NULL;
    int* rcounts = NULL;
    int* rdispls = NULL;
    double* sbuf = NULL;
	double* rbuf = NULL;
#ifdef USE_CUBLAS
    // Allocate buffer in GPU memory
    // Managed memory because we intend to manipulate this memory from host side
    CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void **>(&ranks), sizeof(int) * nprocs));
    CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void **>(&rcounts), sizeof(int) * nprocs));
    CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void **>(&rdispls), sizeof(int) * (nprocs+1) ));
    // Original buffers are not managed, they should only be accessed by device only
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sbuf), sizeof(double) * (myrank+1) ));
    //CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void **>(&sbuf), sizeof(double) * (myrank+1) ));
#else
    // Allocate buffer in main memory
    ranks = new int[nprocs];
    rcounts = new int[nprocs];
    rdispls = new int[nprocs+1];
    sbuf = new double[myrank+1];
#endif
    
    // Gather ranks. Not necessary, but just to test allgather
    MPI_Allgather(&myrank, 1, MPI_INT, ranks, 1, MPI_INT, MPI_COMM_WORLD);
	
	printf("ranks myrank %d: ", myrank);
	for(int i=0; i < nprocs; i++) {
		if(i < nprocs-1) printf("%d ", ranks[i]);
		else printf("%d\n", ranks[i]);
	}

    // Same instruction works independent of whether host or device memory
    // because we are using cuda managed memory
    for(int i = 0; i < nprocs; i++) rcounts[i] = ranks[i] + 1;

	printf("rcounts myrank %d: ", myrank);
	for(int i=0; i < nprocs; i++) {
		if(i < nprocs-1) printf("%d ", rcounts[i]);
		else printf("%d\n", rcounts[i]);
	}

    rdispls[0] = 0;
    std::partial_sum(rcounts, rcounts + nprocs, rdispls+1);

	printf("rdispls myrank %d: ", myrank);
	for(int i=0; i < nprocs+1; i++) {
		if(i < nprocs) printf("%d ", rdispls[i]);
		else printf("%d\n", rdispls[i]);
	}

#ifdef USE_CUBLAS
    // To mimic the GPU buffer having appropriate content, we are filling a host buffer and then copying to the device buffer
    double* sbuf_h = new double[myrank+1];
    std::fill( sbuf_h, sbuf_h + (myrank+1), static_cast<double>(myrank+1) );
    CUDA_CHECK(cudaMemcpy(sbuf, sbuf_h, sizeof(double) * (myrank+1), cudaMemcpyHostToDevice));
    delete[] sbuf_h;
#else
    std::fill( sbuf, sbuf+(myrank+1), static_cast<double>(myrank+1) );
#endif

    // Allocate receive buffer
#ifdef USE_CUBLAS
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&rbuf), sizeof(double) * (rdispls[nprocs]) ) );
#else
    rbuf = new double[rdispls[nprocs]];
#endif
    
    // Do Allgather
    MPI_Allgatherv(sbuf, myrank+1, MPI_DOUBLE, rbuf, rcounts, rdispls, MPI_DOUBLE, MPI_COMM_WORLD);

    // Copy to a device buffer to print
    double* rbuf_d = new double[rdispls[nprocs]];
#ifdef USE_CUBLAS
    CUDA_CHECK(cudaMemcpy(rbuf_d, rbuf, sizeof(double) * rdispls[nprocs], cudaMemcpyDeviceToHost));
#else
    memcpy(rbuf_d, rbuf, sizeof(double) * rdispls[nprocs] );
#endif
    for (int i = 0; i < rdispls[nprocs]; i++){
        if(i < rdispls[nprocs]-1) printf("%0.2lf ", rbuf_d[i]);
        else printf("%0.2lf\n", rbuf_d[i]);
    }
    delete[] rbuf_d;


#ifdef USE_CUBLAS
    // Allocate buffer in GPU memory
    CUDA_CHECK(cudaFree(ranks));
    CUDA_CHECK(cudaFree(rcounts));
    CUDA_CHECK(cudaFree(rdispls));
    CUDA_CHECK(cudaFree(sbuf));
    CUDA_CHECK(cudaFree(rbuf));
#else
    // Allocate buffer in main memory
    delete[] ranks;
    delete[] rcounts;
    delete[] rdispls;
    delete[] sbuf;
    delete[] rbuf;
#endif
    */

    // Finalize MPI
    MPI_Finalize();
	return 0;
}
