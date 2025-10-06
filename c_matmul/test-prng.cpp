#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include <iostream>

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

// CUDA API error checking
//#define CUDA_CHECK(err)                                                        \
  //do {                                                                         \
    //cudaError_t err_ = (err);                                                  \
    //if (err_ != cudaSuccess) {                                                 \
      //std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
      //throw std::runtime_error("CUDA error");                                  \
    //}                                                                          \
  //} while (0)
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

#else
#include <mkl.h>
#endif


static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

class Xoroshiro128Plus {
public:
    Xoroshiro128Plus(uint64_t seed1, uint64_t seed2) : state1(seed1), state2(seed2) {}

    uint64_t next() {
        uint64_t s0 = state1;
        uint64_t s1 = state2;
        uint64_t result = s0 + s1;

        //s1 ^= s0;
        //state1 = (s0 << 55) | (s0 >> (64 - 55)); // Rotate left
        //state2 = s1 ^ (s1 << 14) ^ (s1 << 36); // Mix
                                               
        s1 ^= s0;
        state1 = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
        state2 = rotl(s1, 37); // c

        return result;
    }

    double nextDouble() {
        //return static_cast<double>(next()) / static_cast<double>(UINT64_MAX);
        return (next() >> 11) * 0x1.0p-53;
    }

private:
    uint64_t state1;
    uint64_t state2;
};

int main() {
    double start, end;

	//size_t arraySize = 10000000; // 10 million
	size_t arraySize = 50000*5000; 
	double* randomNumbers = new double[arraySize];


    //uint64_t* randomNumbers = new uint64_t[arraySize];


    start = omp_get_wtime();
#pragma omp parallel
	{
		int tid = omp_get_thread_num();

        // Initialize the PRNG with two seeds
        Xoroshiro128Plus prng(123456789+tid, 987654321+tid);

		/* decide how many numbers this thread will produce */
		size_t per_thr = arraySize / omp_get_num_threads();
        size_t local_arraySize;

        if(tid == omp_get_num_threads()-1 ){
            local_arraySize = arraySize - per_thr * tid;
        }
        else{
            local_arraySize = per_thr; 
        }

        for (size_t i = 0; i < local_arraySize; ++i) {
            randomNumbers[per_thr * tid + i] = prng.nextDouble();
            //randomNumbers[i] = prng.next();
            //std::cout << randomNumbers[i] << std::endl;
        }
	}

    end = omp_get_wtime();

    std::cout << "Time taken to fill the array of " << arraySize << " with CPU: " << end-start << " seconds" << std::endl;

	
    using data_type = double;
    data_type* d_data = NULL;
    data_type* d_data_2 = NULL;
    data_type* h_data = new data_type[arraySize];
    data_type* h_data_2 = new data_type[arraySize];
#ifdef USE_CUBLAS
	int deviceCount;
	double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;
	cudaDeviceProp prop;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable GPU detected." << std::endl;
    } else {
        std::cout << deviceCount << " CUDA-capable GPU(s) detected." << std::endl;
    }


	// Following example from NVIDIA: 
	// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuRAND/Host/mrg32k3a/curand_mrg32k3a_uniform_example.cpp
	cudaStream_t stream = NULL;
	curandGenerator_t gen = NULL;
	//curandRngType_t rng = CURAND_RNG_PSEUDO_MRG32K3A; // Fastest according to: https://developer.nvidia.com/curand
    //curandRngType_t rng = CURAND_RNG_PSEUDO_XORWOW; 
    curandRngType_t rng = CURAND_RNG_PSEUDO_PHILOX4_32_10; 
	curandOrdering_t order = CURAND_ORDERING_PSEUDO_DEFAULT;

	const unsigned long long offset = 0ULL;
	const unsigned long long seed = 1234ULL;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data),
                                sizeof(data_type) * arraySize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data_2),
                                sizeof(data_type) * arraySize));

    start = omp_get_wtime();
	t0 = omp_get_wtime();
	//CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	t1 = omp_get_wtime();
    //CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data),
                                //sizeof(data_type) * arraySize));
    //CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data_2),
                                //sizeof(data_type) * arraySize));
	t2 = omp_get_wtime();
    CURAND_CHECK(curandCreateGenerator(&gen, rng));
	t3 = omp_get_wtime();
    //CURAND_CHECK(curandSetStream(gen, stream));
	t4 = omp_get_wtime();
    CURAND_CHECK(curandSetGeneratorOffset(gen, offset));
	t5 = omp_get_wtime();
    CURAND_CHECK(curandSetGeneratorOrdering(gen, order));
	t6 = omp_get_wtime();
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
	t7 = omp_get_wtime();
    CURAND_CHECK(curandGenerateUniformDouble(gen, d_data, arraySize));
	t8 = omp_get_wtime();
    CURAND_CHECK(curandGenerateUniformDouble(gen, d_data, arraySize));
	t9 = omp_get_wtime();
    CUDA_CHECK(cudaDeviceSynchronize());
	//CUDA_CHECK(cudaMemcpyAsync(h_data, d_data,
							 //sizeof(data_type) * arraySize,
							 //cudaMemcpyDeviceToHost, stream));
	//CUDA_CHECK(cudaMemcpyAsync(h_data_2, d_data_2,
							 //sizeof(data_type) * arraySize,
							 //cudaMemcpyDeviceToHost, stream));
	//CUDA_CHECK(cudaMemcpy(d_data, h_data, sizeof(data_type) * arraySize, cudaMemcpyHostToDevice));
	//CUDA_CHECK(cudaMemcpy(d_data_2, h_data_2, sizeof(data_type) * arraySize, cudaMemcpyHostToDevice));
	t10 = omp_get_wtime();
    //CUDA_CHECK(cudaStreamSynchronize(stream));
	t11 = omp_get_wtime();
    //CUDA_CHECK(cudaFree(d_data));
    //CUDA_CHECK(cudaFree(d_data_2));
	t12 = omp_get_wtime();
    end = omp_get_wtime();

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_data_2));

    std::cout << "Time taken to fill the array of " << arraySize << " with GPU: " << end-start << " seconds" << std::endl;
    std::cout << "\tTime to create CUDA stream: " << t1-t0 << std::endl;
    std::cout << "\tTime to allocate memory: " << t2-t1 << std::endl;
    std::cout << "\tTime to create generator: " << t3-t2 << std::endl;
    std::cout << "\tTime to set generator to the stream: " << t4-t3 << std::endl;
    std::cout << "\tTime to set offset to the generator: " << t5-t4 << std::endl;
    std::cout << "\tTime to set ordering to the generator: " << t6-t5 << std::endl;
    std::cout << "\tTime to set seed to the generator: " << t7-t6 << std::endl;
    std::cout << "\tTime to generate 1st batch: " << t8-t7 << std::endl;
    std::cout << "\tTime to generate 2nd batch: " << t9-t8 << std::endl;
    std::cout << "\tTime to copy data from gpu to cpu: " << t10-t9 << std::endl;
    std::cout << "\tTime to sync stream: " << t11-t10 << std::endl;
    std::cout << "\tTime to free memory: " << t12-t11 << std::endl;
#else
    start = omp_get_wtime();
#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		/* each thread creates its own stream */
		VSLStreamStatePtr thr_stream;
		unsigned int thr_seed = 1234 + tid;   // different seed per thread
		vslNewStream(&thr_stream, VSL_BRNG_PHILOX4X32X10, thr_seed);

		/* decide how many numbers this thread will produce */
		size_t per_thr = arraySize / omp_get_num_threads();
        size_t local_arraySize;

        if(tid == omp_get_num_threads()-1 ){
            local_arraySize = arraySize - per_thr * tid;
        }
        else{
            local_arraySize = per_thr; 
        }

		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
					 thr_stream,
					 local_arraySize,
					 randomNumbers + tid * per_thr,
					 0.0, 1.0);

		vslDeleteStream(&thr_stream);
	}
    end = omp_get_wtime();

    std::cout << "Time taken to fill the array of " << arraySize << " with CPU: " << end-start << " seconds" << std::endl;
#endif

	size_t idx;
	printf("Pick an index in the range [0,10000000):");
	scanf("%ulld", &idx);
    //idx = 101;
	printf("h_data[%d]: %lf\n", idx, h_data[idx]);
	printf("h_data_2[%d]: %lf\n", idx, h_data_2[idx]);
	printf("randomNumbers[%d]: %lf\n", idx, randomNumbers[idx]);

    delete[] randomNumbers;
    delete[] h_data;

    return 0;
}

