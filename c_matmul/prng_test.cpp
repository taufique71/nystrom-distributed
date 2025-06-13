#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include <iostream>


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
    const size_t arraySize = 10000000; // 10 million
    double* randomNumbers = new double[arraySize];
    //uint64_t* randomNumbers = new uint64_t[arraySize];

    // Initialize the PRNG with two seeds
    Xoroshiro128Plus prng(123456789, 987654321);

    double start = omp_get_wtime();

    for (size_t i = 0; i < arraySize; ++i) {
        randomNumbers[i] = prng.nextDouble();
        //randomNumbers[i] = prng.next();
        //std::cout << randomNumbers[i] << std::endl;
    }

    double end = omp_get_wtime();

    std::cout << "Time taken to fill the array: " << end-start << " seconds" << std::endl;

    delete[] randomNumbers;

    return 0;
}

