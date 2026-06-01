#ifndef PRNG_H
#define PRNG_H

#include <cassert>
#include <numeric>

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



#endif
