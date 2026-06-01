#ifndef UTILS_H
#define UTILS_H

#include <algorithm>

// Given n elements and nsplit splits, fills distrib[] with the ScaLAPACK-style
// block distribution: first floor(n / block) blocks of size ceil(n/nsplit), then
// the remainder, then zeros if nsplit > n. Sum of distrib[] equals n.
inline void findSplits(int n, int nsplit, int* distrib){
    int block = (n + nsplit - 1) / nsplit;
    for (int i = 0; i < nsplit; i++){
        int start = std::min(i * block, n);
        int end = std::min((i + 1) * block, n);
        distrib[i] = end - start;
    }
}

#endif
