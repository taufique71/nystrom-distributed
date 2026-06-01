#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <cmath>

// Given n, and number of splits, finds out element distribution according to Scalapack convention
void findSplits(int n, int nsplit, int* distrib){
    for (int i = 0; i < nsplit; i++){
        if(i < nsplit-1){
            // If not the last split, move ahead by ceil(n/nsplit) elements from previous split
            distrib[i] = ceil( double(n) / nsplit );
        }
        else{
            // If the last split, just move ahead by n elements from the offset
            distrib[i] = n - ceil( double(n) / nsplit ) * i;
        }
    }
}

#endif
