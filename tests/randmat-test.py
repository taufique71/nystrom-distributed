import numpy as np
import random
import time

n = 100000
niter = 1

print(">>> Generating", n ,"random numbers one by one")

t0 = time.time()

for it in range(niter):
    seeds = np.arange(n, dtype=np.int32)
    # random_numbers = np.array(list( map(lambda x: np.random.RandomState(x).rand(), seeds) ) )

    # random_numbers = np.array(list( map(lambda x: 1.0 / (x+1), seeds) ) )

    random_numbers = np.arange(n, dtype=np.float64)
    for i in range(n):
        random_numbers[i] = np.random.RandomState(seeds[i]).rand()

t1 = time.time()
print((t1-t0)/niter, "seconds")

print(">>> Generating", n ,"random numbers at a time")

t0 = time.time()

for it in range(niter):
    seed = 20250506 
    random_numbers = np.random.RandomState(seed).rand(n)

t1 = time.time()
print((t1-t0)/niter, "seconds")

