import os
import sys
import argparse
from mpi4py import MPI
import numpy as np
from communicator import ProcGrid
from matrix import ParMat
from utils import *
from randomgen import Xoroshiro128
from numpy.random import Generator

def matmul(A, B):
    assert(A.nColGlobal == B.nRowGlobal)

    npDtype = A.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)
    
    # Gather local matrices of A along the grid fibers
    # targetA = allGather(A, A.grid.fibWorld)
    targetA, agTime, agvTime, totTime = allGatherAndConcat(A.localMat, A.grid.fibWorld, concat="col")
    if (A.grid.myrank == 0):
        print("Time to gather A:", totTime, "sec")
        print("\tAllgather:", agTime, "sec")
        print("\tAllgatherv:", agvTime, "sec")

    # Gather local matrices of B along the grid columns
    # targetB = allGather(B, B.grid.colWorld)
    targetB, agTime, agvTime, totTime = allGatherAndConcat(B.localMat, B.grid.colWorld, concat="col")
    if (B.grid.myrank == 0):
        print("Time to gather B:", totTime, "sec")
        print("\tAllgather:", agTime, "sec")
        print("\tAllgatherv:", agvTime, "sec")
    
    # Create distributed C matrix with appropriate dimension that has no content
    C = ParMat(A.nRowGlobal, B.nColGlobal, A.grid, 'C') # Use the same process grid as A. Grid does not change for A, B or C. Only the face of the grid change which is specific to the matrix

    # Multiply gathered A with gathered B
    t0 = MPI.Wtime()
    C.localMat = np.matmul(targetA, targetB, order='F')
    t1 = MPI.Wtime()
    if (C.grid.myrank == 0):
        print("Time for local multiply:", t1-t0, "sec")

    # Distribute the C contribution from multiplying gathered A and B
    # Which are local matrices of C along the grid rows 
    # reduceScatter(C, C.grid.rowWorld)
    C.localMat, rsTime, totTime = splitAndReduceScatter(C.localMat, C.grid.rowWorld, split='col')
    if (C.grid.myrank == 0):
        print("Time to scatter and reduce C:", totTime, "sec")
        print("\tReduceScatter:", rsTime, "sec")
    
    # print(C.grid.myrank, C.localMat)
    
    return C

def matmul1_gen(A, B, generator = 'xoroshiro'):
    assert(A.nColGlobal == B.nRowGlobal)

    npDtype = A.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)
    
    # No need to gather local matrices of A along the grid fibers.
    # Only local copy of A would be used for 1d matmul
    t0 = MPI.Wtime()
    targetA = A.localMat
    t1 = MPI.Wtime()

    # Generate B
    t0 = MPI.Wtime()
    prng = None
    if generator == 'xoroshiro':
        prng = Generator(Xoroshiro128(123456789, plusplus=False))
    # targetB = rg.random((B.nRowGlobal, B.nColGlobal)).astype(npDtype, order='F') # Each process needs entire B
    targetB = np.zeros( (B.nRowGlobal, B.nColGlobal), dtype=npDtype, order='F')
    prng.random(targetB.shape, dtype=npDtype, out=targetB)
    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time to generate B:", t1-t0, "sec")

    # Create distributed C matrix with appropriate dimension that has no content
    C = ParMat(A.nRowGlobal, B.nColGlobal, A.grid, 'C') # Use the same process grid as A. Grid does not change for A, B or C. Only the face of the grid change which is specific to the matrix

    # Multiply local A with generated B
    t0 = MPI.Wtime()
    C.localMat = np.matmul(targetA, targetB, order='F')
    t1 = MPI.Wtime()
    if (C.grid.myrank == 0):
        print("Time for local multiply:", t1-t0, "sec")

    return C

def matmul1_comm(A, B, generator = 'xoroshiro'):
    assert(A.nColGlobal == B.nRowGlobal)

    npDtype = A.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)
    
    # No need to gather local matrices of A along the grid fibers.
    # Only local copy of A would be used for 1d matmul
    targetA = A.localMat

    # Gather local matrices of B along the grid columns
    t0 = MPI.Wtime()
    B.generate_rand(dtype=npDtype, generator="xoroshiro")
    C.localMat = np.matmul(targetA, targetB, order='F')
    t1 = MPI.Wtime()
    if (B.grid.myrank == 0):
        print("Time to generate B:", t1 - t0, "sec")

    targetB, agTime, agvTime, totTime = allGatherAndConcat(B.localMat, B.grid.colWorld, concat="col")
    if (B.grid.myrank == 0):
        print("Time to gather B:", totTime, "sec")
        # print("\tAllgather:", agTime, "sec")
        # print("\tAllgatherv:", agvTime, "sec")

    # Create distributed C matrix with appropriate dimension that has no content
    C = ParMat(A.nRowGlobal, B.nColGlobal, A.grid, 'C') # Use the same process grid as A. Grid does not change for A, B or C. Only the face of the grid change which is specific to the matrix

    # Multiply local A with generated B
    t0 = MPI.Wtime()
    C.localMat = np.matmul(targetA, targetB, order='F')
    t1 = MPI.Wtime()
    if (C.grid.myrank == 0):
        print("Time for local multiply:", t1-t0, "sec")

    return C

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numrows", type=int, help="Number of rows and columns of the matrix", default=100)
    parser.add_argument("-r", "--rank", type=int, help="Value of rank for low rank approximation", default=20)
    parser.add_argument("-p1", "--p1", type=int, help="Number of process grid rows for first matmul")
    parser.add_argument("-p2", "--p2", type=int, help="Number of process grid cols for first matmul")
    parser.add_argument("-p3", "--p3", type=int, help="Number of process grid cols for first matmul")
    parser.add_argument("-n1", "--n1", type=int, help="Number of rows in A")
    parser.add_argument("-n2", "--n2", type=int, help="Number of columns in A / Number of rows in B")
    parser.add_argument("-n3", "--n3", type=int, help="Number of columns in B")
    args = parser.parse_args()
    # print(args)

    n = args.numrows
    r = args.rank
    p1 = args.p1
    p2 = args.p2
    p3 = args.p3
    n1 = args.n1
    n2 = args.n2
    n3 = args.n3

    myrank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    
    p = nprocs
    assert(p == args.p1 * args.p2 * args.p3)
    grid = ProcGrid(args.p1, args.p2, args.p3)

    A = ParMat(n1, n2, grid, 'A')
    MPI.COMM_WORLD.Barrier()
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.int32)
    B.generate(dtype=np.int32)
    C = matmul(A,B)
    
    ## Check correctness
    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()
    if A.grid.myrank == 0:
        print(Ag)
        print("x")
    if B.grid.myrank == 0:
        print(Bg)
        print("=")
    if C.grid.myrank == 0:
        print(Cg)
        print("---")

    if np.array_equal( np.matmul(Ag, Bg), Cg):
        if A.grid.myrank == 0:
            print("Correct")
    else:
        if A.grid.myrank == 0:
            print("Incorrect")
