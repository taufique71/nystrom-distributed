import os
import sys
import argparse
from mpi4py import MPI
import numpy as np
from communicator import ProcGrid
from matrix import *
from utils import *
from randomgen import Xoroshiro128
from numpy.random import Generator

def nystrom_1d_noredist_1d(A, r):
    npDtype = A.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)
    
    # Only local copy of A would be used for 1d matmul
    targetA = A.localMat

    # Create distributed B matrix with appropriate dimension 

    t0 = MPI.Wtime()

    targetB = np.zeros( (A.nColGlobal, r), dtype=np.float64, order='F')
    prng = Generator(Xoroshiro128(1234, plusplus=False))
    prng.random(targetB.shape, dtype=np.float64, out=targetB)

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time to generate B:", t1 - t0, "sec")

    Y = ParMat(A.nRowGlobal, r, A.grid, 'C') 

    # Multiply local A with generated B
    t0 = MPI.Wtime()

    Y.localMat = np.matmul(targetA, targetB, order='F')

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time for first local multiply:", t1-t0, "sec")

    # Second matmul in the Nystrom method
    # Number of columns needed from B.T is the number of rows in local Y
    myrank = Y.grid.myrank
    if myrank < (Y.grid.nprocs - 1) :
        colStart = myrank * (Y.nRowGlobal // Y.grid.nprocs)
        colEnd = (myrank + 1) * (Y.nRowGlobal // Y.grid.nprocs)
    elif myrank == (Y.grid.nprocs - 1) :
        colStart = myrank * (Y.nRowGlobal // Y.grid.nprocs)
        colEnd = Y.nRowGlobal 
    targetY = Y.localMat

    Z = ParMat(r, Y.nColGlobal, A.grid, 'B') # Use the same process grid as A. Grid does not change for A, B or C. Only the face of the grid change which is specific to the matrix
   
    t0 = MPI.Wtime()

    Z.localMat = np.matmul(np.swapaxes(targetB, 0, 1)[:, colStart:colEnd], targetY, order='F')
    
    t1 = MPI.Wtime()

    if (A.grid.myrank == 0):
        print("Time for second local multiply:", t1-t0, "sec")

    Z.localMat, rsTime, totTime = splitAndReduceScatter(Z.localMat, Z.grid.colWorld, split='col')
    
    if (A.grid.myrank == 0):
        print("Time to scatter and reduce Z:", rsTime, "sec")

    return Y, Z


def nystrom_1d_redist_1d(A, r):
    pass

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numrows", type=int, help="Number of rows and columns of the matrix", default=100)
    parser.add_argument("-r", "--rank", type=int, help="Value of rank for low rank approximation", default=20)
    parser.add_argument("-p1", "--p1", type=int, help="Number of process grid rows for first matmul")
    parser.add_argument("-p2", "--p2", type=int, help="Number of process grid cols for first matmul")
    parser.add_argument("-p3", "--p3", type=int, help="Number of process grid cols for first matmul")
    parser.add_argument("-alg", "--alg", type=str, help="Multiplication to use")
    args = parser.parse_args()

    n = args.numrows
    r = args.rank
    p1 = args.p1
    p2 = args.p2
    p3 = args.p3
    alg = args.alg

    myrank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()

    if myrank == 0:
        print(f"Nystrom approximation of {n}x{n} matrix to rank {r} on {p1}x{p2}x{p3} grid");
    
    p = nprocs
    assert(p == args.p1 * args.p2 * args.p3)
    grid = ProcGrid(args.p1, args.p2, args.p3)

    A = ParMat(n, n, grid, 'A')
    A.generate(dtype=np.float64)

    if alg == "nystrom-1d-noredist-1d":
        Y, Z = nystrom_1d_noredist_1d(A,r)
    elif alg == "nystrom-1d-redist-1d":
        Y, Z = nystrom_1d_redist_1d(A,r)
