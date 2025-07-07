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

def checkCorrectness(Amat, r, Ymat, Zmat):
    Bmat = np.zeros( (A.nColGlobal, r), dtype=np.float64, order='F')
    prng = Generator(Xoroshiro128(1234, plusplus=False))
    prng.random(Bmat.shape, dtype=np.float64, out=Bmat) 

    # First multiplication
    if np.allclose( np.matmul(Amat, Bmat), Ymat):
        print("First multiplication is Correct")
    else:
        print("First multiplication is Incorrect")

    # Second multiplication
    # print(Bmat.shape, Ymat.shape, Zmat.shape)
    if np.allclose( np.matmul(Bmat.T, Ymat), Zmat):
        print("Second multiplication is Correct")
    else:
        print("Second multiplication is Incorrect")
    
    # Nystrom Approximation
    A1 = Ymat @ np.linalg.pinv(Zmat)
    A2 = Bmat.T @ Amat
    A_new = A1 @ A2

    if np.allclose(A_new, Amat):
        print("Nystrom Correct")
    else:
        print("Nystrom Incorrect")

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

    # AlltoAll
    # Before communication
    rowsInLastProc = int(Y.nRowGlobal - (Y.nRowGlobal//Y.grid.nprocs) * (Y.grid.nprocs - 1))
    rowsInOtherProc = int(Y.nRowGlobal//Y.grid.nprocs)
    # After communication
    colsInLastProc = int(r - (r//Y.grid.nprocs) * (Y.grid.nprocs - 1))
    colsInOtherProc = r//Y.grid.nprocs

    if Y.grid.myrank < Y.grid.nprocs - 1:
        recvCols = colsInOtherProc
        sendcounts = [rowsInOtherProc * colsInOtherProc] * (Y.grid.nprocs - 1) + [rowsInOtherProc * colsInLastProc]
        # sdispls = np.arange(0, rowsInOtherProc * colsInOtherProc * Y.grid.nprocs, rowsInOtherProc * colsInOtherProc)
        sdispls = np.zeros(Y.grid.nprocs, dtype=np.int32)
        np.cumsum(sendcounts[:-1], out=sdispls[1:])
        recvcounts = [rowsInOtherProc * colsInOtherProc] * (Y.grid.nprocs - 1) + [rowsInLastProc * colsInOtherProc]
        # rdispls = np.arange(0, rowsInOtherProc * colsInOtherProc * Y.grid.nprocs, rowsInOtherProc * colsInOtherProc)
        rdispls = np.zeros(Y.grid.nprocs, dtype=np.int32)
        np.cumsum(recvcounts[:-1], out=rdispls[1:])
    else:
        recvCols = colsInLastProc
        sendcounts = [rowsInLastProc * colsInOtherProc] * (Y.grid.nprocs - 1) + [rowsInLastProc * colsInLastProc]
        # sdispls = np.arange(0, rowsInLastProc * colsInOtherProc * Y.grid.nprocs, rowsInLastProc * colsInOtherProc)
        sdispls = np.zeros(Y.grid.nprocs, dtype=np.int32)
        np.cumsum(sendcounts[:-1], out=sdispls[1:])
        recvcounts = [rowsInOtherProc * colsInLastProc] * (Y.grid.nprocs - 1) + [rowsInLastProc * colsInLastProc]
        # rdispls = np.arange(0, rowsInOtherProc * colsInLastProc * Y.grid.nprocs, rowsInOtherProc * colsInLastProc)
        rdispls = np.zeros(Y.grid.nprocs, dtype=np.int32)
        np.cumsum(recvcounts[:-1], out=rdispls[1:])


    recvBuf = np.zeros( (Y.nRowGlobal * recvCols), dtype=np.float64, order='F')

    Y.grid.colWorld.Alltoallv([Y.localMat, (sendcounts, sdispls), MPI.DOUBLE],
               [recvBuf, (recvcounts, rdispls), MPI.DOUBLE])

    targetY = np.zeros( (A.nRowGlobal , recvCols), dtype=np.float64, order='F')
    blockRows = [0] + [rowsInOtherProc] * (Y.grid.nprocs - 1)
    for c in range(recvCols):
        for p in range(Y.grid.nprocs):
            targetYrowS = p * blockRows[p]
            targetYrowE = targetYrowS +blockRows[p]
            recvBufS = p * blockRows[p] * recvCols + c * blockRows[p]
            recBufE = recvBufS + blockRows[p]
            targetY[targetYrowS:targetYrowE, c] = recvBuf[recvBufS:recBufE]
    
    # Second matmul in the Nystrom method

    Z = ParMat(r, Y.nColGlobal, A.grid, 'B') # Use the same process grid as A. Grid does not change for A, B or C. Only the face of the grid change which is specific to the matrix
   
    t0 = MPI.Wtime()

    Z.localMat = np.matmul(np.swapaxes(targetB, 0, 1), targetY, order='F')
    
    t1 = MPI.Wtime()

    if (A.grid.myrank == 0):
        print("Time for second local multiply:", t1-t0, "sec")

    return Y, Z

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
    # A.generate(dtype=np.float64)
    A.gen_symm_pos_semidef(rank=r, dtype=np.float64)

    if alg == "nystrom-1d-noredist-1d":
        Y, Z = nystrom_1d_noredist_1d(A,r)
    elif alg == "nystrom-1d-redist-1d":
        Y, Z = nystrom_1d_redist_1d(A,r)

    Amat = A.allGather()
    Ymat = Y.allGather()
    Zmat = Z.allGather()

    if A.grid.myrank == 0:
        checkCorrectness(Amat, r, Ymat, Zmat)

