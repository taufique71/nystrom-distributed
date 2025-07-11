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

def nystrom_1d_noredist_1d(A, r, Y, Z):
    npDtype = A.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)

    if (A.grid.myrank == 0):
        print(f"matmul1 in {A.grid.nProcRow}x{A.grid.nProcCol}x{A.grid.nProcFib} grid")
    
    # Only local copy of A would be used for 1d matmul
    targetA = A.localMat

    # Create distributed B matrix with appropriate dimension 

    t0 = MPI.Wtime()

    targetB = np.zeros( (A.nColGlobal, r), dtype=np.float64, order='F')
    prng = Generator(Xoroshiro128(1234, plusplus=False))
    prng.random(targetB.shape, dtype=np.float64, out=targetB)

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time to generate Omega:", t1 - t0, "sec")
    
    # Already defined outside the function and passed as parameter
    # Y = ParMat(A.nRowGlobal, r, A.grid, 'C') 

    # Multiply local A with generated B
    t0 = MPI.Wtime()

    Y.localMat = np.matmul(targetA, targetB, order='F')

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time for first dgemm:", t1-t0, "sec")

    if (A.grid.myrank == 0):
        print(f"matmul2 in {Y.grid.nProcRow}x{Y.grid.nProcCol}x{Y.grid.nProcFib} grid")

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

    # # Use the same process grid as A. Grid does not change for A, B or C. Only the face of the grid change which is specific to the matrix
    # Z = ParMat(r, Y.nColGlobal, A.grid, 'B') 
   
    t0 = MPI.Wtime()

    Z.localMat = np.matmul(np.swapaxes(targetB, 0, 1)[:, colStart:colEnd], targetY, order='F')
    
    t1 = MPI.Wtime()

    if (A.grid.myrank == 0):
        print("Time for second dgemm:", t1-t0, "sec")

    Z.localMat, rsTime, totTime = splitAndReduceScatter(Z.localMat, Z.grid.colWorld, split='col')
    
    if (A.grid.myrank == 0):
        print("Time to scatter and reduce Z:", rsTime, "sec")

    return Y, Z


def nystrom_1d_redist_1d(A, r, Y, Z):
    npDtype = A.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)

    if (A.grid.myrank == 0):
        print(f"matmul1 in {A.grid.nProcRow}x{A.grid.nProcCol}x{A.grid.nProcFib} grid")
    
    # Only local copy of A would be used for 1d matmul
    targetA = A.localMat

    # Create distributed B matrix with appropriate dimension 

    t0 = MPI.Wtime()

    targetB = np.zeros( (A.nColGlobal, r), dtype=np.float64, order='F')
    prng = Generator(Xoroshiro128(1234, plusplus=False))
    prng.random(targetB.shape, dtype=np.float64, out=targetB)

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time to generate Omega:", t1 - t0, "sec")

    Ytemp = ParMat(A.nRowGlobal, r, A.grid, 'C') 

    # Multiply local A with generated B
    t0 = MPI.Wtime()

    Ytemp.localMat = np.matmul(targetA, targetB, order='F')

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time for first dgemm:", t1-t0, "sec")

    # Second matmul: OmegaT x Y
    # Redistribute(copy) appropriate contents of Ytemp to Y 
    # Y is expected to be initialized with approriate process grid with appriate face, hence Y buffer is ready with appropriate size
    if (A.grid.myrank == 0):
        print(f"matmul2 in {Y.grid.nProcRow}x{Y.grid.nProcCol}x{Y.grid.nProcFib} grid")

    t0 = MPI.Wtime()
    t2 = MPI.Wtime()
    # AlltoAll
    # Before communication
    rowsInLastProc = int(Ytemp.nRowGlobal - (Ytemp.nRowGlobal//Ytemp.grid.nprocs) * (Ytemp.grid.nprocs - 1))
    rowsInOtherProc = int(Ytemp.nRowGlobal//Ytemp.grid.nprocs)
    # After communication
    colsInLastProc = int(r - (r//Ytemp.grid.nprocs) * (Ytemp.grid.nprocs - 1))
    colsInOtherProc = r//Ytemp.grid.nprocs

    if Ytemp.grid.myrank < Ytemp.grid.nprocs - 1:
        recvCols = colsInOtherProc
        sendcounts = [rowsInOtherProc * colsInOtherProc] * (Ytemp.grid.nprocs - 1) + [rowsInOtherProc * colsInLastProc]
        sendcounts = np.array(sendcounts)
        # sdispls = np.arange(0, rowsInOtherProc * colsInOtherProc * Ytemp.grid.nprocs, rowsInOtherProc * colsInOtherProc)
        sdispls = np.zeros(Ytemp.grid.nprocs, dtype=np.int32)
        np.cumsum(sendcounts[:-1], out=sdispls[1:])
        recvcounts = [rowsInOtherProc * colsInOtherProc] * (Ytemp.grid.nprocs - 1) + [rowsInLastProc * colsInOtherProc]
        recvcounts = np.array(recvcounts)
        # rdispls = np.arange(0, rowsInOtherProc * colsInOtherProc * Ytemp.grid.nprocs, rowsInOtherProc * colsInOtherProc)
        rdispls = np.zeros(Ytemp.grid.nprocs, dtype=np.int32)
        np.cumsum(recvcounts[:-1], out=rdispls[1:])
    else:
        recvCols = colsInLastProc
        sendcounts = [rowsInLastProc * colsInOtherProc] * (Ytemp.grid.nprocs - 1) + [rowsInLastProc * colsInLastProc]
        sendcounts = np.array(sendcounts)
        # sdispls = np.arange(0, rowsInLastProc * colsInOtherProc * Ytemp.grid.nprocs, rowsInLastProc * colsInOtherProc)
        sdispls = np.zeros(Ytemp.grid.nprocs, dtype=np.int32)
        np.cumsum(sendcounts[:-1], out=sdispls[1:])
        recvcounts = [rowsInOtherProc * colsInLastProc] * (Ytemp.grid.nprocs - 1) + [rowsInLastProc * colsInLastProc]
        recvcounts = np.array(recvcounts)
        # rdispls = np.arange(0, rowsInOtherProc * colsInLastProc * Ytemp.grid.nprocs, rowsInOtherProc * colsInLastProc)
        rdispls = np.zeros(Ytemp.grid.nprocs, dtype=np.int32)
        np.cumsum(recvcounts[:-1], out=rdispls[1:])


    recvBuf = np.zeros( (Ytemp.nRowGlobal * recvCols), dtype=np.float64, order='F')

    t3 = MPI.Wtime()
    tBuffPrep = t3-t2
    
    t2 = MPI.Wtime()
    Ytemp.grid.colWorld.Alltoallv([Ytemp.localMat, (sendcounts, sdispls), MPI.DOUBLE],
               [recvBuf, (recvcounts, rdispls), MPI.DOUBLE])
    t3 = MPI.Wtime()
    tAlltoallv = t3-t2
    
    t2 = MPI.Wtime()
    # targetY = np.zeros( (A.nRowGlobal , recvCols), dtype=np.float64, order='F')
    # if (A.grid.myrank == 0):
        # print("targetY", targetY.shape)
        # print("Y.localMat", Y.localMat.shape)
    blockRows = [0] + [rowsInOtherProc] * (Ytemp.grid.nprocs - 1)
    for c in range(recvCols):
        for p in range(Ytemp.grid.nprocs):
            targetYrowS = p * blockRows[p]
            targetYrowE = targetYrowS +blockRows[p]
            recvBufS = p * blockRows[p] * recvCols + c * blockRows[p]
            recBufE = recvBufS + blockRows[p]
            Y.localMat[targetYrowS:targetYrowE, c] = recvBuf[recvBufS:recBufE]
    t3 = MPI.Wtime()
    tUnpack = t3-t2

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time to redistribute Y:", t1 - t0, "sec")
        print("\tTime to prepare buffer for alltoallv:", tBuffPrep, "sec")
        print("\tTime to do alltoallv", tAlltoallv, "sec")
        print("\tTime to unpack:", tUnpack, "sec")
    

    # Use the same process grid as A. Grid does not change for A, B or C. Only the face of the grid change which is specific to the matrix
    # Z = ParMat(r, Y.nColGlobal, A.grid, 'B') 
   
    t0 = MPI.Wtime()

    Z.localMat = np.matmul(np.swapaxes(targetB, 0, 1), Y.localMat, order='F')
    
    t1 = MPI.Wtime()

    if (A.grid.myrank == 0):
        print("Time for second dgemm:", t1-t0, "sec")

    return Y, Z

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numrows", type=int, help="Number of rows and columns of the matrix", default=100)
    parser.add_argument("-r", "--rank", type=int, help="Value of rank for low rank approximation", default=20)
    parser.add_argument("-matmul1p1", "--matmul1p1", type=int, help="Number of process grid rows for first matmul")
    parser.add_argument("-matmul1p2", "--matmul1p2", type=int, help="Number of process grid cols for first matmul")
    parser.add_argument("-matmul1p3", "--matmul1p3", type=int, help="Number of process grid cols for first matmul")
    parser.add_argument("-matmul2p1", "--matmul2p1", type=int, help="Number of process grid rows for second matmul")
    parser.add_argument("-matmul2p2", "--matmul2p2", type=int, help="Number of process grid cols for second matmul")
    parser.add_argument("-matmul2p3", "--matmul2p3", type=int, help="Number of process grid cols for second matmul")
    parser.add_argument("-alg", "--alg", type=str, help="Multiplication to use")
    args = parser.parse_args()

    n = args.numrows
    r = args.rank
    matmul1p1 = args.matmul1p1
    matmul1p2 = args.matmul1p2
    matmul1p3 = args.matmul1p3
    matmul2p1 = args.matmul2p1
    matmul2p2 = args.matmul2p2
    matmul2p3 = args.matmul2p3
    alg = args.alg

    myrank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()

    if myrank == 0:
        print(f"Nystrom approximation of {n}x{n} matrix to rank {r} using {alg}");
    
    p = nprocs
    assert(p == args.matmul1p1 * args.matmul1p2 * args.matmul1p3)
    assert(p == args.matmul2p1 * args.matmul2p2 * args.matmul2p3)
    grid1 = ProcGrid(args.matmul1p1, args.matmul1p2, args.matmul1p3)
    grid2 = ProcGrid(args.matmul2p1, args.matmul2p2, args.matmul2p3)

    A = ParMat(n, n, grid1, 'A', dtype=np.float64)
    A.generate()
    # A.gen_symm_pos_semidef(rank=r, dtype=np.float64)

    if alg == "nystrom-1d-noredist-1d":
        Y = ParMat(n, r, grid1, 'C') # Distribute Y, the outcome of matmul1 to the C face of grid1
        Z = ParMat(r, r, grid1, 'B') # Distribute Z to B face of the grid1 to avoid memory access
        nystrom_1d_noredist_1d(A,r,Y,Z)
    elif alg == "nystrom-1d-redist-1d":
        Y = ParMat(n, r, grid2, 'B')
        Z = ParMat(r, r, grid2, 'C') 
        nystrom_1d_redist_1d(A,r,Y,Z)

    # Amat = A.allGather()
    # Ymat = Y.allGather()
    # Zmat = Z.allGather()

    # if A.grid.myrank == 0:
        # checkCorrectness(Amat, r, Ymat, Zmat)

