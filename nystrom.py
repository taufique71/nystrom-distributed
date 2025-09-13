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
    A_new = A1 @ Ymat.T

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

def nystrom_1d_redist_2d(A, rowSplitsA1d, r, Y2d, rowSplitsY2d, colSplitsY2d, Z): 
    npDtype = A.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)
    
    # Only local copy of A would be used for 1d matmul
    targetA = A.localMat

    t0 = MPI.Wtime()

    targetB = np.zeros( (A.nColGlobal, r), dtype=np.float64, order='F')
    prng = Generator(Xoroshiro128(1234, plusplus=False))
    prng.random(targetB.shape, dtype=np.float64, out=targetB)

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time to generate Omega:", t1 - t0, "sec")

    Y = ParMat(A.nRowGlobal, r, A.grid, 'C') 

    # Multiply local A with generated B
    t0 = MPI.Wtime()

    Y.localMat = np.matmul(targetA, targetB, order='F')

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time for first dgemm:", t1-t0, "sec")

    t0 = MPI.Wtime()
    t2 = MPI.Wtime()
    # AlltoAll
    p2 = Y2d.grid.nProcCol
    p3 = Y2d.grid.nProcFib

    rowsToSend = rowSplitsA1d[A.grid.rankInColWorld]
    colSplitstToSend = np.array(colSplitsY2d)

    rowsToRecv = rowSplitsY2d[Y2d.grid.rankInRowWorld]
    rowSplitsToRecv = np.array(rowSplitsA1d)[Y2d.grid.rankInRowWorld * p3 : Y2d.grid.rankInRowWorld * p3 + p3]
    colsToRecv = colSplitsY2d[Y2d.grid.rankInFibWorld]

    sendcounts = colSplitstToSend * rowsToSend

    sdispls = np.zeros(p3, dtype=np.int32)
    np.cumsum(sendcounts[:-1], out=sdispls[1:])
    recvcounts = rowSplitsToRecv * colsToRecv

    rdispls = np.zeros(Y.grid.nprocs//p2, dtype=np.int32)
    np.cumsum(recvcounts[:-1], out=rdispls[1:])
    recvBuf = np.zeros((rowsToRecv * colsToRecv), dtype=np.float64, order='F')
    
    t3 = MPI.Wtime()
    tBuffPrep = t3-t2

    t2 = MPI.Wtime()
    Y2d.grid.fibWorld.Alltoallv([Y.localMat, (sendcounts, sdispls), MPI.DOUBLE],
               [recvBuf, (recvcounts, rdispls), MPI.DOUBLE])

    t3 = MPI.Wtime()
    tAlltoallv = t3-t2

    t2 = MPI.Wtime()
    Y2d.localMat = np.zeros( (rowsToRecv, colsToRecv), dtype=np.float64, order='F')

    cumSum = np.cumsum(rowSplitsToRecv)
    for p in range(len(rowSplitsToRecv)):
        for c in range(colsToRecv):
            targetY2drowS = cumSum[p] - rowSplitsToRecv[p]
            targetY2drowE = cumSum[p]
            if p == 0:
                recvBufS = c * rowSplitsToRecv[p]
                recBufE = recvBufS + rowSplitsToRecv[p]
            else:
                recvBufS = cumSum[p-1] * colsToRecv + c * rowSplitsToRecv[p]
                recBufE = recvBufS + rowSplitsToRecv[p]
            Y2d.localMat[targetY2drowS:targetY2drowE, c] = recvBuf[recvBufS:recBufE]

    t3 = MPI.Wtime()
    tUnpack = t3-t2

    t1 = MPI.Wtime()


    if (A.grid.myrank == 0):
        print("Time to redistribute Y:", t1 - t0, "sec")
        print("\tTime to prepare buffer for alltoallv:", tBuffPrep, "sec")
        print("\tTime to do alltoallv", tAlltoallv, "sec")
        print("\tTime to unpack:", tUnpack, "sec")
    # Second matmul in the Nystrom method
    s = np.sum(rowSplitsY2d[:Y2d.grid.rankInRowWorld])
    e = np.sum(rowSplitsY2d[:Y2d.grid.rankInRowWorld+1])

    t0 = MPI.Wtime()
    Z.localMat = np.matmul(targetB.T[:, s:e], Y2d.localMat, order='C')
    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time for second dgemm:", t1-t0, "sec")


    Z.localMat, rsTime, totTime = splitAndReduceScatter(Z.localMat, Z.grid.rowWorld, split='row')
    if (A.grid.myrank == 0):
        print("Time for reduce scatter Z:", rsTime, "sec")
    return Y, Z

def nystrom_2d_redist_1d(A, colSplitsA2d, r, colSplitsY2d, Y1d, rowSplitsY1d, Z):
    targetA = A.localMat
    # print("Local A, Rank", A.grid.myrank, targetA)

    # Create distributed B matrix with appropriate dimension 
    t0 = MPI.Wtime()

    B = np.zeros( (A.nColGlobal, r), dtype=np.float64, order='F')
    prng = Generator(Xoroshiro128(1234, plusplus=False))
    prng.random(B.shape, dtype=np.float64, out=B)

    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time to generate Omega:", t1 - t0, "sec")

    
    # Processes in 2D grid
    p2 = A.grid.nProcCol 
    p3 = A.grid.nProcFib 

    cumSumColSplits = np.zeros(p3+1, dtype=int)
    cumSumColSplits[1:] = np.cumsum(colSplitsA2d)
    rFib = A.grid.rankInFibWorld
    BrowStart = cumSumColSplits[rFib]
    BrowEnd = cumSumColSplits[rFib+1]
    targetB = B[BrowStart:BrowEnd, :]

    Y2d = ParMat(A.nRowGlobal, r, A.grid, 'B') 
    # First matmul
    t0 = MPI.Wtime()
    Y2d.localMat = np.matmul(targetA, targetB, order='F')
    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time for first dgemm:", t1-t0, "sec")
    
    Y2d.localMat, rsTime, totTime = Reduce_ScallaPackScatter(Y2d.localMat, Y2d.grid.fibWorld, colSplitsY2d, split='col')
    if (A.grid.myrank == 0):
        print("Time for first reduce scatter", rsTime, "sec")


    t0 = MPI.Wtime()
    rowSplitsToSend = rowSplitsY1d[Y2d.grid.rankInRowWorld * p3 : Y2d.grid.rankInRowWorld * p3 + p3]# np.array(ScaLAPACK(rowsToSend, p3))
    colsToSend = colSplitsY2d[Y2d.grid.rankInFibWorld]

    # Compute sendcounts and displacements
    sendcounts = rowSplitsToSend * colsToSend
    sdispls = np.zeros_like(sendcounts)
    np.cumsum(sendcounts[:-1], out=sdispls[1:])

    # Get how many rows and cols each rank in 1D grid should receive
    recvRowSplits = rowSplitsY1d 
    colSplits = np.array(colSplitsY2d)
    rowsToRecv = recvRowSplits[Y1d.grid.rankInColWorld]
    colsToRecv = r

    recvcounts = colSplits * rowsToRecv
    rdispls = np.zeros_like(recvcounts)
    np.cumsum(recvcounts[:-1], out=rdispls[1:])

    # Flatten 2D localMat for sending
    flatSendBuf = Y2d.localMat.flatten(order='C')
    recvBuf = np.zeros(rowsToRecv * colsToRecv, dtype=np.float64, order='F')
    t1 = MPI.Wtime()

    # Alltoallv from 2D grid to 1D grid
    Y2d.grid.fibWorld.Alltoallv([flatSendBuf, (sendcounts, sdispls), MPI.DOUBLE],
                                [recvBuf, (recvcounts, rdispls), MPI.DOUBLE])


    t2 = MPI.Wtime()
    tAlltoallv = t2 - t0
    tBuffPrep = t1 - t0

    if (A.grid.myrank == 0):
        print("\tTime to prepare buffer for alltoallv:", tBuffPrep, "sec")
        print("\tTime to do alltoallv", tAlltoallv, "sec")
    
    t0 = MPI.Wtime()
    # Reshape received buffer into matrix
    Y1d.localMat = np.zeros((rowsToRecv, colsToRecv), dtype=np.float64, order='F')

    cumSum = np.cumsum(colSplits)
    for p in range(p3):
        for r in range(rowsToRecv):
            targetY1DcolS = cumSum[p] - colSplits[p]
            targetY1DcolE = cumSum[p]

            recvBufS = rdispls[p] + r * colSplits[p]
            recvBufE = recvBufS + colSplits[p]
            
            Y1d.localMat[r, targetY1DcolS:targetY1DcolE] = recvBuf[recvBufS:recvBufE]

    cumSum = np.zeros(p2 * p3 + 1, dtype=int)
    cumSum[1:] = np.cumsum(recvRowSplits)
    colStart = cumSum[Y1d.grid.myrank]
    colEnd = cumSum[Y1d.grid.myrank + 1]
    t1 = MPI.Wtime()
    if (A.grid.myrank == 0):
        print("Time to unpack after alstoall", t1-t0, "sec")


    # Second matmul
    t0 = MPI.Wtime()
    Z.localMat = np.matmul(np.swapaxes(B, 0, 1)[:, colStart:colEnd], Y1d.localMat, order='F')
    t1 = MPI.Wtime()

    if (A.grid.myrank == 0):
        print("Time for second dgemm:", t1-t0, "sec")

    Z.localMat, rsTime, totTime = splitAndReduceScatter(Z.localMat, Z.grid.colWorld, split='col')

    if (A.grid.myrank == 0):
        print("Time for second reduce scatter", rsTime, "sec")

    return Y1d, Z

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


    dataPath = "/deac/csc/ballardGrp/rahmm224/nystrom/nystrom-distributed/data/cifar10.bin"
    if alg == "nystrom-1d-noredist-1d":
        A = ParMat(n, n, grid1, 'A', dtype=np.float64)
        A.localMat = getCIFAR10GramMatrix(path = dataPath, world = MPI.COMM_WORLD, n = n, nRowLocal = A.nRowLocal, nColLocal = A.nColLocal, localRowStart = A.localRowStart, localColStart = A.localColStart)

        Y = ParMat(n, r, grid1, 'C') # Distribute Y, the outcome of matmul1 to the C face of grid1
        Z = ParMat(r, r, grid1, 'B') # Distribute Z to B face of the grid1 to avoid memory access
        nystrom_1d_noredist_1d(A,r,Y,Z)
    elif alg == "nystrom-1d-redist-1d":
        A = ParMat(n, n, grid1, 'A', dtype=np.float64)
        A.localMat = getCIFAR10GramMatrix(path = dataPath, world = MPI.COMM_WORLD, n = n, nRowLocal = A.nRowLocal, nColLocal = A.nColLocal, localRowStart = A.localRowStart, localColStart = A.localColStart)

        Y = ParMat(n, r, grid2, 'B')
        Z = ParMat(r, r, grid2, 'C') 
        nystrom_1d_redist_1d(A,r,Y,Z)

    elif alg == "nystrom-1d-redist-2d":
        A = ParMat(n, n, grid1, 'A', dtype=np.float64)

        rowSplitsY2d = ScaLAPACK(n, matmul2p2)
        colSplitsY2d = ScaLAPACK(r, matmul2p3)
        rowSplitsA1d = hierarchical_dist(n, matmul2p2, matmul2p3)
        nRowLocal = hierarchical_dist(n, matmul2p2, matmul2p3)[A.grid.rankInColWorld]
        
        A.localMat = getCIFAR10GramMatrix(path = dataPath, world = MPI.COMM_WORLD, n = n, nRowLocal = A.nRowLocal, nColLocal = A.nColLocal, localRowStart = A.localRowStart, localColStart = A.localColStart)

        Y = ParMat(n, r, grid2, 'B')
        Z = ParMat(r, r, grid2, 'B') 
        Y, Z = nystrom_1d_redist_2d(A, rowSplitsA1d, r, Y, rowSplitsY2d, colSplitsY2d , Z)

    elif alg == "nystrom-2d-redist-1d":
        A = ParMat(n, n, grid1, 'B', dtype=np.float64)

        rowSplitsA2d = ScaLAPACK(n, matmul1p2)
        colSplitsA2d = ScaLAPACK(n, matmul1p3)
        colSplitsY2d = ScaLAPACK(r, matmul1p3)
        rowSplitsY1d = hierarchical_dist(n, matmul1p2, matmul1p3)

        nRowLocal = rowSplitsA2d[A.grid.rankInRowWorld]
        nColLocal = colSplitsA2d[A.grid.rankInFibWorld]
        # A.gen_A_2d1d(p2, p3, nRowLocal, nColLocal,  dtype=np.float64)
        

        blockStartRows = np.zeros(matmul1p2, dtype=int)
        blockStartRows[1:] = np.cumsum(rowSplitsA2d[:-1])
        blockStartCols = np.zeros(matmul1p3, dtype=int)
        blockStartCols[1:] = np.cumsum(colSplitsA2d[:-1])
        localRowStart = blockStartRows[A.grid.rankInRowWorld]
        localColStart = blockStartCols[A.grid.rankInFibWorld]

        # A.gen_symm_pos_semidef(nRowLocal, nColLocal, localRowStart, localColStart, r, dtype=np.float64)
        A.localMat = getCIFAR10GramMatrix(path = dataPath, world = MPI.COMM_WORLD, n = n, nRowLocal = nRowLocal, nColLocal = nColLocal, localRowStart = localRowStart, localColStart = localColStart)

        Y = ParMat(n, r, grid2, 'A')
        Z = ParMat(r, r, grid2, 'B') 
        Y, Z = nystrom_2d_redist_1d(A, colSplitsA2d, r, colSplitsY2d, Y, rowSplitsY1d, Z)
    # Amat = A.allGather()
    # Ymat = Y.allGather()
    # Zmat = Z.allGather()

    # if A.grid.myrank == 0:
        # checkCorrectness(Amat, r, Ymat, Zmat)

