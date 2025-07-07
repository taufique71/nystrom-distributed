from mpi4py import MPI
import numpy as np
from communicator import ProcGrid
from utils import *
from randomgen import Xoroshiro128
from numpy.random import Generator

class ParMat:
    def __init__(self, m, n, grid, frontFace):
        self.nRowGlobal = m
        self.nColGlobal = n
        self.nRowLocal = None
        self.nColLocal = None
        self.grid = grid
        self.frontFace = frontFace
        self.localRowStart = None
        self.localColStart = None
        self.localRowStart = None
        self.localColStart = None
        self.localMat = None
        
        if self.frontFace == 'A':
            # Distribute matrix rows
            self.nRowLocal = int(self.nRowGlobal / self.grid.nProcRow)
            self.localRowStart = int(self.nRowGlobal / self.grid.nProcRow) * self.grid.rankInColWorld
            if self.grid.rankInColWorld == self.grid.nProcRow-1:
                # Last process in a column group would get different treatment
                self.nRowLocal = self.nRowGlobal - int(self.nRowGlobal / self.grid.nProcRow) * self.grid.rankInColWorld

            # Distribute matrix columns
            self.localColStart = int(self.nColGlobal / self.grid.nProcCol) * self.grid.rankInRowWorld
            self.nColLocal = int(self.nColGlobal / self.grid.nProcCol)
            if self.grid.rankInRowWorld == self.grid.nProcCol-1:
                # Last process in a row group would get different treatment
                self.nColLocal = self.nColGlobal - int(self.nColGlobal / self.grid.nProcCol) * self.grid.rankInRowWorld

            # Further distribute matrix columns to fibers
            self.localColStart = self.localColStart + int(self.nColLocal / self.grid.nProcFib) * self.grid.rankInFibWorld
            if self.grid.rankInFibWorld < self.grid.nProcFib-1:
                self.nColLocal = int(self.nColLocal / self.grid.nProcFib)
            elif self.grid.rankInFibWorld == self.grid.nProcFib-1:
                # Treat the last process in the fiber differently
                self.nColLocal = self.nColLocal - int(self.nColLocal / self.grid.nProcFib) * self.grid.rankInFibWorld

        elif self.frontFace == 'B':
            # Now grid columns become equivalent to grid rows, grid fibers become equivalent to grid columns and grid rows become equivalent to grid fibers
            # Distribute matrix rows
            self.nRowLocal = int(self.nRowGlobal / self.grid.nProcCol)
            self.localRowStart = int(self.nRowGlobal / self.grid.nProcCol) * self.grid.rankInRowWorld
            if self.grid.rankInRowWorld == self.grid.nProcCol-1:
                # Last process in a column group would get different treatment
                self.nRowLocal = self.nRowGlobal - int(self.nRowGlobal / self.grid.nProcCol) * self.grid.rankInRowWorld

            # Distribute matrix columns
            self.localColStart = int(self.nColGlobal / self.grid.nProcFib) * self.grid.rankInFibWorld
            self.nColLocal = int(self.nColGlobal / self.grid.nProcFib)
            if self.grid.rankInFibWorld == self.grid.nProcFib-1:
                # Last process in a row group would get different treatment
                self.nColLocal = self.nColGlobal - int(self.nColGlobal / self.grid.nProcFib) * self.grid.rankInFibWorld

            # Further distribute matrix columns to grid fibers
            self.localColStart = self.localColStart + int(self.nColLocal / self.grid.nProcRow) * self.grid.rankInColWorld
            if self.grid.rankInColWorld < self.grid.nProcRow-1:
                self.nColLocal = int(self.nColLocal / self.grid.nProcRow)
            elif self.grid.rankInColWorld == self.grid.nProcRow-1:
                # Treat the last process in the fiber differently
                self.nColLocal = self.nColLocal - int(self.nColLocal / self.grid.nProcRow) * self.grid.rankInColWorld

        elif self.frontFace == 'C':
            # Now grid fibers become equivalent to grid columns whole grid rows stay equivalent to grid rows.
            # Distribute matrix rows
            self.nRowLocal = int(self.nRowGlobal / self.grid.nProcRow)
            self.localRowStart = int(self.nRowGlobal / self.grid.nProcRow) * self.grid.rankInColWorld
            if self.grid.rankInColWorld == self.grid.nProcRow-1:
                self.nRowLocal = self.nRowGlobal - int(self.nRowGlobal / self.grid.nProcRow) * self.grid.rankInColWorld

            # Distribute matrix columns
            self.localColStart = int(self.nColGlobal / self.grid.nProcFib) * self.grid.rankInFibWorld
            self.nColLocal = int(self.nColGlobal / self.grid.nProcFib)
            if self.grid.rankInFibWorld == self.grid.nProcFib-1:
                self.nColLocal = self.nColGlobal - int(self.nColGlobal / self.grid.nProcFib) * self.grid.rankInFibWorld

            # Further distribute matrix columns to grid fibers
            self.localColStart = self.localColStart + int(self.nColLocal / self.grid.nProcCol) * self.grid.rankInRowWorld
            if self.grid.rankInRowWorld < self.grid.nProcCol-1:
                self.nColLocal = int(self.nColLocal / self.grid.nProcCol)
            elif self.grid.rankInRowWorld == self.grid.nProcCol-1:
                # Treat the last process in the fiber differently
                self.nColLocal = self.nColLocal - int(self.nColLocal / self.grid.nProcCol) * self.grid.rankInRowWorld

        # print(f"globalrank = {self.grid.myrank}, row_rank={self.grid.rowRank}, col_rank={self.grid.colRank}, fib_rank={self.grid.fibRank}, local_mat={self.nRowLocal}x{self.nColLocal}")

    def generate(self, dtype=np.int32):
        self.localMat = np.zeros( (self.nRowLocal, self.nColLocal), dtype=dtype, order='F')
        for idxRowLocal in range(0, self.nRowLocal):
            for idxColLocal in range(0, self.nColLocal):
                idxRowGlobal = self.localRowStart + idxRowLocal
                idxColGlobal = self.localColStart + idxColLocal
                self.localMat[idxRowLocal,idxColLocal] = idxRowGlobal * self.nColGlobal + idxColGlobal
        # if (self.grid.myrank == 4):
            # print(self.localMat.shape)
            # print(self.localMat)

    def gen_symm_pos_semidef(self, rank=100, dtype=np.float64):
        elements = self.nRowGlobal * rank
        A = np.arange(elements)
        A = A.reshape(self.nRowGlobal, rank)
        A = A @ A.T
        self.localMat = np.zeros( (self.nRowLocal, self.nColLocal), dtype=dtype, order='F')
        for idxRowLocal in range(0, self.nRowLocal):
            for idxColLocal in range(0, self.nColLocal):
                idxRowGlobal = self.localRowStart + idxRowLocal
                idxColGlobal = self.localColStart + idxColLocal
                self.localMat[idxRowLocal,idxColLocal] = A[idxRowGlobal,idxColGlobal]

    def generate_rand(self,seed, dtype=np.float64, generator="xoroshiro"):
        self.localMat = np.zeros( (self.nRowLocal, self.nColLocal), dtype=dtype, order='F')
        prng = None
        if generator == 'xoroshiro':
            prng = Generator(Xoroshiro128(seed, plusplus=False))
        prng.random(self.localMat.shape, dtype=dtype, out=self.localMat)

    def allGather(self):
        if self.frontFace == 'A':
            x, agTime, agvTime, totTime = allGatherAndConcat(self.localMat, self.grid.fibWorld, concat='col')
            y, agTime, agvTime, totTime = allGatherAndConcat(x, self.grid.rowWorld, concat='col')
            z, agTime, agvTime, totTime = allGatherAndConcat(y, self.grid.colWorld, concat='row')
            return z
        elif self.frontFace == 'B':
            x, agTime, agvTime, totTime = allGatherAndConcat(self.localMat, self.grid.colWorld, concat='col')
            y, agTime, agvTime, totTime = allGatherAndConcat(x, self.grid.fibWorld, concat='col')
            z, agTime, agvTime, totTime = allGatherAndConcat(y, self.grid.rowWorld, concat='row')
            return z
        elif self.frontFace == 'C':
            x, agTime, agvTime, totTime = allGatherAndConcat(self.localMat, self.grid.rowWorld, concat='col')
            y, agTime, agvTime, totTime = allGatherAndConcat(x, self.grid.fibWorld, concat='col')
            z, agTime, agvTime, totTime = allGatherAndConcat(y, self.grid.colWorld, concat='row')
            return z


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
