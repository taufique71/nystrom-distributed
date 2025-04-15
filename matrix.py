from mpi4py import MPI
from communicator import ProcGrid
import numpy as np

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

        


