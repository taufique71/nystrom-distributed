import torch  
import torch.distributed as torchdist  
from communicator import ProcGrid

class ParMat:
    def __init__(self, m, n, grid, frontFace):
        self.nRowGlobal = m
        self.nColGlobal = n
        self.grid = grid
        self.frontFace = frontFace
        
        if self.frontFace == 'A':
            # Rows
            self.nRowLocal = int(self.nRowGlobal / self.grid.nProcRow)
            if self.grid.rankInColGroup == self.grid.nProcRow-1:
                # Last process in a column group would get different treatment
                self.nRowLocal = self.nRowGlobal - int(self.nRowGlobal / self.grid.nProcRow) * self.grid.rankInColGroup

            # Columns
            self.nColLocal = int(self.nColGlobal / self.grid.nProcCol)
            if self.grid.rankInRowGroup == self.grid.nProcCol-1:
                # Last process in a row group would get different treatment
                self.nColLocal = self.nColGlobal - int(self.nColGlobal / self.grid.nProcCol) * self.grid.rankInRowGroup

            # Further distribute the columns to fibers
            if self.grid.rankInFibGroup < self.grid.nProcFib-1:
                self.nColLocal = int(self.nColLocal / self.grid.nProcFib)
            elif self.grid.rankInFibGroup == self.grid.nProcFib-1:
                # Treat the last process in the fiber differently
                # x = self.nColLocal
                self.nColLocal = self.nColLocal - int(self.nColLocal / self.grid.nProcFib) * self.grid.rankInFibGroup


        elif self.frontFace == 'B':
            # Now grid columns become equivalent to grid rows, grid fibers become equivalent to grid columns and grid rows become equivalent to grid fibers
            # Rows
            self.nRowLocal = int(self.nRowGlobal / self.grid.nProcCol)
            if self.grid.rankInRowGroup == self.grid.nProcCol-1:
                # Last process in a column group would get different treatment
                self.nRowLocal = self.nRowGlobal - int(self.nRowGlobal / self.grid.nProcCol) * self.grid.rankInRowGroup

            # Columns
            self.nColLocal = int(self.nColGlobal / self.grid.nProcFib)
            if self.grid.rankInFibGroup == self.grid.nProcFib-1:
                # Last process in a row group would get different treatment
                self.nColLocal = self.nColGlobal - int(self.nColGlobal / self.grid.nProcFib) * self.grid.rankInFibGroup

            # Further distribute the columns to grid rows
            if self.grid.rankInColGroup < self.grid.nProcRow-1:
                self.nColLocal = int(self.nColLocal / self.grid.nProcRow)
            elif self.grid.rankInColGroup == self.grid.nProcRow-1:
                # Treat the last process in the fiber differently
                # x = self.nColLocal
                self.nColLocal = self.nColLocal - int(self.nColLocal / self.grid.nProcRow) * self.grid.rankInColGroup

        # elif self.frontFace == 'C':
            # self.nRowLocal = self.nRowGlobal / self.grid.nProcRow
            # self.nColLocal = self.nColGlobal / self.grid.nProcFib

        # print(f"globalrank = {self.grid.myrank}, row_rank={self.grid.rowRank}, col_rank={self.grid.colRank}, fib_rank={self.grid.fibRank}, local_mat={self.nRowLocal}x{self.nColLocal}")


