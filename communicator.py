from mpi4py import MPI
import numpy as np

class ProcGrid:
    def __init__(self, nProcRow, nProcCol, nProcFib):
        self.nProcRow = nProcRow
        self.nProcCol = nProcCol
        self.nProcFib = nProcFib
        self.myrank = MPI.COMM_WORLD.Get_rank()
        self.nprocs = MPI.COMM_WORLD.Get_size()

        if(self.myrank == 0):
            print("Grid: ", self.nProcRow, "x", self.nProcCol, "x", self.nProcFib)

        # Initialize with the default group involved with MPI.COMM_WORLD
        # Would be updated later
        self.rowGroup = MPI.COMM_WORLD.Get_group()
        self.colGroup = MPI.COMM_WORLD.Get_group()
        self.fibGroup = MPI.COMM_WORLD.Get_group()
        
        # Actual MPI communicators from the MPI groups
        self.rowWorld = None
        self.colWorld = None
        self.fibWorld = None
        
        # Ranks in the row, column and fiber communicator groups
        self.rankInRowWorld = None
        self.rankInColWorld = None
        self.rankInFibWorld = None

        self.rowRank, self.colRank, self.fibRank = np.unravel_index(self.myrank, (self.nProcRow, self.nProcCol, self.nProcFib), order='C')
        allRowRanks, allColRanks, allFibRanks = np.unravel_index(range(0, self.nprocs), (self.nProcRow, self.nProcCol, self.nProcFib), order='C')
        # print(f"myrank:={self.myrank}, rowRank={self.rowRank}, colRank={self.colRank}, fibRank={self.fibRank}")

        rowGroupRanks = []
        for i in range(self.nProcRow):
            rowGroupRanks.append([])
            for j in range(self.nProcFib):
                rowGroupRanks[i].append([])

        colGroupRanks = []
        for i in range(self.nProcCol):
            colGroupRanks.append([])
            for j in range(self.nProcFib):
                colGroupRanks[i].append([])

        fibGroupRanks = []
        for i in range(self.nProcRow):
            fibGroupRanks.append([])
            for j in range(self.nProcCol):
                fibGroupRanks[i].append([])

        for i in range(0, self.nprocs):
            a, b, c = np.unravel_index(i, (self.nProcRow, self.nProcCol, self.nProcFib), order='C')
            rowGroupRanks[a][c].append(i)
            colGroupRanks[b][c].append(i)
            fibGroupRanks[a][b].append(i)
        
        # if self.myrank == 16:
            # print(rowGroupRanks[self.rowRank][self.fibRank])
            # print(colGroupRanks[self.colRank][self.fibRank])
            # print(fibGroupRanks[self.rowRank][self.colRank])

        # https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Group.html#mpi4py.MPI.Group.Incl
        # https://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/
        self.rowGroup = self.rowGroup.Incl(rowGroupRanks[self.rowRank][self.fibRank])
        self.rowWorld = MPI.COMM_WORLD.Create(self.rowGroup)
        self.colGroup = self.colGroup.Incl(colGroupRanks[self.colRank][self.fibRank])
        self.colWorld = MPI.COMM_WORLD.Create(self.colGroup)
        self.fibGroup = self.fibGroup.Incl(fibGroupRanks[self.rowRank][self.colRank])
        self.fibWorld = MPI.COMM_WORLD.Create(self.fibGroup)

        self.rankInRowWorld = self.rowWorld.Get_rank();
        self.rankInColWorld = self.colWorld.Get_rank();
        self.rankInFibWorld = self.fibWorld.Get_rank();
