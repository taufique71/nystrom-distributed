from mpi4py import MPI
import numpy as np
# import torch  
# import torch.distributed as torchdist  

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
        
    # def __init__(self, nProcRow, nProcCol, nProcFib):
        # self.nProcRow = nProcRow
        # self.nProcCol = nProcCol
        # self.nProcFib = nProcFib
        # self.myrank = MPI.COMM_WORLD.Get_rank()
        # self.nprocs = MPI.COMM_WORLD.Get_size()

        # if(self.myrank == 0):
            # print("Grid: ", self.nProcRow, "x", self.nProcCol, "x", self.nProcFib)

        # self.rankInRowGroup = None
        # self.rankInColGroup = None
        # self.rankInFibGroup = None

        # self.rowRank = None
        # self.colRank = None
        # self.fibRank = None

        # self.rowGroup = None
        # self.colGroup = None
        # self.fibGroup = None

        # torchdist.init_process_group(
            # backend='mpi',  #'nccl' for GPU or 'gloo' for CPU
            # init_method='env://',  # or 'tcp://<ip>:<port>' for manual setup
            # world_size=self.nprocs,  # Total number of processes
            # rank=self.myrank  # Unique rank for each process
        # )

        # self.rowRank, self.colRank, self.fibRank = torch.unravel_index(torch.tensor(self.myrank), (self.nProcRow, self.nProcCol, self.nProcFib))
        # allRowRanks, allColRanks, allFibRanks = torch.unravel_index(torch.tensor( range(0, self.nprocs) ), (self.nProcRow, self.nProcCol, self.nProcFib))
        
        # # ranksInMyRow = []
        # # ranksInMyCol = []
        # # ranksInMyFib = []
        # # for i in range(0, self.nprocs):
            # # if (allRowRanks[i] == rankInRow) and (allFibRanks[i] == rankInFib):
                # # ranksInMyRow.append(i)
            # # if (allColRanks[i] == rankInCol) and (allFibRanks[i] == rankInFib):
                # # ranksInMyCol.append(i)
            # # if (allRowRanks[i] == rankInRow) and (allColRanks[i] == rankInCol):
                # # ranksInMyFib.append(i)

        # # rowGroupRanks = [ [] for _ in range(self.nProcRow) ]
        # # colGroupRanks = [ [] for _ in range(self.nProcCol) ]
        # # fibGroupRanks = [ [] for _ in range(self.nProcFib) ]

        # rowGroupRanks = []
        # for i in range(self.nProcRow):
            # rowGroupRanks.append([])
            # for j in range(self.nProcFib):
                # rowGroupRanks[i].append([])

        # colGroupRanks = []
        # for i in range(self.nProcCol):
            # colGroupRanks.append([])
            # for j in range(self.nProcFib):
                # colGroupRanks[i].append([])

        # fibGroupRanks = []
        # for i in range(self.nProcRow):
            # fibGroupRanks.append([])
            # for j in range(self.nProcCol):
                # fibGroupRanks[i].append([])

        # for i in range(0, self.nprocs):
            # a, b, c = torch.unravel_index(torch.tensor(i), (self.nProcRow, self.nProcCol, self.nProcFib))
            # rowGroupRanks[a][c].append(i)
            # colGroupRanks[b][c].append(i)
            # fibGroupRanks[a][b].append(i)

        # for a in range(self.nProcRow):
            # for c in range(self.nProcFib):
                # g = torchdist.new_group(rowGroupRanks[a][c])
                # if a == self.rowRank and c == self.fibRank:
                    # self.rowGroup = g
        # for b in range(self.nProcCol):
            # for c in range(self.nProcFib):
                # g = torchdist.new_group(colGroupRanks[b][c])
                # if b == self.colRank and c == self.fibRank:
                    # self.colGroup = g
        # for a in range(self.nProcRow):
            # for b in range(self.nProcCol):
                # g = torchdist.new_group(fibGroupRanks[a][b])
                # if a == self.rowRank and b == self.colRank:
                    # self.fibGroup = g

        # self.rankInRowGroup = torchdist.get_group_rank(self.rowGroup, self.myrank)
        # self.rankInColGroup = torchdist.get_group_rank(self.colGroup, self.myrank)
        # self.rankInFibGroup = torchdist.get_group_rank(self.fibGroup, self.myrank)

        # # ranksInMyRowGroup = torchdist.get_process_group_ranks(self.rowGroup)
        # # ranksInMyColGroup = torchdist.get_process_group_ranks(self.colGroup)
        # # ranksInMyFibGroup = torchdist.get_process_group_ranks(self.fibGroup)

        # # print(self.myrank, ":", self.rowRank, "=", self.rankInColGroup, ",", self.colRank, "=", self.rankInRowGroup, ",", self.fibRank, "=", self.rankInFibGroup)
        # # print(self.myrank, ":", ranksInMyRowGroup, ranksInMyColGroup, ranksInMyFibGroup)

    
    # def cleanup(self):
        # torchdist.destroy_process_group()
