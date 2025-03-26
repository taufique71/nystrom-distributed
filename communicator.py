from mpi4py import MPI
import torch  
import torch.distributed as torchdist  

class ProcGrid:
    def __init__(self, nProcRow, nProcCol, nProcFib):
        self.nProcRow = nProcRow
        self.nProcCol = nProcCol
        self.nProcFib = nProcFib
        self.myrank = MPI.COMM_WORLD.Get_rank()
        self.nprocs = MPI.COMM_WORLD.Get_size()

        self.rankInRowGroup = None
        self.rankInColGroup = None
        self.rankInFibGroup = None

        self.rowRank = None
        self.colRank = None
        self.fibRank = None

        self.rowGroup = None
        self.colGroup = None
        self.fibGroup = None

        torchdist.init_process_group(
            backend='mpi',  #'nccl' for GPU or 'gloo' for CPU
            init_method='env://',  # or 'tcp://<ip>:<port>' for manual setup
            world_size=self.nprocs,  # Total number of processes
            rank=self.myrank  # Unique rank for each process
        )

        self.rowRank, self.colRank, self.fibRank = torch.unravel_index(torch.tensor(self.myrank), (self.nProcRow, self.nProcCol, self.nProcFib))
        allRowRanks, allColRanks, allFibRanks = torch.unravel_index(torch.tensor( range(0, self.nprocs) ), (self.nProcRow, self.nProcCol, self.nProcFib))
        
        # ranksInMyRow = []
        # ranksInMyCol = []
        # ranksInMyFib = []
        # for i in range(0, self.nprocs):
            # if (allRowRanks[i] == rankInRow) and (allFibRanks[i] == rankInFib):
                # ranksInMyRow.append(i)
            # if (allColRanks[i] == rankInCol) and (allFibRanks[i] == rankInFib):
                # ranksInMyCol.append(i)
            # if (allRowRanks[i] == rankInRow) and (allColRanks[i] == rankInCol):
                # ranksInMyFib.append(i)

        rowGroupRanks = [ [] for _ in range(self.nProcRow) ]
        colGroupRanks = [ [] for _ in range(self.nProcCol) ]
        fibGroupRanks = [ [] for _ in range(self.nProcFib) ]

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
        for i in range(self.nProcFib):
            fibGroupRanks.append([])
            for j in range(self.nProcRow):
                fibGroupRanks[i].append([])

        for i in range(0, self.nprocs):
            a, b, c = torch.unravel_index(torch.tensor(i), (self.nProcRow, self.nProcCol, self.nProcFib))
            rowGroupRanks[a][c].append(i)
            colGroupRanks[b][c].append(i)
            fibGroupRanks[c][a].append(i)

        for a in range(self.nProcRow):
            for c in range(self.nProcFib):
                g = torchdist.new_group(rowGroupRanks[a][c])
                if a == self.rowRank and c == self.fibRank:
                    self.rowGroup = g
        for b in range(self.nProcCol):
            for c in range(self.nProcFib):
                g = torchdist.new_group(colGroupRanks[b][c])
                if b == self.colRank and c == self.fibRank:
                    self.colGroup = g
        for c in range(self.nProcFib):
            for a in range(self.nProcRow):
                g = torchdist.new_group(fibGroupRanks[c][a])
                if c == self.fibRank and a == self.rowRank:
                    self.fibGroup = g

        self.rankInRowGroup = torchdist.get_group_rank(self.rowGroup, self.myrank)
        self.rankInColGroup = torchdist.get_group_rank(self.colGroup, self.myrank)
        self.rankInFibGroup = torchdist.get_group_rank(self.fibGroup, self.myrank)

        print(self.myrank, ":", self.rowRank, "=", self.rankInColGroup, ",", self.colRank, "=", self.rankInRowGroup, "," self.fibRank, "=", self.rankInFibGroup)

    
    def cleanup(self):
        torchdist.destroy_process_group()
