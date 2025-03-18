#!/usr/bin/env python
import os
import argparse
from mpi4py import MPI
import torch  
import torch.distributed as torchdist  
import torch.multiprocessing as mp

myrank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

def setup():
    # Initialize the process group
    torchdist.init_process_group(
        backend='gloo',  #'nccl' for GPU or 'gloo' for CPU
        init_method='env://',  # or 'tcp://<ip>:<port>' for manual setup
        world_size=nprocs,  # Total number of processes
        rank=myrank  # Unique rank for each process
    )

def cleanup():
    torchdist.destroy_process_group()

if __name__ == "__main__":
    setup()
    
    rank_tensor = torch.tensor([myrank], dtype=torch.int32)
    rank_list = [torch.zeros(1, dtype=torch.int32) for _ in range(nprocs)]
    # rank_list = torch.zeros(nprocs, dtype=torch.int32)
    # print(rank_list)
    torchdist.all_gather(rank_list, rank_tensor)
    print(myrank, rank_list)
    rank_sum = torch.tensor(rank_tensor)
    torchdist.all_reduce(rank_sum, op=torchdist.ReduceOp.SUM)
    print(myrank, rank_sum)

    cleanup()
