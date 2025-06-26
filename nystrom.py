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

def nystrom_1d_noredist_1d(A, r):
    pass

def nystrom_1d_redist_1d(A, r):
    pass

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
        print(f"testing {n1}x{n2} with {n2}x{n3} on {p1}x{p2}x{p3} grid")
        print(f"Nystrom approximation of {n}x{n} matrix to rank {r} on {p1}x{p2}x{p3} grid");
    
    p = nprocs
    assert(p == args.p1 * args.p2 * args.p3)
    grid = ProcGrid(args.p1, args.p2, args.p3)

    A = ParMat(n1, n2, grid, 'A')
    A.generate(dtype=np.float64)


    if alg == "nystrom-1d-noredist-1d":
        nystrom_1d_noredist_1d(A,r)
    elif alg == "nystrom-1d-redist-1d":
        nystrom_1d_redist_1d(A,r)
