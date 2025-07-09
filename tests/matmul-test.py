import os
import sys
from os import path
import argparse
from mpi4py import MPI
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from matrix import matmul
from matrix import matmul1_gen
from matrix import matmul1_comm
from communicator import ProcGrid
from matrix import ParMat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--p1", type=int, help="Number of process grid rows")
    parser.add_argument("-p2", "--p2", type=int, help="Number of process grid columns")
    parser.add_argument("-p3", "--p3", type=int, help="Number of process grid fibers")
    parser.add_argument("-n1", "--n1", type=int, help="Number of rows in A")
    parser.add_argument("-n2", "--n2", type=int, help="Number of columns in A / Number of rows in B")
    parser.add_argument("-n3", "--n3", type=int, help="Number of columns in B")
    parser.add_argument("-alg", "--alg", type=str, help="Multiplication to use")
    args = parser.parse_args()

    p1 = args.p1
    p2 = args.p2
    p3 = args.p3
    n1 = args.n1
    n2 = args.n2
    n3 = args.n3
    alg = args.alg

    myrank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    
    if myrank == 0:
        print(f"testing {n1}x{n2} with {n2}x{n3} on {p1}x{p2}x{p3} grid")
    
    grid = ProcGrid(p1, p2, p3)
    A = ParMat(n1, n2, grid, 'A')
    A.generate()

    B = ParMat(n2, n3, grid, 'B')
    B.generate()

    C = None

    if alg == "matmul":
        C = matmul(A,B)
    elif alg == "matmul1gen":
        C = matmul1_gen(A, B, "xoroshiro")
    elif alg == "matmul1comm":
        C = matmul1_comm(A, B, "xoroshiro")
