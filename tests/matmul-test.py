import os
import sys
from os import path
import argparse
from mpi4py import MPI
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from nystrom import matmul
from communicator import ProcGrid
from matrix import ParMat
    
def test_3d(p1, p2, p3, n1, n2, n3):
    myrank = MPI.COMM_WORLD.Get_rank()
    if myrank == 0:
        print(f"[test_3d] testing {n1}x{n2} with {n2}x{n3} on {p1}x{p2}x{p3} grid")

    grid = ProcGrid(p1, p2, p3)
    A = ParMat(n1, n2, grid, 'A')
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.int32)
    B.generate(dtype=np.int32)
    C = matmul(A,B)

    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()

    flag = np.array_equal( np.matmul(Ag, Bg), Cg)
    if myrank == 0:
        if flag:
            print(f"[test_3d] success")
        else:
            print(f"[test_3d] failure")

def test_2d(p1, p2, p3, n1, n2, n3):
    myrank = MPI.COMM_WORLD.Get_rank()
    if myrank == 0:
        print(f"[test_2d] testing {n1}x{n2} with {n2}x{n3} on {p1}x{p2*p3}x{1} grid")

    grid = ProcGrid(p1, p2*p3, 1)
    A = ParMat(n1, n2, grid, 'A')
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.int32)
    B.generate(dtype=np.int32)
    C = matmul(A,B)

    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()

    flag = np.array_equal( np.matmul(Ag, Bg), Cg)
    if myrank == 0:
        if flag:
            print(f"[test_2d] success")
        else:
            print(f"[test_2d] failure")

    if myrank == 0:
        print(f"[test_2d] testing {n1}x{n2} with {n2}x{n3} on {p1*p2}x{1}x{p3} grid")

    grid = ProcGrid(p1*p2, 1, p3)
    A = ParMat(n1, n2, grid, 'A')
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.int32)
    B.generate(dtype=np.int32)
    C = matmul(A,B)

    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()

    flag = np.array_equal( np.matmul(Ag, Bg), Cg)
    if myrank == 0:
        if flag:
            print(f"[test_2d] success")
        else:
            print(f"[test_2d] failure")

    if myrank == 0:
        print(f"[test_2d] testing {n1}x{n2} with {n2}x{n3} on {1}x{p1*p2}x{p3} grid")

    grid = ProcGrid(1, p1*p2, p3)
    A = ParMat(n1, n2, grid, 'A')
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.int32)
    B.generate(dtype=np.int32)
    C = matmul(A,B)

    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()

    flag = np.array_equal( np.matmul(Ag, Bg), Cg)
    if myrank == 0:
        if flag:
            print(f"[test_2d] success")
        else:
            print(f"[test_2d] failure")

def test_1d(p1, p2, p3, n1, n2, n3):
    myrank = MPI.COMM_WORLD.Get_rank()

    if myrank == 0:
        print(f"[test_1d] testing {n1}x{n2} with {n2}x{n3} on {p1*p2*p3}x{1}x{1} grid")

    grid = ProcGrid(p1*p2*p3, 1, 1)
    A = ParMat(n1, n2, grid, 'A')
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.int32)
    B.generate(dtype=np.int32)
    C = matmul(A,B)

    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()

    flag = np.array_equal( np.matmul(Ag, Bg), Cg)
    if myrank == 0:
        if flag:
            print(f"[test_1d] success")
        else:
            print(f"[test_1d] failure")

    if myrank == 0:
        print(f"[test_1d] testing {n1}x{n2} with {n2}x{n3} on {1}x{p1*p2*p3}x{1} grid")

    grid = ProcGrid(1, p1*p2*p3, 1)
    A = ParMat(n1, n2, grid, 'A')
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.int32)
    B.generate(dtype=np.int32)
    C = matmul(A,B)

    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()

    flag = np.array_equal( np.matmul(Ag, Bg), Cg)
    if myrank == 0:
        if flag:
            print(f"[test_1d] success")
        else:
            print(f"[test_1d] failure")

    if myrank == 0:
        print(f"[test_1d] testing {n1}x{n2} with {n2}x{n3} on {1}x{1}x{p1*p2*p3} grid")

    grid = ProcGrid(1, 1, p1*p2*p3)
    A = ParMat(n1, n2, grid, 'A')
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.int32)
    B.generate(dtype=np.int32)
    C = matmul(A,B)

    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()

    flag = np.array_equal( np.matmul(Ag, Bg), Cg)
    if myrank == 0:
        if flag:
            print(f"[test_1d] success")
        else:
            print(f"[test_1d] failure")

def test_3d_float(p1, p2, p3, n1, n2, n3):
    myrank = MPI.COMM_WORLD.Get_rank()
    if myrank == 0:
        print(f"[test_3d_float] testing {n1}x{n2} with {n2}x{n3} on {p1}x{p2}x{p3} grid")

    grid = ProcGrid(p1, p2, p3)
    A = ParMat(n1, n2, grid, 'A')
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.float32)
    B.generate(dtype=np.float32)
    C = matmul(A,B)

    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()

    flag = np.array_equal( np.matmul(Ag, Bg), Cg)
    if myrank == 0:
        if flag:
            print(f"[test_3d_float] success")
        else:
            print(f"[test_3d_float] failure")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--p1", type=int, help="Number of process grid rows")
    parser.add_argument("-p2", "--p2", type=int, help="Number of process grid columns")
    parser.add_argument("-p3", "--p3", type=int, help="Number of process grid fibers")
    parser.add_argument("-n1", "--n1", type=int, help="Number of rows in A")
    parser.add_argument("-n2", "--n2", type=int, help="Number of columns in A / Number of rows in B")
    parser.add_argument("-n3", "--n3", type=int, help="Number of columns in B")
    args = parser.parse_args()

    p1 = args.p1
    p2 = args.p2
    p3 = args.p3
    n1 = args.n1
    n2 = args.n2
    n3 = args.n3

    myrank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()

    test_3d(p1, p2, p3, n1, n2, n3)
    test_3d_float(p1, p2, p3, n1, n2, n3)
    test_2d(p1, p2, p3, n1, n2, n3)
    test_1d(p1, p2, p3, n1, n2, n3)
