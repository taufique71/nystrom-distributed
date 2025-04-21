import os
import sys
import argparse
from mpi4py import MPI
import numpy as np
from communicator import ProcGrid
from matrix import ParMat
from utils import *

# def allGather(M, world):
    # # Gather local copies of the distributed matrix M
    # # Concatenates the copies along the column and return the concatenated copy
    # # All matrices are stored in column major order in Fortran
    # # Assumes same number of matrix rows in all processes
    # npDtype = M.localMat.dtype
    # mpiDtype = npDtypeToMpiDtype(npDtype)

    # rankInWorld = world.Get_rank()

    # nColToSend = np.array(M.nColLocal, dtype=npDtype)
    # nColToRecv = np.zeros(world.Get_size(), dtype=npDtype)
    # world.Allgather([nColToSend, mpiDtype], [nColToRecv, mpiDtype])
    # nValToRecv = M.nRowLocal * nColToRecv # Multiplying all entries with number of local rows would give the number of entries
    # recvDispls = np.zeros(world.Get_size() + 1, dtype=npDtype) # One extra entry because our prefix sum would start from 0 as initial entry
    # np.cumsum(nValToRecv, out=recvDispls[1:])
    # targetM = np.zeros(recvDispls[-1], dtype=npDtype).reshape((M.nRowLocal, np.sum(nColToRecv)), order='F')

    # world.Allgatherv([M.localMat, M.nColLocal*M.nRowLocal, mpiDtype], [targetM, nValToRecv, recvDispls[:-1], mpiDtype])
    # return targetM



# def reduceScatter(M, world):
    # # Modifies the contents of M matrix
    # # Split the local matrix, redistributes the chunks of M along the world and performs reduction
    # npDtype = M.localMat.dtype
    # mpiDtype = npDtypeToMpiDtype(npDtype)

    # rankInWorld = world.Get_rank()

    # ## Gather number of columns to send to each process
    # nColToSend = np.array(M.nColLocal, dtype=npDtype)
    # nColToRecv = np.zeros(world.Get_size(), dtype=npDtype) 
    # world.Allgather([nColToSend, mpiDtype], [nColToRecv, mpiDtype])
    # nValToRecv = M.nRowLocal * nColToRecv
    # splitLocs = np.zeros(world.Get_size() + 1, dtype=npDtype)
    # np.cumsum(nColToRecv, out=splitLocs[1:])
    # targetM = np.zeros((M.nRowLocal, nColToRecv[rankInWorld]), dtype=npDtype, order='F')

    # # Supports vector version even though there is no `v` suffix. Similar case in C API
    # # https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html#mpi4py.MPI.Comm.Reduce_scatter
    # world.Reduce_scatter([M.localMat, mpiDtype], [targetM, nValToRecv[rankInWorld], mpiDtype], nValToRecv, op=MPI.SUM)

    # M.localMat = targetM

def matmul(A, B):
    assert(A.nColGlobal == B.nRowGlobal)

    npDtype = A.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)
    
    # Gather local matrices of A along the grid fibers
    # targetA = allGather(A, A.grid.fibWorld)
    targetA = allGatherAndConcat(A.localMat, A.grid.fibWorld, concat="col")
    # if (A.grid.myrank == 4):
        # print(targetA)

    # Gather local matrices of B along the grid columns
    # targetB = allGather(B, B.grid.colWorld)
    targetB = allGatherAndConcat(B.localMat, B.grid.colWorld, concat="col")
    # if (B.grid.myrank == 4):
        # print(targetB)
    
    # Create distributed C matrix with appropriate dimension that has no content
    C = ParMat(n1, n3, A.grid, 'C') # Use the same process grid as A. Grid does not change for A, B or C. Only the face of the grid change which is specific to the matrix

    # Multiply gathered A with gathered B
    C.localMat = np.matmul(targetA, targetB, order='F')
    # print(C.grid.myrank, C.localMat)
    # print(C.localMat.flags)

    # MPI.COMM_WORLD.Barrier()
    # sys.stdout.flush();
    # sys.stderr.flush();
    # if (C.grid.myrank == 0):
        # print("---")
    # MPI.COMM_WORLD.Barrier()
    
    # Distribute the C contribution from multiplying gathered A and B
    # Which are local matrices of C along the grid rows 
    # reduceScatter(C, C.grid.rowWorld)
    C.localMat = splitAndReduceScatter(C.localMat, C.grid.rowWorld, split='col')
    
    # print(C.grid.myrank, C.localMat)
    
    return C

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numrows", type=int, help="Number of rows and columns of the matrix", default=100)
    parser.add_argument("-r", "--rank", type=int, help="Value of rank for low rank approximation", default=20)
    parser.add_argument("-p1", "--p1", type=int, help="Number of process grid rows for first matmul")
    parser.add_argument("-p2", "--p2", type=int, help="Number of process grid cols for first matmul")
    parser.add_argument("-p3", "--p3", type=int, help="Number of process grid cols for first matmul")
    parser.add_argument("-n1", "--n1", type=int, help="Number of rows in A")
    parser.add_argument("-n2", "--n2", type=int, help="Number of columns in A / Number of rows in B")
    parser.add_argument("-n3", "--n3", type=int, help="Number of columns in B")
    args = parser.parse_args()
    # print(args)

    n = args.numrows
    r = args.rank
    p1 = args.p1
    p2 = args.p2
    p3 = args.p3
    n1 = args.n1
    n2 = args.n2
    n3 = args.n3

    myrank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    
    p = nprocs
    assert(p == args.p1 * args.p2 * args.p3)
    grid = ProcGrid(args.p1, args.p2, args.p3)

    A = ParMat(n1, n2, grid, 'A')
    MPI.COMM_WORLD.Barrier()
    B = ParMat(n2, n3, grid, 'B')

    A.generate(dtype=np.int32)
    B.generate(dtype=np.int32)
    C = matmul(A,B)
    
    ## Check correctness
    Ag = A.allGather()
    Bg = B.allGather()
    Cg = C.allGather()
    if A.grid.myrank == 0:
        print(Ag)
        print("x")
    if B.grid.myrank == 0:
        print(Bg)
        print("=")
    if C.grid.myrank == 0:
        print(Cg)
        print("---")

    if np.array_equal( np.matmul(Ag, Bg), Cg):
        if A.grid.myrank == 0:
            print("Correct")
    else:
        if A.grid.myrank == 0:
            print("Incorrect")
