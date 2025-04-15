import os
import argparse
from mpi4py import MPI
import numpy as np
from communicator import ProcGrid
from matrix import ParMat

def npDtypeToMpiDtype(npType):
    # Translate numpy datatypes to mpi4py datatypes
    # Assumes same datatype is being used for both A, B and C
    # Assumes only 32 bit or 64 bit integers or floating point numbers would be used
    npDtype = A.localMat.dtype
    mpiDtype = None
    if npDtype == np.int32:
        mpiDtype = MPI.INT
    elif npDtype == np.int64:
        mpiDtype = MPI.LONG
    elif npDtype == np.float32:
        mpiDtype = MPI.FLOAT
    elif npDtype == np.float64:
        mpiDtype = MPI.DOUBLE
    return mpiDtype

def gather(M, world):
    # Gather local copies of the distributed matrix M
    # Concatenates the copies along the column and return the concatenated copy
    # All matrices are stored in column major order in Fortran
    # Assumes same number of matrix rows in all processes
    npDtype = M.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)

    nColToSend = np.array(M.nColLocal, dtype=npDtype)
    nColToRecv = np.zeros(world.Get_size(), dtype=npDtype)
    world.Allgather([nColToSend, mpiDtype], [nColToRecv, mpiDtype])
    nValToRecv = M.nRowLocal * nColToRecv # Multiplying all entries with number of local rows would give the number of entries
    recvDispls = np.zeros(world.Get_size() + 1, dtype=npDtype) # One extra entry because our prefix sum would start from 0 as initial entry
    np.cumulative_sum(nValToRecv, out=recvDispls, include_initial=True)
    recvVals = np.zeros(recvDispls[-1], dtype=npDtype)
    world.Allgatherv([M.localMat, M.nColLocal*M.nRowLocal, mpiDtype], [recvVals, nValToRecv, recvDispls[:-1], mpiDtype])
    recvM = []
    for i in range(world.Get_size()):
        recvM.append(np.array(recvVals[ recvDispls[i] : recvDispls[i+1]], order='F').reshape((M.nRowLocal, nColToRecv[i]), order='F') )
    targetM = np.concatenate(recvM, axis=1) # axis=1 for column concatenation
    # if (M.grid.myrank == 4):
        # print(targetM)
    # return targetM

def reduceScatter(M, world):
    # Split the local matrix
    # Gather local copies of the distributed matrix M
    # Concatenates the copies along the column and return the concatenated copy
    # All matrices are stored in column major order in Fortran
    # Assumes same number of matrix rows in all processes
    npDtype = M.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)

    ## Gather number of columns to send to each process
    nColToSend = np.array(M.nColLocal, dtype=npDtype)
    nColToRecv = np.zeros(world.Get_size(), dtype=npDtype) 
    world.Allgather([nColToSend, mpiDtype], [nColToRecv, mpiDtype])
    splitLocs = np.zeros(world.Get_size() + 1, dtype=npDtype)
    np.cumulative_sum(nColToRecv, out=splitLocs, include_initial=True)
    chunksM = np.split(M.localMat, splitLocs[1:-1], axis=1)
    # C.localMat = 
    for i in range(C.grid.rowWorld.Get_size()):
        sendBuff = np.array(chunksC[i], order='F')
        recvBuff = np.array(chunksC[i], order='F')
        world.Reduce([sendBuff, M.nRowLocal * nColToRecv[i], mpiDtype], [recvBuff, M.nRowLocal * nColToRecv[i], mpiDtype], op=MPI.SUM, root=i)
        if i == world.Get_rank():
            M.localMat = recvbuff
    

def matmul(A, B):
    assert(A.nColGlobal == B.nRowGlobal)

    npDtype = A.localMat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)
    
    # Gather local matrices of A along the grid fibers
    targetA = gather(A, A.grid.fibWorld)

    # Gather local matrices of B along the grid columns
    targetB = gather(B, B.grid.colWorld)
    
    # Create distributed C matrix with appropriate dimension that has no content
    C = ParMat(n1, n3, A.grid, 'C') # Use the same process grid as A. Grid does not change for A, B or C. Only the face of the grid change

    # Multiply gathered A with gathered B
    C.localMat = np.matmul(targetA, targetB)
    if (C.grid.myrank == 0):
        print(C.localMat)
    
    # Distribute the result from multiplying gathered A and B as the content of C
    # Distribute local matrices of C along the grid rows 
    reduceScatter(C, C.grid.rowWorld)
    
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
    matmul(A,B)
