from mpi4py import MPI
import numpy as np
from torchvision.datasets import CIFAR10


def npDtypeToMpiDtype(npDtype):
    # Translate numpy datatypes to mpi4py datatypes
    # Assumes same datatype is being used for both A, B and C
    # Assumes only 32 bit or 64 bit integers or floating point numbers would be used
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

def allGatherAndConcat(mat, world, concat="col"):
    # Gather all matrices from the world
    # Concatenates the copies along the column and return the concatenated copy
    # Assumes all matrices to be stored in the column major order
    # For column concatenation, assumes all matrices to have the same number of rows
    # For row concatenation, assumes all matrices to have the same number of columns
    npDtype = mat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)

    rankInWorld = world.Get_rank()

    agTime = 0.0
    agvTime = 0.0
    totTime = 0.0

    if concat == 'col':
        t0 = MPI.Wtime()

        nColToSend = np.array(mat.shape[1], dtype=np.int32)
        nColToRecv = np.zeros(world.Get_size(), dtype=np.int32)

        t1 = MPI.Wtime()
        world.Allgather([nColToSend, MPI.INT], [nColToRecv, MPI.INT])
        t2 = MPI.Wtime()
        agTime = agTime + (t2-t1)

        nValToRecv = mat.shape[0] * nColToRecv # Multiplying all entries with number of local rows would give the number of entries
        recvDispls = np.zeros(world.Get_size() + 1, dtype=np.int32) # One extra entry because our prefix sum would start from 0 as initial entry
        np.cumsum(nValToRecv, out=recvDispls[1:])
        targetMat = np.zeros(recvDispls[-1], dtype=npDtype).reshape((mat.shape[0], np.sum(nColToRecv)), order='F')
        
        t3 = MPI.Wtime()
        world.Allgatherv([mat, mat.shape[0]*mat.shape[1], mpiDtype], [targetMat, nValToRecv, recvDispls[:-1], mpiDtype])
        t4 = MPI.Wtime()
        agvTime = agvTime + (t4-t3)

        totTime = totTime + (t4-t0)

        return targetMat, agTime, agvTime, totTime

    elif concat == 'row':
        t0 = MPI.Wtime()

        nRowToSend = np.array(mat.shape[0], dtype=np.int32)
        nRowToRecv = np.zeros(world.Get_size(), dtype=np.int32)

        t1 = MPI.Wtime()
        world.Allgather([nRowToSend, MPI.INT], [nRowToRecv, MPI.INT])
        t2 = MPI.Wtime()
        agTime = agTime + (t2-t1)

        nValToRecv = mat.shape[1] * nRowToRecv # Multiplying all entries with number of local cols would give the number of entries
        recvDispls = np.zeros(world.Get_size() + 1, dtype=np.int32) # One extra entry because our prefix sum would start from 0 as initial entry
        np.cumsum(nValToRecv, out=recvDispls[1:])
        recvBuff = np.zeros(recvDispls[-1], dtype=npDtype)

        t3 = MPI.Wtime()
        world.Allgatherv([mat, mat.shape[0]*mat.shape[1], mpiDtype], [recvBuff, nValToRecv, recvDispls[:-1], mpiDtype])
        t4 = MPI.Wtime()
        agvTime = agvTime + (t4-t3)

        recvMats = []
        for r in range(world.Get_size()):
            recvMats.append(np.array(recvBuff[recvDispls[r]:recvDispls[r+1]]).reshape(nRowToRecv[r], mat.shape[1], order='F'))
        targetMat = np.concatenate(recvMats, axis=0).reshape( (np.sum(nRowToRecv), mat.shape[1]), order='F')

        t5 = MPI.Wtime()
        totTime = totTime + (t5-t0)

        return targetMat, agTime, agvTime, totTime

def splitAndReduceScatter(mat, world, split="col"):
    # Split the local matrix into equal chunks, redistributes the chunks of M along the world and performs reduction
    npDtype = mat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)

    rankInWorld = world.Get_rank()

    rsTime = 0.0
    totTime = 0.0
    
    if split == 'col':
        
        t0 = MPI.Wtime()
        nColToRecv = np.zeros(world.Get_size(), dtype=np.int32) 
        for r in range(world.Get_size()):
            if r < (world.Get_size() - 1) :
                nColToRecv[r] = int(mat.shape[1] / world.Get_size()) # If rankInWorld is not the last rank in the world
            elif r == (world.Get_size() - 1) :
                nColToRecv[r] = mat.shape[1] - r * int( mat.shape[1] / world.Get_size() )

        nValToRecv = mat.shape[0] * nColToRecv
        splitLocs = np.zeros(world.Get_size() + 1, dtype=npDtype)
        np.cumsum(nColToRecv, out=splitLocs[1:])
        targetMat = np.zeros((mat.shape[0], nColToRecv[rankInWorld]), dtype=npDtype, order='F')

        t1 = MPI.Wtime()
        # Supports vector version even though there is no `v` suffix. Similar case in C API
        # https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html#mpi4py.MPI.Comm.Reduce_scatter
        world.Reduce_scatter([mat, mpiDtype], [targetMat, nValToRecv[rankInWorld], mpiDtype], nValToRecv, op=MPI.SUM)
        t2 = MPI.Wtime()
        rsTime = t2-t1
        totTime = t2-t0

        return targetMat, rsTime, totTime
    
    elif split == 'row':
        t0 = MPI.Wtime()
        nRowToRecv = np.zeros(world.Get_size(), dtype=np.int32) 
        for r in range(world.Get_size()):
            if r < (world.Get_size() - 1) :
                nRowToRecv[r] = int(mat.shape[0] / world.Get_size()) # If rankInWorld is not the last rank in the world
            elif r == (world.Get_size() - 1) :
                nRowToRecv[r] = mat.shape[0] - r * int( mat.shape[0] / world.Get_size() )

        nValToRecv =  nRowToRecv * mat.shape[1]
        splitLocs = np.zeros(world.Get_size() + 1, dtype=npDtype)
        np.cumsum(nRowToRecv, out=splitLocs[1:])

        targetMat = np.zeros((nRowToRecv[rankInWorld], mat.shape[1]), dtype=npDtype, order='F')
        revMat = np.zeros((nRowToRecv[rankInWorld] * mat.shape[1]), dtype=npDtype, order='F')
        
        t1 = MPI.Wtime()

        world.Reduce_scatter([mat, mpiDtype], [revMat, nValToRecv[rankInWorld], mpiDtype], nValToRecv, op=MPI.SUM)

        t2 = MPI.Wtime()
        for r in range(nRowToRecv[rankInWorld]):
            targetMat[r, :] = revMat[r * mat.shape[1] : (r + 1) * mat.shape[1]]
        
        rsTime = t2-t1
        totTime = t2-t0
        
        return targetMat, rsTime, totTime

def Reduce_ScallaPackScatter(mat, world, colSplitsY2d, split="col"):
    # Split the local matrix into equal chunks, redistributes the chunks of M along the world and performs reduction
    npDtype = mat.dtype
    mpiDtype = npDtypeToMpiDtype(npDtype)

    rankInWorld = world.Get_rank()

    rsTime = 0.0
    totTime = 0.0
    
    if split == 'col':
        
        t0 = MPI.Wtime()
        nColToRecv = np.array(colSplitsY2d) 
        
        nValToRecv = mat.shape[0] * nColToRecv
        splitLocs = np.zeros(world.Get_size() + 1, dtype=npDtype)
        np.cumsum(nColToRecv, out=splitLocs[1:])
        targetMat = np.zeros((mat.shape[0], nColToRecv[rankInWorld]), dtype=npDtype, order='F')

        t1 = MPI.Wtime()
        # Supports vector version even though there is no `v` suffix. Similar case in C API
        # https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html#mpi4py.MPI.Comm.Reduce_scatter
        world.Reduce_scatter([mat, mpiDtype], [targetMat, nValToRecv[rankInWorld], mpiDtype], nValToRecv, op=MPI.SUM)
        t2 = MPI.Wtime()
        rsTime = t2-t1
        totTime = t2-t0

        return targetMat, rsTime, totTime   

def ScaLAPACK(n, p):
        ceil = int(np.ceil(n / p))
        dist = np.full(p, ceil, dtype=int)
        dist[-1] = n - (p - 1) * ceil
        return dist
def hierarchical_dist(n, p1, p2):
    dist = ScaLAPACK(n, p1)
    double_scalapack = np.concatenate([ScaLAPACK(r, p2) for r in dist])

    return double_scalapack

def getCIFAR10GramMatrix(path, world, n, nRowLocal, nColLocal, localRowStart, localColStart):

    lsizes = [nRowLocal, nColLocal]
    gsizes = [n, n]
    starts = [localRowStart, localColStart]
    # Open file collectively
    fh = MPI.File.Open(world, path, MPI.MODE_RDONLY)

    base_type = MPI.DOUBLE  

    # Define subarray view
    view = base_type.Create_subarray(
        sizes=gsizes,        # global tensor shape
        subsizes=lsizes,     # local block shape
        starts=starts,       # starting indices of local block
        # order=MPI.ORDER_FORTRAN  # column-major (Fortran order)
    )
    view.Commit()

    disp = 0  # bite offset to skip into the file before MPI applies the view
    fh.Set_view(disp, filetype=view)

    # Allocate buffer for local block
    buf = np.empty(np.prod(lsizes), dtype=np.float64)

    # Collective read: each process gets its block
    fh.Read_all(buf)

    fh.Close()
    view.Free()
    localdata = np.empty(lsizes, dtype=np.float64, order='F')
    # Copy data from buffer to local array
    for i in range(lsizes[0]):
        localdata[i, :] = buf[i * lsizes[1]:(i + 1) * lsizes[1]]

    return localdata

def writeCIFAR10GramMatrix(path, world, n, nRowLocal, nColLocal, localRowStart, localColStart, localdata):

    lsizes = [nRowLocal, nColLocal]     # local block size
    gsizes = [n, n]                     # global matrix size
    starts = [localRowStart, localColStart]  # starting indices for this process

    # Open file collectively for writing (create if doesn't exist, overwrite)
    fh = MPI.File.Open(world, path, MPI.MODE_WRONLY | MPI.MODE_CREATE)

    base_type = MPI.DOUBLE

    # Define subarray view for this process
    view = base_type.Create_subarray(
        sizes=gsizes,
        subsizes=lsizes,
        starts=starts,
        # order=MPI.ORDER_FORTRAN 
    )
    view.Commit()

    disp = 0  # byte displacement at file start
    fh.Set_view(disp, filetype=view)

    # Ensure the buffer is contiguous
    buf = np.ascontiguousarray(localdata, dtype=np.float64)

    # Collective write: all processes write their sub-blocks in parallel
    fh.Write_all(buf)

    # Clean up
    fh.Close()
    view.Free()


def compare_bin_matrices(file1, file2, shape=(20, 20), dtype=np.float64):
    # Read both binary files
    a = np.fromfile(file1, dtype=dtype).reshape(shape)
    b = np.fromfile(file2, dtype=dtype).reshape(shape)

    # Check exact or near-equality
    if np.array_equal(a, b):
        print("Matrices are identical.")
    else:
        print("Matrices differ.")
    # print("File content p0", a[:3,:3])
    # print("File content p1", a[:3,10:13])
    # print("File content p2", a[10:13,0:3])
    # print("File content p3", a[10:13,10:13])


