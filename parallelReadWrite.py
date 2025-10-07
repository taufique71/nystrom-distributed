from utils import *

myrank = MPI.COMM_WORLD.Get_rank()
n=20
nRowLocal=10
nColLocal=10
if myrank==0:
    localRowStart=0
    localColStart=0
if myrank==1:
    localRowStart=0
    localColStart=10
if myrank==2:
    localRowStart=10
    localColStart=0
if myrank==3:
    localRowStart=10
    localColStart=10
dataPath="/deac/csc/ballardGrp/rahmm224/new_nystrom/data/cifar10-linear.bin"
# localMat = getCIFAR10GramMatrix(path = dataPath, world = MPI.COMM_WORLD, n = n, nRowLocal = nRowLocal, nColLocal = nColLocal, localRowStart = localRowStart, localColStart = localColStart)
# writeDataPath="/deac/csc/ballardGrp/rahmm224/new_nystrom/data/written.bin"
# writeCIFAR10GramMatrix(path=writeDataPath, world = MPI.COMM_WORLD, n = n, nRowLocal = nRowLocal, nColLocal = nColLocal, localRowStart = localRowStart, localColStart = localColStart, localdata=localMat)
# print(f"Rank {myrank} data {localMat[:3,:3]}")

file1="/deac/csc/ballardGrp/rahmm224/new_nystrom/data/sample-linear.bin"
file2="/deac/csc/ballardGrp/rahmm224/new_nystrom/data/sample-written.bin"
if myrank==0:
    compare_bin_matrices(file1, file2, shape=(4, 4), dtype=np.float64)
    