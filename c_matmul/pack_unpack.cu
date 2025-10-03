#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

__global__ void __1d_redist_1d_unpack__ (
        double* odata, 
        double* idata, 
        int nRow,
        int nCol,
        int* recvDispls,
        int* nRowToRecvPrefixSum
        ){
    int iidx = recvDispls[blockIdx.z] +
               (nRowToRecvPrefixSum[blockIdx.z+1] - nRowToRecvPrefixSum[blockIdx.z]) * blockIdx.x * blockDim.x +
               blockIdx.y * blockDim.y +
               (nRowToRecvPrefixSum[blockIdx.z+1] - nRowToRecvPrefixSum[blockIdx.z]) * threadIdx.y +
               threadIdx.x;
    int oidx = nRow * blockIdx.x * blockDim.x +
               nRowToRecvPrefixSum[blockIdx.z] +
               blockIdx.y * blockDim.y +
               nRow * threadIdx.y +
               threadIdx.x;

    int r = blockIdx.y * blockDim.y + threadIdx.x;
    int c = blockIdx.x * blockDim.x + threadIdx.y;
    int max_r = (nRowToRecvPrefixSum[blockIdx.z+1] - nRowToRecvPrefixSum[blockIdx.z]);
    int max_c = nCol;

    if (r < max_r && c < max_c) odata[oidx] = idata[iidx];
}

__global__ void __2d_noredist_1d_pack__ (
        double* odata,
        double* idata,
        int nRow,
        int nCol,
        int* sendDispls,
        int* nRowToSendPrefixSum
        ){
    int iidx = nRow * blockIdx.x * blockDim.x +
               nRowToSendPrefixSum[blockIdx.z] +
               blockIdx.y * blockDim.y +
               nRow * threadIdx.y +
               threadIdx.x;
    int oidx = sendDispls[blockIdx.z] +
               (nRowToSendPrefixSum[blockIdx.z+1] - nRowToSendPrefixSum[blockIdx.z]) * blockIdx.x * blockDim.x +
               blockIdx.y * blockDim.y +
               (nRowToSendPrefixSum[blockIdx.z+1] - nRowToSendPrefixSum[blockIdx.z]) * threadIdx.y +
               threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.x;
    int c = blockIdx.x * blockDim.x + threadIdx.y;
    int max_r = (nRowToSendPrefixSum[blockIdx.z+1] - nRowToSendPrefixSum[blockIdx.z]);
    int max_c = nCol;

    if (r < max_r && c < max_c) odata[oidx] = idata[iidx];
               
}

extern void nystrom_1d_redist_1d_unpack(
        double* odata, 
        double* idata, 
        int nRow,
        int nCol,
        int* recvDispls,
        int* nRowToRecvPrefixSum,
        int gridDimX,
        int gridDimY,
        int gridDimZ,
        int blockDimX,
        int blockDimY
        ){
    
    dim3 gridSize = dim3(gridDimX, gridDimY, gridDimZ);
    dim3 blockSize = dim3(blockDimX, blockDimY, 1);
    __1d_redist_1d_unpack__ <<< gridSize, blockSize >>>(
            odata,
            idata,
            nRow,
            nCol,
            recvDispls,
            nRowToRecvPrefixSum
    );
}

extern void nystrom_2d_noredist_1d_pack(
        double* odata,
        double* idata,
        int nRow,
        int nCol,
        int* sendDispls,
        int* nRowToSendPrefixSum,
        int gridDimX,
        int gridDimY,
        int gridDimZ,
        int blockDimX,
        int blockDimY
        ){
    
    dim3 gridSize = dim3(gridDimX, gridDimY, gridDimZ);
    dim3 blockSize = dim3(blockDimX, blockDimY, 1);
    __2d_noredist_1d_pack__ <<< gridSize, blockSize >>>(
            odata,
            idata,
            nRow,
            nCol,
            sendDispls,
            nRowToSendPrefixSum
    );
}
