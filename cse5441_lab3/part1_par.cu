#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define dim 4097

__global__ void kernel(double *F) {
    int idx = blockIdx.x * 4 + 1;
    int jdx = threadIdx.x * 4;
    for (int k = 0; k < 100; k++) {
        for (int i = idx; i < idx + 4; i++ )
            for (int j = jdx; j < jdx + 4; j++)
                F[(i*1024) + j] = F[1024*(i-1) + j+1] + F[(1024*i) + j+1];
    }
}

int main() {
    double *F;    // host pointer
    double *d_a;  // device pointer

    // thread hierarchy
    int nblocks = 1024;
    int tpb = 1024;

    // allocate memory
    size_t memSize;
    F = new double[dim * dim];
    memSize = dim * dim * sizeof(double);
    cudaMalloc((void**)&d_a, memSize);

    // generate random array & copy
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            F[(i*1024) + j] = 1.0 + ((double)rand() / (double)RAND_MAX);
        }
    }
    cudaMemcpy(d_a,F,memSize,cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimGrid(nblocks);
    dim3 dimBlock(tpb);
    kernel<<<dimGrid,dimBlock>>>(d_a);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    else
        printf("Success: terminating...\n");
}

