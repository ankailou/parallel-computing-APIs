#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define dim 1024
#define nblocks 1
#define tpb 32

__global__ void kernel(int *F) {
    // compute block/thread offset
    int block_offset = (dim * dim * blockIdx.x) / (nblocks * 2);
    int thread_offset = (dim * dim * threadIdx.x) / (nblocks * tpb * 2);
    int num_flops = (dim * dim) / (nblocks * tpb * 2);
    int idx = block_offset + thread_offset;
    int end = (dim * dim) - 1;
    for ( int i = idx; i < idx + num_flops; i++) {
        int tmp = F[i];
        F[i] = F[end - i];
        F[end - i] = tmp;
    }
}

int main() {
    int *F;    // host pointer
    int *F_v;  // host verification
    int *d_a;  // device pointer

    // allocate memory
    size_t memSize;
    F = new int[dim*dim];
    F_v = new int[dim*dim];
    memSize = dim * dim * sizeof(int);
    cudaMalloc((void**)&d_a, memSize);

    // generate random array & copy
    for (int i = 0; i < dim * dim; i++) {
        F[i] = 1 + (int)(((double)rand() / (double)RAND_MAX) * 999);
    }
    memcpy(F_v,F,memSize);
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
        printf("Success: verifying...\n");

    // get memory back
    cudaMemcpy(F, d_a, memSize, cudaMemcpyDeviceToHost);

    // verify
    int checkSize = (dim * dim) / 2;
    int correct = 1;
    for ( int k = 0; k < checkSize; k++) {
        if (F[k] != F_v[(dim * dim) - 1 - k]) {
            printf("Error @ Index %d!\n",k);
            correct = 0;
            break;
        }
    }
    if (correct) printf("Reversal successful!");
}

