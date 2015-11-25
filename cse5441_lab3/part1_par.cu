#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define dim 4097

__global__ void kernel(double **F) {
    for (int k = 0; k < 100; k++)
        for (int i = 1; i < dim; i++)
            for (int j = 0; j < dim - 1; j++)
                F[i][j] = F[i-1][j+1] + F[i][j+1];

    int i = blockIdx.x + 1;
    int j = blockIdx.y;
    F[i][j] = F[i-1][j+1] + F[i][j+1];
}

int main() {
    double **F;    // host pointer
    double **d_a;  // device pointer

    // thread hierarchy
    int nblocks = 4096;
    int tpb = 100;

    // allocate memory
    size_t memSize;
    F = new double*[dim];
    memSize = dim * dim * sizeof(double);
    cudaMalloc((void***)&d_a, memSize);

    // generate random array & copy
    for (int i = 0; i < dim; i++) {
        F[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            F[i][j] = 1.0 + ((double)rand() / (double)RAND_MAX);
        }
    }
    cudaMemcpy(d_a,F,memSize,cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimGrid(tpb);
    dim3 dimBlock(nblocks,nblocks);
    kernel<<dimGrid,dimBlock>>(d_a);
}

