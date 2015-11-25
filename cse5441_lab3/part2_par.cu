#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define dim 4096

__global__ void kernel(double **A, double** C) {
    int idx = threadIdx.x * 4;
    int jdx = blockIdx.x * 4;
    for (int k = 0; k < dim; k++) {
        for (int i = idx; i < idx + 4; i++)
            for (int j = jdx; j < jdx + 2; j++)
                C[i][j] += A[k][i] * A[k][j];
    }
}

int main() {
    double **A;    // host pointer operand
    double **C;    // host pointer result
    double **d_a;  // device pointer operand
    double **d_c;  // device pointer result

    // thread hierarchy
    int nblocks = 1024;
    int tpb = 1024;

    // allocate memory
    size_t memSize;
    A = new double*[dim];
    C = new double*[dim];
    memSize = dim * dim * sizeof(double);
    cudaMalloc((void***)&d_a, memSize);
    cudaMalloc((void***)&d_c, memSize);

    // generate random array & copy
    for (int i = 0; i < dim; i++) {
        A[i] = new double[dim];
        C[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            C[i][j] = 0.0;
            A[i][j] = 1.0 + ((double)rand() / (double)RAND_MAX);
        }
    }
    cudaMemcpy(d_a,A,memSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,C,memSize,cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimGrid(tpb);
    dim3 dimBlock(nblocks);
    kernel<<<dimGrid,dimBlock>>>(d_a,d_c);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    else
        printf("Success: terminating!\n");
}

