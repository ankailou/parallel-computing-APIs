#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define dim 4096

__global__ void kernel(double **A, double** C) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    for (int k = 0; k < dim; k++)
        C[i][j] += A[k][i] * A[k][j];
}

int main() {
    double **A;    // host pointer operand
    double **C;    // host pointer result
    double **d_a;  // device pointer operand
    double **d_c;  // device pointer result

    // thread hierarchy
    int nblocks = 4096;
    int tpb = 4096;

    // allocate memory
    size_t memSize;
    A = new double*[dim];
    C = new double*[dim]
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
}

