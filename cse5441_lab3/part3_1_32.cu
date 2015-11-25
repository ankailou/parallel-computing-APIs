#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define dim 1024

__global__ void kernel(int *F) {
    int idx = threadIdx.x * 16384;
    int end = (dim * dim) - 1;
    for ( int i = idx; i < idx + 16384 - 1; i++) {
        int tmp = F[i];
        F[i] = F[end - i];
        F[end - i] = tmp;
    }
}

int main() {
    int *F;    // host pointer
    int *F_v;  // host verification
    int *d_a;  // device pointer

    // thread hierarchy
    int nblocks = 1;
    int tpb = 32;

    // allocate memory
    size_t memSize;
    F = new int[dim*dim];
    F_v = new int[dim*dim];
    memSize = dim * dim * sizeof(int);
    cudaMalloc((void**)&d_a, memSize);

    // generate random array & copy
    for (int i = 0; i < dim * dim; i++) {
        F[i] = 1 + (int)(((double)rand() / (double)RAND_MAX) * 999);
        F_v[i] = F[i];
        printf("%d ",F[i]);
    }
    printf("\n\n\n")
    cudaMemcpy(d_a,F,memSize,cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimGrid(tpb);
    dim3 dimBlock(nblocks,nblocks);
    kernel<<<dimGrid,dimBlock>>>(d_a);

    // get memory back
    cudaMemcpy(F, d_a, memSize, cudaMemcpyDeviceToHost);

    for (int t = 0; t < dim * dim; t++)
        printf("%d ",F[i]);

    // verify
    int end = (dim * dim) - 1;
    int checkSize = (dim * dim) / 2;
    int correct = 1;
    for ( int k = 0; k < checkSize; k++) {
        if (F[k] != F_v[end - k]) {
            printf("Error @ Index %d!\n",k);
            correct = 0;
            break;
        }
    }
    if (correct) printf("Reversal successful!");
}

