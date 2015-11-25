#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>

#define dim 4097

void kernel(double[dim][dim] F) {
    for (int k = 0; k < 100; k++)
        for (int i = 1; i < dim; i++)
            for (int j = 0; j < dim - 1; j++)
                F[i][j] = F[i-1][j+1] + F[i][j+1];
}

int main() {
    double F[dim][dim];
    // generate random array
    for (int i = 0; i < 4097; i++)
        for (int j = 0; j < 4097; j++)
            F[i][j] = 1.0 + ((double)rand() / (double)RAND_MAX);
    // call kernel
    kernel(F);
}

