#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define dim 4097

void kernel(double **F) {
    for (int k = 0; k < 100; k++)
        for (int i = 1; i < dim; i++)
            for (int j = 0; j < dim - 1; j++)
                F[i][j] = F[i-1][j+1] + F[i][j+1];
}

int main() {
    double **F;
    F = new double*[dim];
    // generate random array
    for (int i = 0; i < dim; i++) {
        F[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            F[i][j] = 1.0 + ((double)rand() / (double)RAND_MAX);
        }
    }
    // call kernel
    kernel(F);
}

