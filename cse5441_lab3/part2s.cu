#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define dim 4096

void kernel(double **A, double **C) {
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            for (int k = 0; k < dim; k++)
               C[i][j] += A[k][i] * A[k][j];
}

int main() {
    double **A;
    double **C;
    A = new double*[dim];
    C = new double*[dim];
    // generate random array
    for (int i = 0; i < dim; i++) {
        A[i] = new double[dim];
        C[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            C[i][j] = 0.0;
            A[i][j] = 1.0 + ((double)rand() / (double)RAND_MAX);
        }
    }
    // call kernel
    kernel(A,C);
}

