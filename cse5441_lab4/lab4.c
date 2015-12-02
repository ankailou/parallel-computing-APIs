#include <mpi.h>
#include <stdio.h>
#include <time.h>

#define MSG_IDX 5
#define ITERATIONS 1000000

int main(int argc, char *argv[]){
    double *A, *B;
    int rank, size, i, k;
    MPI_Status status;
    MPI_Init( &argc, &argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank); // process 0 sending to process 1
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    if (size != 2) {
        fprintf(stderr, "World size must be two for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int msg[MSG_IDX] = { 32, 256, 512, 1024, 2048 };
    MPI_Barrier(MPI_COMM_WORLD);
    for (k = 0; k < MSG_IDX; k++) {
        int msgSize = msg[k];
        A = (double *)malloc( msgSize * sizeof(double) );
        B = (double *)malloc( msgSize * sizeof(double) );
        for (i = 0; i < msgSize; i++) {
            A[i] = (double)rand();
            B[i] = (double)rand();
        }

        printf("Trial %d, Process %d: Message Size = %d dpfp...\n", k, rank, msg[k]);
        // start timer
        clock_t start = clock(), diff;
        for (i = 0; i < ITERATIONS; i++) {
            if ( rank == 0 ) {
                MPI_Send( &A[0], msgSize, MPI_DOUBLE_PRECISION, 1, 0, MPI_COMM_WORLD);
                MPI_Recv( &B[0], msgSize, MPI_DOUBLE_PRECISION, 1, 0, MPI_COMM_WORLD, &status );
                MPI_Send( &B[0], msgSize, MPI_DOUBLE_PRECISION, 1, 0, MPI_COMM_WORLD);
                MPI_Recv( &A[0], msgSize, MPI_DOUBLE_PRECISION, 1, 0, MPI_COMM_WORLD, &status );
            } else if ( rank == 1 ) {
                MPI_Recv( &A[0], msgSize, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Send( &B[0], msgSize, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD );
                MPI_Recv( &B[0], msgSize, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Send( &A[0], msgSize, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD );
            }
        }
        diff = clock() - start;
        long tm = diff / ITERATIONS;
        long bandwidth = (long)(sizeof(double) * msg[k]) / (long)time;
        printf("Trial %d, Process %d: time = %lu, bandwidth = %lu", k, rank, tm, bandwidth);
        // compute runtime + bandwidth
        free(A);
        free(B);
    }
    MPI_Finalize();
    return 0;
}
