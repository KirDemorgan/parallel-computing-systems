#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>

void array_operations_sequential(double* a, double* b, double* sum, double* diff, double* prod, double* quot, int N) {
    for (int i = 0; i < N; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        if (b[i] != 0.0) {
            quot[i] = a[i] / b[i];
        } else {
            quot[i] = 0.0;
        }
    }
}

void array_operations_parallel(double* a, double* b, double* sum, double* diff, double* prod, double* quot, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        if (b[i] != 0.0) {
           quot[i] = a[i] / b[i];
        } else {
           quot[i] = 0.0;
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 10000000;
    int opt;

    while ((opt = getopt(argc, argv, "n:")) != -1) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);
                if (N <= 0) {
                    fprintf(stderr, "Error: N must be a positive integer.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                fprintf(stderr, "Usage: %s [-n array_size]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    double* a = malloc(N * sizeof(double));
    double* b = malloc(N * sizeof(double));
    double* sum_seq = malloc(N * sizeof(double));
    double* diff_seq = malloc(N * sizeof(double));
    double* prod_seq = malloc(N * sizeof(double));
    double* quot_seq = malloc(N * sizeof(double));
    double* sum_par = malloc(N * sizeof(double));
    double* diff_par = malloc(N * sizeof(double));
    double* prod_par = malloc(N * sizeof(double));
    double* quot_par = malloc(N * sizeof(double));

    if (!a || !b || !sum_seq || !diff_seq || !prod_seq || !quot_seq ||
        !sum_par || !diff_par || !prod_par || !quot_par) {
        perror("Failed to allocate memory for arrays");
        free(a); free(b); free(sum_seq); free(diff_seq); free(prod_seq); free(quot_seq);
        free(sum_par); free(diff_par); free(prod_par); free(quot_par);
        exit(EXIT_FAILURE);
    }


    srand(time(NULL) ^ getpid());
    for (int i = 0; i < N; i++) {
        a[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        b[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
    }

    double start_seq = omp_get_wtime();
    array_operations_sequential(a, b, sum_seq, diff_seq, prod_seq, quot_seq, N);
    double end_seq = omp_get_wtime();

    double start_par = omp_get_wtime();
    array_operations_parallel(a, b, sum_par, diff_par, prod_par, quot_par, N);
    double end_par = omp_get_wtime();

    printf("Sequential time: %.5f seconds\n", end_seq - start_seq);
    printf("Parallel time: %.5f seconds\n", end_par - start_par);

    free(a); free(b); free(sum_seq); free(diff_seq); free(prod_seq); free(quot_seq);
    free(sum_par); free(diff_par); free(prod_par); free(quot_par);

    return 0;
}