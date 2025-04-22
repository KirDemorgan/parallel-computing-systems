//
// Created by demorgan on 22.04.2025.
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 100000

void array_operations_sequential(double* a, double* b, double* sum, double* diff, double* prod, double* quot) {
    for (int i = 0; i < N; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        quot[i] = a[i] / b[i];
    }
}

void array_operations_parallel(double* a, double* b, double* sum, double* diff, double* prod, double* quot) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        quot[i] = a[i] / b[i];
    }
}

int main() {
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

    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        a[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        b[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
    }

    double start_seq = omp_get_wtime();
    array_operations_sequential(a, b, sum_seq, diff_seq, prod_seq, quot_seq);
    double end_seq = omp_get_wtime();

    printf("Sequential operations:\n");
    printf("Elapsed time: %.5f seconds\n\n", end_seq - start_seq);

    double start_par = omp_get_wtime();
    array_operations_parallel(a, b, sum_par, diff_par, prod_par, quot_par);
    double end_par = omp_get_wtime();

    printf("Parallel operations:\n");
    printf("Elapsed time: %.5f seconds\n\n", end_par - start_par);

    printf("First 5 elements comparison:\n");
    printf("Index\tSum\t\tDiff\t\tProd\t\tQuot\n");
    for (int i = 0; i < 5; i++) {
        printf("%d\t%.2f/%.2f\t%.2f/%.2f\t%.2f/%.2f\t%.2f/%.2f\n",
               i,
               sum_seq[i], sum_par[i],
               diff_seq[i], diff_par[i],
               prod_seq[i], prod_par[i],
               quot_seq[i], quot_par[i]);
    }

    free(a);
    free(b);
    free(sum_seq);
    free(diff_seq);
    free(prod_seq);
    free(quot_seq);
    free(sum_par);
    free(diff_par);
    free(prod_par);
    free(quot_par);

    return 0;
}