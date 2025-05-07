#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 100000

void main() {
    double *array = malloc(N * sizeof(double));
    double final_sum = 0.0;

    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        array[i] = rand();
    }

    double start_par = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        final_sum += array[i];
    }
    double end_par = omp_get_wtime();

    printf("Final sum: %f\n", final_sum);
    printf("Sequential time: %.5f seconds\n", end_par - start_par);

    final_sum = 0.0;
    start_par = 0.0;
    end_par = 0.0;

    start_par = omp_get_wtime();

#pragma omp parallel for reduction(+:final_sum)
    for (int i = 0; i < N; i++) {
        final_sum += array[i];
    }
    end_par = omp_get_wtime();

    printf("\nFinal parallel sum: %.2f,\nParallel time: %.5f seconds\n", final_sum, end_par - start_par);

    free(array);
}
