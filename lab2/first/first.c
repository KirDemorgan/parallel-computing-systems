#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    int N = 100000;
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

    double* array = malloc(N * sizeof(double));
    if (array == NULL) {
        perror("Failed to allocate memory for array");
        exit(EXIT_FAILURE);
    }
    double final_sum_seq = 0.0;
    double final_sum_par = 0.0;

    srand(time(NULL) ^ getpid());
    for (int i = 0; i < N; i++) {
        array[i] = (double)rand() / RAND_MAX;
    }

    double start_seq = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        final_sum_seq += array[i];
    }
    double end_seq = omp_get_wtime();

    double start_par = omp_get_wtime();
    #pragma omp parallel for reduction(+:final_sum_par)
    for (int i = 0; i < N; i++) {
        final_sum_par += array[i];
    }
    double end_par = omp_get_wtime();

    printf("Sequential time: %.5f seconds\n", end_seq - start_seq);
    printf("Parallel time: %.5f seconds\n", end_par - start_par);

    free(array);
    return 0;
}