//
// Created by demorgan on 13.04.2025.
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>

void swap(double* a, double* b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

int partition(double* arr, int low, int high) {
    double pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quicksort_sequential(double* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort_sequential(arr, low, pi - 1);
        quicksort_sequential(arr, pi + 1, high);
    }
}

void quicksort_parallel(double* arr, int low, int high) {
        int pi = partition(arr, low, high);

        #pragma omp task
        quicksort_parallel(arr, low, pi - 1);

        #pragma omp task
        quicksort_parallel(arr, pi + 1, high);
    }

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

    double* array_seq = malloc(N * sizeof(double));
    double* array_par = malloc(N * sizeof(double));

    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        array_seq[i] = array_par[i] = (double)rand() / RAND_MAX * 1000.0;
    }

    double start_seq = omp_get_wtime();
    quicksort_sequential(array_seq, 0, N - 1);
    double end_seq = omp_get_wtime();

    printf("Sequential quicksort:\n");
    printf("Sequential time: %.5f seconds\n\n", end_seq - start_seq);

    double start_par = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quicksort_parallel(array_par, 0, N - 1);
    }
    double end_par = omp_get_wtime();

    printf("Parallel quicksort:\n");
    printf("Parallel time: %.5f seconds\n", end_par - start_par);

    free(array_seq);
    free(array_par);

    return 0;
}