#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <math.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void array_operations_kernel(double* a, double* b, double* sum,
                                       double* diff, double* prod, double* quot, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        sum[idx] = a[idx] + b[idx];
        diff[idx] = a[idx] - b[idx];
        prod[idx] = a[idx] * b[idx];
        quot[idx] = (b[idx] != 0.0) ? a[idx] / b[idx] : 0.0;
    }
}

void array_operations_sequential(double* a, double* b, double* sum,
                                double* diff, double* prod, double* quot, int N) {
    for (int i = 0; i < N; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        quot[i] = (b[i] != 0.0) ? a[i] / b[i] : 0.0;
    }
}

int main(int argc, char* argv[]) {
    int N = 1000000;
    int threadsPerBlock = 256;
    int opt;

    while ((opt = getopt(argc, argv, "n:")) != -1) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);
                if (N <= 0) {
                    fprintf(stderr, "Error: Array size must be a positive integer.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                fprintf(stderr, "Usage: %s [-n array_size]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    double *a, *b, *sum_seq, *diff_seq, *prod_seq, *quot_seq;
    double *sum_par, *diff_par, *prod_par, *quot_par;

    a = (double*)malloc(N * sizeof(double));
    b = (double*)malloc(N * sizeof(double));
    sum_seq = (double*)malloc(N * sizeof(double));
    diff_seq = (double*)malloc(N * sizeof(double));
    prod_seq = (double*)malloc(N * sizeof(double));
    quot_seq = (double*)malloc(N * sizeof(double));
    sum_par = (double*)malloc(N * sizeof(double));
    diff_par = (double*)malloc(N * sizeof(double));
    prod_par = (double*)malloc(N * sizeof(double));
    quot_par = (double*)malloc(N * sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        b[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
    }

    clock_t start_total_time = clock();

    clock_t start_seq_time = clock();
    array_operations_sequential(a, b, sum_seq, diff_seq, prod_seq, quot_seq, N);
    clock_t end_seq_time = clock();
    double sequential_time = (double)(end_seq_time - start_seq_time) / CLOCKS_PER_SEC;

    double *d_a, *d_b, *d_sum, *d_diff, *d_prod, *d_quot;
    CHECK(cudaMalloc((void**)&d_a, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_sum, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_diff, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_prod, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_quot, N * sizeof(double)));

    CHECK(cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));

    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_event, stop_event;
    CHECK(cudaEventCreate(&start_event));
    CHECK(cudaEventCreate(&stop_event));

    CHECK(cudaEventRecord(start_event, 0));
    array_operations_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_sum, d_diff, d_prod, d_quot, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop_event, 0));
    CHECK(cudaEventSynchronize(stop_event));

    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    double parallel_time = milliseconds / 1000.0;

    CHECK(cudaMemcpy(sum_par, d_sum, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(diff_par, d_diff, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(prod_par, d_prod, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(quot_par, d_quot, N * sizeof(double), cudaMemcpyDeviceToHost));

    clock_t end_total_time = clock(); // End total execution time
    double total_time = (double)(end_total_time - start_total_time) / CLOCKS_PER_SEC;

    printf("Sequential time: %.10f seconds\n", sequential_time);
    printf("Parallel time: %.10f seconds\n", parallel_time);

    free(a); free(b);
    free(sum_seq); free(diff_seq); free(prod_seq); free(quot_seq);
    free(sum_par); free(diff_par); free(prod_par); free(quot_par);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_diff));
    CHECK(cudaFree(d_prod));
    CHECK(cudaFree(d_quot));

    CHECK(cudaEventDestroy(start_event));
    CHECK(cudaEventDestroy(stop_event));

    return 0;
}