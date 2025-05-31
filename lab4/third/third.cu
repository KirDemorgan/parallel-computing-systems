#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
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

void process_in_chunks(double* a, double* b, double* sum, double* diff, double* prod, double* quot, int N, int chunk_size, int threadsPerBlock) {
    double *d_a, *d_b, *d_sum, *d_diff, *d_prod, *d_quot;
    CHECK(cudaMalloc((void**)&d_a, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_b, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_sum, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_diff, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_prod, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_quot, chunk_size * sizeof(double)));

    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < N; i += chunk_size) {
        int current_chunk_size = (i + chunk_size > N) ? (N - i) : chunk_size;

        CHECK(cudaMemcpy(d_a, a + i, current_chunk_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b, b + i, current_chunk_size * sizeof(double), cudaMemcpyHostToDevice));

        array_operations_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_sum, d_diff, d_prod, d_quot, current_chunk_size);
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(sum + i, d_sum, current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(diff + i, d_diff, current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(prod + i, d_prod, current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(quot + i, d_quot, current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost));
    }

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_diff));
    CHECK(cudaFree(d_prod));
    CHECK(cudaFree(d_quot));
}

int main() {
    int N = 100000000;
    int threadsPerBlock = 256;
    int chunk_size = 1000000; // Adjust this based on available memory

    double *a, *b, *sum, *diff, *prod, *quot;
    a = (double*)malloc(N * sizeof(double));
    b = (double*)malloc(N * sizeof(double));
    sum = (double*)malloc(N * sizeof(double));
    diff = (double*)malloc(N * sizeof(double));
    prod = (double*)malloc(N * sizeof(double));
    quot = (double*)malloc(N * sizeof(double));

    if (!a || !b || !sum || !diff || !prod || !quot) {
        perror("Memory allocation failed");
        free(a); free(b); free(sum); free(diff); free(prod); free(quot);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        a[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        b[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
    }

    process_in_chunks(a, b, sum, diff, prod, quot, N, chunk_size, threadsPerBlock);

    free(a); free(b); free(sum); free(diff); free(prod); free(quot);
    return 0;
}