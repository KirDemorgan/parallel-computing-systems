#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

__global__ void addKernel(float *a, float *b, float *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) result[idx] = a[idx] + b[idx];
}

__global__ void subtractKernel(float *a, float *b, float *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) result[idx] = a[idx] - b[idx];
}

__global__ void multiplyKernel(float *a, float *b, float *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) result[idx] = a[idx] * b[idx];
}

__global__ void divideKernel(float *a, float *b, float *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) result[idx] = a[idx] / b[idx];
}

void sequentialOperations(float *a, float *b, float *result, int size, char op) {
    for (int i = 0; i < size; i++) {
        switch (op) {
            case '+': result[i] = a[i] + b[i]; break;
            case '-': result[i] = a[i] - b[i]; break;
            case '*': result[i] = a[i] * b[i]; break;
            case '/': result[i] = a[i] / b[i]; break;
        }
    }
}

int main(int argc, char **argv) {
    const int rows = 316;
    const int cols = 316;
    const int size = rows * cols;
    const int bytes = size * sizeof(float);

    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *result_seq = (float *)malloc(bytes);
    float *result_par = (float *)malloc(bytes);

    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        a[i] = (float)(rand() % 100 + 1);
        b[i] = (float)(rand() % 100 + 1);
    }

    float *d_a, *d_b, *d_result;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_result, bytes));
    CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    // Parallel
    cudaEvent_t start, stop;
    float elapsed;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    addKernel<<<grid, block>>>(d_a, d_b, d_result, size);
    subtractKernel<<<grid, block>>>(d_a, d_b, d_result, size);
    multiplyKernel<<<grid, block>>>(d_a, d_b, d_result, size);
    divideKernel<<<grid, block>>>(d_a, d_b, d_result, size);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Parallel time: %.5f seconds\n", elapsed / 1000.0f);

    CHECK(cudaMemcpy(result_par, d_result, bytes, cudaMemcpyDeviceToHost));

    // Sequential
    clock_t seq_start = clock();
    sequentialOperations(a, b, result_seq, size, '+');
    sequentialOperations(a, b, result_seq, size, '-');
    sequentialOperations(a, b, result_seq, size, '*');
    sequentialOperations(a, b, result_seq, size, '/');
    clock_t seq_end = clock();
    float seq_time = (float)(seq_end - seq_start) / CLOCKS_PER_SEC;
    printf("Sequential time: %.5f seconds\n", seq_time);

    free(a); free(b); free(result_seq); free(result_par);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_result));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));

    return 0;
}
