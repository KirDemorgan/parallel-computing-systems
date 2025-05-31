#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                               \
        {                                                                       \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                            \
        }                                                                       \
    }

__global__ void addKernel(float *a, float *b, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void subtractKernel(float *a, float *b, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void multiplyKernel(float *a, float *b, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void divideKernel(float *a, float *b, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] / b[idx];
    }
}

void sequentialOperations(float *a, float *b, float *result, int n, char op)
{
    for (int i = 0; i < n; i++)
    {
        switch (op)
        {
        case '+':
            result[i] = a[i] + b[i];
            break;
        case '-':
            result[i] = a[i] - b[i];
            break;
        case '*':
            result[i] = a[i] * b[i];
            break;
        case '/':
            result[i] = a[i] / b[i];
            break;
        }
    }
}

void merge(float *arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    float *L = (float *)malloc(n1 * sizeof(float));
    float *R = (float *)malloc(n2 * sizeof(float));

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

/*ПОСЛЕДОВАТЕЛЬНАЯ СОРТИРОВКА СЛИЯНИЕМ*/
void mergeSort(float *arr, int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

__global__ void mergeKernel(float *arr, int size, int segment_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int segment = idx * segment_size * 2;

    if (segment >= size)
        return;

    int end = min(segment + segment_size * 2, size);
    int middle = min(segment + segment_size, size);

    int i = segment;
    int j = middle;
    int k = 0;

    float *temp = (float *)malloc((end - segment) * sizeof(float));

    while (i < middle && j < end)
    {
        if (arr[i] < arr[j])
        {
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
        }
    }

    while (i < middle)
        temp[k++] = arr[i++];
    while (j < end)
        temp[k++] = arr[j++];

    for (int m = 0; m < k; m++)
    {
        arr[segment + m] = temp[m];
    }

    free(temp);
}

/*ПАРАЛЛЕЛЬНАЯ СОРТИРОВКА СЛИЯНИЕМ*/
void parallelMergeSort(float *arr, int size)
{
    float *d_arr;
    CHECK(cudaMalloc((void **)&d_arr, size * sizeof(float)));
    CHECK(cudaMemcpy(d_arr, arr, size * sizeof(float), cudaMemcpyHostToDevice));

    for (int segment_size = 1; segment_size < size; segment_size *= 2)
    {
        int num_blocks = (size + 2 * segment_size - 1) / (2 * segment_size);
        mergeKernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, size, segment_size);
        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(arr, d_arr, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_arr));
}

int main(int argc, char **argv)
{
    int n = 100000;
    int threads = 256;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            n = atoi(argv[i + 1]);
            i++;
        }
    }

    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *result_seq = (float *)malloc(n * sizeof(float));
    float *result_par = (float *)malloc(n * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < n; i++)
    {
        a[i] = (float)rand() / RAND_MAX * 100.0f;
        b[i] = (float)rand() / RAND_MAX * 100.0f + 0.1f; // Чтобы избежать деления на 0
    }

    float *d_a, *d_b, *d_result;
    CHECK(cudaMalloc((void **)&d_a, n * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_result, n * sizeof(float)));

    CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(threads);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));

    addKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, n);
    CHECK(cudaMemcpy(result_par, d_result, n * sizeof(float), cudaMemcpyDeviceToHost));

    subtractKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, n);
    multiplyKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, n);
    divideKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, n);

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float parallel_time = 0;
    CHECK(cudaEventElapsedTime(&parallel_time, start, stop));
    parallel_time /= 1000.0f;

    clock_t seq_start = clock();

    sequentialOperations(a, b, result_seq, n, '+');
    sequentialOperations(a, b, result_seq, n, '-');
    sequentialOperations(a, b, result_seq, n, '*');
    sequentialOperations(a, b, result_seq, n, '/');

    clock_t seq_end = clock();
    float sequential_time = (float)(seq_end - seq_start) / CLOCKS_PER_SEC;

    float *arr_seq = (float *)malloc(n * sizeof(float));
    float *arr_par = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        arr_seq[i] = a[i];
        arr_par[i] = a[i];
    }

    clock_t sort_seq_start = clock();
    mergeSort(arr_seq, 0, n - 1);
    clock_t sort_seq_end = clock();
    float sort_seq_time = (float)(sort_seq_end - sort_seq_start) / CLOCKS_PER_SEC;

    clock_t sort_par_start = clock();
    parallelMergeSort(arr_par, n);
    clock_t sort_par_end = clock();
    float sort_par_time = (float)(sort_par_end - sort_par_start) / CLOCKS_PER_SEC;

    printf("Array operations:\n");
    printf("Parallel time: %.5f seconds\n", parallel_time);
    printf("Sequential time: %.5f seconds\n\n", sequential_time);

    printf("Merge sort:\n");
    printf("Parallel time: %.5f seconds\n", sort_par_time);
    printf("Sequential time: %.5f seconds\n", sort_seq_time);

    free(a);
    free(b);
    free(result_seq);
    free(result_par);
    free(arr_seq);
    free(arr_par);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_result));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}