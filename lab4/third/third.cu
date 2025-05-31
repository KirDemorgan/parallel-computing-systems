#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <unistd.h>

// Макрос для проверки ошибок CUDA
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Ошибка: %s:%d, ", __FILE__, __LINE__); \
        printf("код: %d, причина: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Ядро CUDA для выполнения операций над массивами
__global__ void array_operations_kernel(double* a, double* b, double* sum, 
                                       double* diff, double* prod, double* quot, int N) {
    // Вычисляем глобальный индекс элемента
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверяем, чтобы индекс не выходил за границы массива
    if (idx < N) {
        // Параллельно выполняем все операции над элементами
        sum[idx] = a[idx] + b[idx];    // Сложение
        diff[idx] = a[idx] - b[idx];   // Вычитание
        prod[idx] = a[idx] * b[idx];   // Умножение
        // Деление с проверкой деления на ноль
        quot[idx] = (b[idx] != 0.0) ? a[idx] / b[idx] : 0.0;
    }
}

// Последовательная версия операций над массивами
void array_operations_sequential(double* a, double* b, double* sum, 
                                double* diff, double* prod, double* quot, int N) {
    for (int i = 0; i < N; i++) {
        sum[i] = a[i] + b[i];         // Сложение
        diff[i] = a[i] - b[i];        // Вычитание
        prod[i] = a[i] * b[i];        // Умножение
        // Деление с проверкой деления на ноль
        quot[i] = (b[i] != 0.0) ? a[i] / b[i] : 0.0;
    }
}

int main(int argc, char* argv[]) {
    int N = 1000000;         // Размер массивов по умолчанию
    int threadsPerBlock = 256; // Количество потоков в блоке по умолчанию
    int opt;

    // Разбор аргументов командной строки
    while ((opt = getopt(argc, argv, "n:t:")) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);  // Установка размера массивов
                if (N <= 0) {
                    fprintf(stderr, "Ошибка: N должно быть положительным числом.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            case 't':
                threadsPerBlock = atoi(optarg); // Установка количества потоков
                if (threadsPerBlock <= 0) {
                    fprintf(stderr, "Ошибка: Количество потоков должно быть положительным.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                fprintf(stderr, "Использование: %s [-n размер_массива] [-t потоков_в_блоке]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    // Выделение памяти на хосте (CPU)
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

    // Проверка успешности выделения памяти
    if (!a || !b || !sum_seq || !diff_seq || !prod_seq || !quot_seq ||
        !sum_par || !diff_par || !prod_par || !quot_par) {
        perror("Ошибка выделения памяти для массивов");
        free(a); free(b); free(sum_seq); free(diff_seq); free(prod_seq); free(quot_seq);
        free(sum_par); free(diff_par); free(prod_par); free(quot_par);
        exit(EXIT_FAILURE);
    }

    // Инициализация массивов случайными значениями
    srand(time(NULL) ^ getpid());
    for (int i = 0; i < N; i++) {
        a[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        b[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
    }

    // Последовательное выполнение операций
    clock_t start_seq = clock();
    array_operations_sequential(a, b, sum_seq, diff_seq, prod_seq, quot_seq, N);
    clock_t end_seq = clock();
    double sequential_time = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;

    // Выделение памяти на устройстве (GPU)
    double *d_a, *d_b, *d_sum, *d_diff, *d_prod, *d_quot;
    CHECK(cudaMalloc((void**)&d_a, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_sum, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_diff, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_prod, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_quot, N * sizeof(double)));

    // Копирование данных на устройство
    CHECK(cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));

    // Настройка параметров запуска ядра
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Создание событий CUDA для измерения времени
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Запись начального события
    CHECK(cudaEventRecord(start));

    // Запуск ядра на GPU
    array_operations_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_sum, d_diff, d_prod, d_quot, N);
    CHECK(cudaGetLastError());

    // Запись конечного события
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    // Расчет времени выполнения на GPU
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double parallel_time = milliseconds / 1000.0;

    // Копирование результатов обратно на хост
    CHECK(cudaMemcpy(sum_par, d_sum, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(diff_par, d_diff, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(prod_par, d_prod, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(quot_par, d_quot, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Проверка результатов (опционально)
    for (int i = 0; i < N; i++) {
        if (sum_seq[i] != sum_par[i] || diff_seq[i] != diff_par[i] || 
            prod_seq[i] != prod_par[i] || quot_seq[i] != quot_par[i]) {
            printf("Несоответствие результатов на индексе %d\n", i);
            break;
        }
    }

    // Вывод результатов измерения времени
    printf("Sequential time: %.10f seconds\n", sequential_time);
    printf("Parallel time: %.10f seconds\n", parallel_time);

    // Освобождение ресурсов
    free(a); free(b); 
    free(sum_seq); free(diff_seq); free(prod_seq); free(quot_seq);
    free(sum_par); free(diff_par); free(prod_par); free(quot_par);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_diff));
    CHECK(cudaFree(d_prod));
    CHECK(cudaFree(d_quot));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}