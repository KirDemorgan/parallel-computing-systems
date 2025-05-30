#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <getopt.h>
#include <math.h>

void initialize_array(double **array, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            array[i][j] = (double)rand() / RAND_MAX * 100.0 + 1.0; // +1.0 to avoid zeros
        }
    }
}

void sequential_operations(double **a, double **b, double **result, int rows, int cols, char op)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            switch (op)
            {
            case '+':
                result[i][j] = a[i][j] + b[i][j];
                break;
            case '-':
                result[i][j] = a[i][j] - b[i][j];
                break;
            case '*':
                result[i][j] = a[i][j] * b[i][j];
                break;
            case '/':
                if (b[i][j] != 0.0)
                {
                    result[i][j] = a[i][j] / b[i][j];
                }
                else
                {
                    result[i][j] = 0.0;
                }
                break;
            }
        }
    }
}

void parallel_operations(double **a, double **b, double **result, int rows, int cols, char op)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            switch (op)
            {
            case '+':
                result[i][j] = a[i][j] + b[i][j];
                break;
            case '-':
                result[i][j] = a[i][j] - b[i][j];
                break;
            case '*':
                result[i][j] = a[i][j] * b[i][j];
                break;
            case '/':
                if (b[i][j] != 0.0)
                {
                    result[i][j] = a[i][j] / b[i][j];
                }
                else
                {
                    result[i][j] = 0.0;
                }
                break;
            }
        }
    }
}

double **allocate_array(int rows, int cols)
{
    double **array = (double **)malloc(rows * sizeof(double *));
    if (array == NULL)
        return NULL;

    for (int i = 0; i < rows; i++)
    {
        array[i] = (double *)malloc(cols * sizeof(double));
        if (array[i] == NULL)
        {
            // Free previously allocated memory
            for (int j = 0; j < i; j++)
                free(array[j]);
            free(array);
            return NULL;
        }
    }
    return array;
}

void free_array(double **array, int rows)
{
    if (array == NULL)
        return;

    for (int i = 0; i < rows; i++)
    {
        if (array[i] != NULL)
        {
            free(array[i]);
        }
    }
    free(array);
}

int main(int argc, char *argv[])
{
    int n = 100000; // default number of elements
    int opt;

    // Parse command line arguments
    while ((opt = getopt(argc, argv, "n:")) != -1)
    {
        switch (opt)
        {
        case 'n':
            n = atoi(optarg);
            if (n < 100000)
            {
                printf("Array size must be at least 100000. Using default 100000.\n");
                n = 100000;
            }
            break;
        default:
            fprintf(stderr, "Usage: %s -n <number_of_elements>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    // Create rectangular arrays to get exactly n elements
    int rows = (int)sqrt(n);
    int cols = n / rows;
    if (rows * cols < n)
        cols++; // Ensure we have at least n elements

    // Allocate memory for arrays
    double **a = allocate_array(rows, cols);
    double **b = allocate_array(rows, cols);
    double **result_seq = allocate_array(rows, cols);
    double **result_par = allocate_array(rows, cols);

    if (a == NULL || b == NULL || result_seq == NULL || result_par == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize arrays with random values
    srand(time(NULL));
    initialize_array(a, rows, cols);
    initialize_array(b, rows, cols);

    // Test each operation
    char operations[] = {'+', '-', '*', '/'};
    const char *op_names[] = {"Addition", "Subtraction", "Multiplication", "Division"};

    for (int op_idx = 0; op_idx < 4; op_idx++)
    {
        char op = operations[op_idx];
        printf("\nOperation: %s\n", op_names[op_idx]);

        // Sequential version
        double start_seq = omp_get_wtime();
        sequential_operations(a, b, result_seq, rows, cols, op);
        double end_seq = omp_get_wtime();
        printf("Sequential time: %.5f seconds\n", end_seq - start_seq);

        // Parallel version (using maximum available threads)
        omp_set_num_threads(omp_get_max_threads());
        double start_par = omp_get_wtime();
        parallel_operations(a, b, result_par, rows, cols, op);
        double end_par = omp_get_wtime();
        printf("Parallel time: %.5f seconds\n", end_par - start_par);

        // Verify results
        int errors = 0;
        for (int i = 0; i < rows && errors < 10; i++)
        {
            for (int j = 0; j < cols && errors < 10; j++)
            {
                if (fabs(result_seq[i][j] - result_par[i][j]) > 1e-9)
                {
                    errors++;
                    if (errors <= 10)
                    {
                        printf("Mismatch at [%d][%d]: seq=%.5f, par=%.5f\n",
                               i, j, result_seq[i][j], result_par[i][j]);
                    }
                }
            }
        }
        if (errors > 0)
        {
            printf("Total mismatches: %d\n", errors);
        }
    }

    // Free memory
    free_array(a, rows);
    free_array(b, rows);
    free_array(result_seq, rows);
    free_array(result_par, rows);

    return 0;
}