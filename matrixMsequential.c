#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMultiplication(float* A, float* B, float* C, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }
}

int main() {
    int width = 1024; // Matrix dimensions
    float *A = (float*)malloc(width * width * sizeof(float));
    float *B = (float*)malloc(width * width * sizeof(float));
    float *C = (float*)malloc(width * width * sizeof(float));

    // Initialize matrices A and B with some values
    for (int i = 0; i < width * width; ++i) {
        A[i] = 1.0f; // Initialize with some value
        B[i] = 2.0f; // Initialize with some value
    }

    // Record start time
    clock_t start_time = clock();

    // Perform matrix multiplication
    matrixMultiplication(A, B, C, width);

    // Record end time
    clock_t end_time = clock();

    // Calculate elapsed time in milliseconds
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;

    printf("Sequential matrix multiplication runtime: %.2f ms\n", elapsed_time);

    // Print resulting matrix C
    /* Uncomment the following code to print the resulting matrix C
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%f ", C[i * width + j]);
        }
        printf("\n");
    }
    */

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}

