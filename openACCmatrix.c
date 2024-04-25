#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void computeAcc(float *P, const float *M, const float *N, int Mh, int Mw, int Nw) {
    #pragma acc parallel loop copyin(M[0:Mh*Mw]) copyin(N[0:Mw*Nw]) copyout(P[0:Mh*Nw])
    for (int i = 0; i < Mh; i++) {
        #pragma acc loop
        for (int j = 0; j < Nw; j++) {
            float sum = 0;
            for (int k = 0; k < Mw; k++) {
                float a = M[i * Mw + k];
                float b = N[k * Nw + j];
                sum += a * b;
            }
            P[i * Nw + j] = sum;
        }
    }
}

int main() {
    int Mh = 1024; // Number of rows in M
    int Mw = 1024; // Number of columns in M and rows in N
    int Nw = 1024; // Number of columns in N
    float *h_M = (float*)malloc(Mh * Mw * sizeof(float));
    float *h_N = (float*)malloc(Mw * Nw * sizeof(float));
    float *h_P = (float*)malloc(Mh * Nw * sizeof(float));

    // Initialize matrices M and N with some values
    for (int i = 0; i < Mh * Mw; ++i) {
        h_M[i] = 1.0f; // Initialize with some value
    }
    for (int i = 0; i < Mw * Nw; ++i) {
        h_N[i] = 2.0f; // Initialize with some value
    }

    float *d_M, *d_N, *d_P;

    d_M = h_M;
    d_N = h_N;
    d_P = h_P;

    clock_t start_time = clock(); // Record start time

    computeAcc(d_P, d_M, d_N, Mh, Mw, Nw);

    clock_t end_time = clock(); // Record end time

    // Calculate elapsed time in milliseconds
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;

    printf("Kernel execution time: %.2f ms\n", elapsed_time);

    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
