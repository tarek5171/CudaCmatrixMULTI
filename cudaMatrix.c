#include <stdio.h>
#include <cuda_runtime.h>

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    // Calculate the row index of the P element and M
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of P and N
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < Width) && (Col < Width)) {
        float Pvalue = 0; // each thread computes one element of the block sub-matrix

        for (int k = 0; k < Width; ++k) {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }

        P[Row * Width + Col] = Pvalue;
    }
}

int main() {
    // Matrix dimensions
    int Width = 1024; // for example

    // Allocate memory on the host
    float *h_M = (float*)malloc(Width * Width * sizeof(float));
    float *h_N = (float*)malloc(Width * Width * sizeof(float));
    float *h_P = (float*)malloc(Width * Width * sizeof(float));

    // Initialize matrices M and N with some values
    for (int i = 0; i < Width * Width; ++i) {
        h_M[i] = 1.0f; // Initialize with some value
        h_N[i] = 1.0f; // Initialize with some value
    }

    // Allocate memory on the device
    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, Width * Width * sizeof(float));
    cudaMalloc(&d_N, Width * Width * sizeof(float));
    cudaMalloc(&d_P, Width * Width * sizeof(float));

    // Copy matrices M and N from host to device
    cudaMemcpy(d_M, h_M, Width * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, Width * Width * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions with fewer threads per block
    dim3 dimBlock(12, 12); 
    dim3 dimGrid((Width + dimBlock.x - 1) / dimBlock.x, (Width + dimBlock.y - 1) / dimBlock.y);

    // Create CUDA events for measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch kernel
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result from device to host
    cudaMemcpy(h_P, d_P, Width * Width * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);

    int totalThreadsPerBlock = dimBlock.x * dimBlock.y;
    int totalBlocks = dimGrid.x * dimGrid.y;
    int totalThreads = totalThreadsPerBlock * totalBlocks;

    printf("Total number of threads: %d\n", totalThreads);      

    return 0;
}


