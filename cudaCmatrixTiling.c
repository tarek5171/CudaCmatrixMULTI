#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Pvalue = 0;

    for (int p = 0; p < Width / TILE_WIDTH; ++p) {
        ds_M[ty][tx] = M[Row * Width + p * TILE_WIDTH + tx];
        ds_N[ty][tx] = N[(p * TILE_WIDTH + ty) * Width + Col];

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            Pvalue += ds_M[ty][i] * ds_N[i][tx];

        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

int main() {
    int Width = 1024;

    float *h_M = (float*)malloc(Width * Width * sizeof(float));
    float *h_N = (float*)malloc(Width * Width * sizeof(float));
    float *h_P = (float*)malloc(Width * Width * sizeof(float));

    for (int i = 0; i < Width * Width; ++i) {
        h_M[i] = 1.0f;
        h_N[i] = 2.0f;
    }

    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, Width * Width * sizeof(float));
    cudaMalloc(&d_N, Width * Width * sizeof(float));
    cudaMalloc(&d_P, Width * Width * sizeof(float));

    cudaMemcpy(d_M, h_M, Width * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, Width * Width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((Width + dimBlock.x - 1) / dimBlock.x, (Width + dimBlock.y - 1) / dimBlock.y);

    // Create CUDA events for measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

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

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
