#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 24)    // 4M elements
#define BLOCK_SIZE 1024

__global__ void reduce0(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();

    // Sequential addressing reduction
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main() {
    // Allocate and initialize host data
    int *h_idata = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_idata[i] = 1;  // Sum should equal N
    }

    int numBlocks = N / BLOCK_SIZE;
    int *h_odata = (int *)malloc(numBlocks * sizeof(int));

    int *d_idata, *d_odata;
    cudaMalloc(&d_idata, N * sizeof(int));
    cudaMalloc(&d_odata, numBlocks * sizeof(int));

    cudaMemcpy(d_idata, h_idata, N * sizeof(int), cudaMemcpyHostToDevice);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    reduce0<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    reduce0<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // Copy partial sums back and finish on CPU
    cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    long long total = 0;
    for (int i = 0; i < numBlocks; i++) {
        total += h_odata[i];
    }

    printf("Reduction result: %lld (expected: %d)\n", total, N);
    printf("Kernel time: %.3f ms\n", kernel_ms);

    // Bandwidth calculation
    double bytes_moved = N * sizeof(int);  // reads from global memory
    double bandwidth_GBs = (bytes_moved / 1e9) / (kernel_ms / 1000.0);

    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_GBs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}
