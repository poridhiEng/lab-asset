#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 24)        // 16,777,216
#define BLOCK_SIZE 1024
#define numBlocks 8192

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = 0;
    while (i < n) { 
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    if (tid < 32)
        warpReduce<blockSize>(sdata, tid);

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}


int main() {
    int *h_idata, *h_odata;
    int *d_idata, *d_odata;

    size_t bytes = N * sizeof(int);

    // Host memory
    h_idata = (int *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_idata[i] = 1;

    // Number of blocks
    // int numBlocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    h_odata = (int *)malloc(numBlocks * sizeof(int));

    // Device memory
    cudaMalloc(&d_idata, bytes);
    cudaMalloc(&d_odata, numBlocks * sizeof(int));

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    // Warm-up launch
    reduce6<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata, N);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce6<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    long long total = 0;
    for (int i = 0; i < numBlocks; i++)
        total += h_odata[i];

    printf("Reduction result: %lld (expected: %d)\n", total, N);
    printf("Kernel time: %.3f ms\n", ms);

    double bandwidth = (double)(N * sizeof(int)) / (ms * 1e6);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}
