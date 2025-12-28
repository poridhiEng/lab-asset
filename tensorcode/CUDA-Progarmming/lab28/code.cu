#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 24)          // 16,777,216 elements
#define BLOCK_SIZE 1024

// Warp-level reduction for the last 32 threads
__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduceWarpUnroll(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + tid;

    // First add during global load
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // Reduction in shared memory (sequential addressing)
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level reduction for last 32 threads
    if (tid < 32)
        warpReduce(sdata, tid);

    // Write result for this block to global memory
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

int main() {
    // Host allocations
    int *h_idata = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        h_idata[i] = 1;

    int numBlocks = N / (BLOCK_SIZE * 2);
    int *h_odata = (int *)malloc(numBlocks * sizeof(int));

    // Device allocations
    int *d_idata, *d_odata;
    cudaMalloc(&d_idata, N * sizeof(int));
    cudaMalloc(&d_odata, numBlocks * sizeof(int));

    cudaMemcpy(d_idata, h_idata, N * sizeof(int), cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    reduceWarpUnroll<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    reduceWarpUnroll<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // Copy back partial sums
    cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    long long total = 0;
    for (int i = 0; i < numBlocks; i++)
        total += h_odata[i];

    printf("Reduction result: %lld (expected: %d)\n", total, N);
    printf("Kernel time: %.3f ms\n", kernel_ms);

    // Bandwidth calculation (global reads only)
    double bytes_moved = N * sizeof(int);
    double bandwidth_GBs = (bytes_moved / 1e9) / (kernel_ms / 1000.0);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_GBs);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}
