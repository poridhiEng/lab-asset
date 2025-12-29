#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 24)          // 16,777,216 elements

// Warp-level reduction: unrolled for last 32 threads
template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

// Block-level reduction: completely unrolled
template <unsigned int blockSize>
__global__ void reduce6(int* g_idata, int* g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

    // First add during load
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // Unrolled reduction
    if (blockSize >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    // Warp-level reduction for last 32 threads
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    // Write block result to global memory
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    const unsigned int BLOCK_SIZE = 1024;   // choose block size
    const unsigned int NUM_BLOCKS = N / (BLOCK_SIZE * 2);

    // Host memory
    int* h_idata = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_idata[i] = 1;  // sum = N

    int* h_odata = (int*)malloc(NUM_BLOCKS * sizeof(int));

    // Device memory
    int *d_idata, *d_odata;
    cudaMalloc(&d_idata, N * sizeof(int));
    cudaMalloc(&d_odata, NUM_BLOCKS * sizeof(int));
    cudaMemcpy(d_idata, h_idata, N * sizeof(int), cudaMemcpyHostToDevice);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel using switch-case for template instantiation
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(NUM_BLOCKS);
    size_t smemSize = BLOCK_SIZE * sizeof(int);

    // Warm-up
    switch (BLOCK_SIZE) {
        case 1024: reduce6<1024><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 512:  reduce6<512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 256:  reduce6<256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 128:  reduce6<128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 64:   reduce6<64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 32:   reduce6<32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 16:   reduce6<16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 8:    reduce6<8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 4:    reduce6<4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 2:    reduce6<2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 1:    reduce6<1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
    }
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    switch (BLOCK_SIZE) {
        case 1024: reduce6<1024><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 512:  reduce6<512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 256:  reduce6<256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 128:  reduce6<128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 64:   reduce6<64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 32:   reduce6<32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 16:   reduce6<16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 8:    reduce6<8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 4:    reduce6<4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 2:    reduce6<2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 1:    reduce6<1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // Copy results back
    cudaMemcpy(h_odata, d_odata, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    // CPU finalize sum
    long long total = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) total += h_odata[i];

    printf("Reduction result: %lld (expected: %d)\n", total, N);
    printf("Kernel time: %.3f ms\n", kernel_ms);

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
