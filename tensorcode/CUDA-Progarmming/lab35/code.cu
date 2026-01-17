
#include <stdio.h>
#include <cuda_runtime.h>
#define N 256

__global__ void tiny_default()
{
    if (threadIdx.x == 0) {
        printf("default stream started \n");
    }

    // approximate delay
    unsigned long long start = clock64();

    // adjust this value for delay length
    const unsigned long long delay_cycles = 1e8; 

    while (clock64() - start < delay_cycles) {
        // busy wait
    }
    if (threadIdx.x == 0) {
        printf("default stream ended\n\n");
    }
}

__global__ void worker(int stream_id)
{
    int tid = threadIdx.x;
    if (tid == 0) {
        printf("[stream %d] block start\n", stream_id);
    }
    if (tid == 0) {
        printf("[stream %d] block end\n\n", stream_id);
    }
}
int main()
{
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }
    for (int i = 0; i < 4; i++) {
        // stream-specific kernel
        worker<<<1, 64, 0, streams[i]>>>(i);
        // default stream kernel
        tiny_default<<<1,1>>>();
    }
    cudaDeviceSynchronize();
    return 0;
}

