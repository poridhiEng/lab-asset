#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

__global__ void work_kernel(int sid)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[Stream %d] STARTED\n", sid);

    float x = 0.0f;
    for (int i = 0; i < 50000000; i++)
        x += 1.0f;

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[Stream %d] FINISHED \n", sid, x);
}

int main()
{
    cudaStream_t streams[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(&streams[i]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    //Saturated 
    printf("\nTEST 1: 128 blocks (Sequential) \n");
    for (int i = 0; i < NUM_STREAMS; i++)
        work_kernel<<<256, 546, 0, streams[i]>>>(i);
    cudaDeviceSynchronize();

    // Concurrent 
    printf("\n TEST 2: 1 block (Concurrent) \n");
    for (int i = 0; i < NUM_STREAMS; i++)
        work_kernel<<<1, 256, 0, streams[i]>>>(i);
    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(streams[i]);

    return 0;
}
