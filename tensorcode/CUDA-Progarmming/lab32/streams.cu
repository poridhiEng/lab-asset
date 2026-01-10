#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)

__global__ void work(int id)
{
    if (threadIdx.x == 0)
        printf("Kernel from stream %d executing\n", id);

    for(int i=0; i<=1000000000; i++)
    {
        int j=0;
    }

    if (threadIdx.x == 0)
        printf("Kernel from stream %d execition completed\n", id);
}

int main()
{
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    float *h1, *h2;
    // PINNED host memory
    cudaMallocHost(&h1, N * sizeof(float));
    cudaMallocHost(&h2, N * sizeof(float));

    float *d1, *d2;
    cudaMalloc(&d1, N * sizeof(float));
    cudaMalloc(&d2, N * sizeof(float));

    // Stream 1
    work<<<1, 1, 0, s1>>>(1);

    // Stream 2
    work<<<1, 1, 0, s2>>>(2);

    cudaDeviceSynchronize();
}
