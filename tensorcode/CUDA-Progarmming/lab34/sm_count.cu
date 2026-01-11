#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    cudaDeviceProp prop;
    int device = 0;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("GPU name                : %s\n", prop.name);
    printf("SM count                : %d\n", prop.multiProcessorCount);
    printf("Max threads per SM      : %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block   : %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per SM (est) : %d\n",
           prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per SM: %zu\n", prop.sharedMemPerMultiprocessor);

    return 0;
}
