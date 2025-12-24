#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("  Memory clock rate: %.2f GHz\n", prop.memoryClockRate / 1000000.0f);
    }

    int currentDevice;
    cudaGetDevice(&currentDevice);
    printf("Currently active device: %d\n", currentDevice);

    return 0;
}
