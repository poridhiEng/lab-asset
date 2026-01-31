#include <cuda_runtime.h>
#include <stdio.h>

__global__ void work_kernel_stream1() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0) printf("Stream 1 completed\n");
}

__global__ void work_kernel_stream2() {
    unsigned long long start = clock64();
    while (clock64() - start < 9000000000ULL) {}
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
        printf("Stream 2 completed\n");
}


__global__ void dependent_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0) printf("A kernel depending on kernel of stream 1 \n");
}

int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch kernels on different streams
    work_kernel_stream1<<<2, 32, 0, stream1>>>();
    work_kernel_stream2<<<2, 32, 0, stream2>>>();

    printf("Kernels launched on both streams \n");

    // Wait ONLY for stream1 - stream2 keeps running
    cudaStreamSynchronize(stream1);
    printf("Stream1 complete - stream2 may still be running\n");

    dependent_kernel<<<2, 32, 0, stream1>>>();
    cudaStreamSynchronize(stream1);

    cudaStreamSynchronize(stream2);

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    printf("end");
    return 0;
}
