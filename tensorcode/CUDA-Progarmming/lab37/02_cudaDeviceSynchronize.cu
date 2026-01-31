#include <cuda_runtime.h>
#include <stdio.h>

/* Fast kernel on stream1 */
__global__ void work_kernel_stream1() {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("Stream 1 completed\n");
}

/* Very long kernel on stream2 */
__global__ void work_kernel_stream2() {
    unsigned long long start = clock64();
    while (clock64() - start < 9000000000ULL) {
        /* busy wait */
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("Stream 2 completed\n");
}

/* Kernel that logically depends only on stream1 */
__global__ void dependent_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("Dependent kernel executed (stream1 only)\n");
}

int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    /* Launch independent work */
    work_kernel_stream1<<<1, 1, 0, stream1>>>();
    work_kernel_stream2<<<1, 1, 0, stream2>>>();

    printf("Kernels launched on both streams\n");

    /*
     * BAD PRACTICE:
     * This blocks the host until ALL streams finish.
     * stream2 delay now stalls everything.
     */
    cudaDeviceSynchronize();
    printf("All streams forced to complete (global barrier)\n");

    /* Dependent kernel is launched ONLY AFTER stream2 finishes */
    dependent_kernel<<<1, 1, 0, stream1>>>();
    cudaDeviceSynchronize();

    printf("Dependent kernel finished (delayed by stream2)\n");

    /* Cleanup */
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    printf("Program end\n");
    return 0;
}


