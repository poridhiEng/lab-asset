// Demo: cudaStreamCreateWithPriority - streams with execution priority
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void work_kernel(int *data, int n, int priority) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 1000; i++) {
            data[idx] += priority;
        }
    }
}

int main() {
    // Query priority range (lower number = higher priority)
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    printf("Priority range: %d (highest) to %d (lowest)\n", priority_high, priority_low);

    const int N = 1 << 20;
    int *d_data_high, *d_data_low;

    cudaMalloc(&d_data_high, N * sizeof(int));
    cudaMalloc(&d_data_low, N * sizeof(int));

    // Create streams with different priorities
    cudaStream_t high_priority_stream, low_priority_stream;
    cudaStreamCreateWithPriority(&high_priority_stream, cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&low_priority_stream, cudaStreamNonBlocking, priority_low);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Launch low priority first
    work_kernel<<<grid, block, 0, low_priority_stream>>>(d_data_low, N, 0);
    printf("Low priority kernel launched\n");

    // Launch high priority - may preempt low priority when resources available
    work_kernel<<<grid, block, 0, high_priority_stream>>>(d_data_high, N, 1);
    printf("High priority kernel launched\n");

    cudaDeviceSynchronize();
    printf("Both complete\n");

    // Cleanup
    cudaStreamDestroy(high_priority_stream);
    cudaStreamDestroy(low_priority_stream);
    cudaFree(d_data_high);
    cudaFree(d_data_low);

    return 0;
}
