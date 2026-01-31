// Demo: cudaStreamQuery - non-blocking status check
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void long_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 10000; i++) {
            data[idx] += 1;
        }
    }
}

int main() {
    const int N = 1 << 20;
    int *d_data;

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemset(d_data, 0, N * sizeof(int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    long_kernel<<<grid, block, 0, stream>>>(d_data, N);
    printf("Kernel launched\n");

    // Poll without blocking
    int poll_count = 0;
    cudaError_t status;

    do {
        status = cudaStreamQuery(stream);
        poll_count++;

        if (status == cudaErrorNotReady) {
            // Do useful host work while GPU is busy
            printf("Poll %d: Still running... (doing host work)\n", poll_count);
        }
    } while (status == cudaErrorNotReady);

    if (status == cudaSuccess) {
        printf("Stream complete after %d polls\n", poll_count);
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    return 0;
}
