#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20) // 1 million elements

__global__ void simpleKernel(int kernel_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[Kernel %d] executed on GPU\n", kernel_id);
}

int main() {
    float *h_data, *d_data;
    cudaStream_t stream;
    size_t size = N * sizeof(float);

    
    h_data = (float*)malloc(size);
    cudaMalloc(&d_data, size);
    cudaStreamCreate(&stream);

    printf("Pageable memory: starting cudaMemcpyAsync...\n");

    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

    simpleKernel<<<1, 1, 0, stream>>>(1);

    cudaStreamSynchronize(stream);

    printf("Pageable memory version finished\n");

    cudaFree(d_data);
    free(h_data);
    cudaStreamDestroy(stream);

    return 0;
}
