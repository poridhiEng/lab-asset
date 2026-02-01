// Demo: cudaStreamAddCallback - host callback after stream completes
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void compute_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * idx;
    }
}

// Callback function - called by CUDA runtime when stream work is done
void CUDART_CB stream_callback(cudaStream_t stream, cudaError_t status, void *userData) {
    int stream_id = *(int*)userData;
    printf("Callback: Stream %d finished with status %d\n", stream_id, status);
    printf("Callback: Performing host-side post-processing...\n");
}

int main() {
    const int N = 1 << 18;
    int *d_data;

    cudaMalloc(&d_data, N * sizeof(int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int stream_id = 1;  // User data to pass to callback

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    printf("Launching kernel...\n");
    compute_kernel<<<grid, block, 0, stream>>>(d_data, N);

    // Register callback - will be called when kernel completes
    cudaStreamAddCallback(stream, stream_callback, &stream_id, 0);

    printf("Host continues immediately (callback registered)\n");

    // Wait for everything including callback
    cudaStreamSynchronize(stream);
    printf("Main: All done\n");

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    return 0;
}
