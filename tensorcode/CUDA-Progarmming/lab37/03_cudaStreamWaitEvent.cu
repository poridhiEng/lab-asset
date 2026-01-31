#include <cuda_runtime.h>
#include <stdio.h>

/* Producer kernel: writes data */
__global__ void producer_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2;  // simple computation
    }
}

/* Consumer kernel: reads producer data and processes it */
__global__ void consumer_kernel(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + 1;  // simple dependent computation
    }
}

int main() {
    const int N = 4000;
    int *d_data, *d_result;

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(int));

    /* Create streams */
    cudaStream_t producer_stream, consumer_stream;
    cudaStreamCreate(&producer_stream);
    cudaStreamCreate(&consumer_stream);

    /* Create events to measure timing */
    cudaEvent_t producer_start, producer_done;
    cudaEvent_t consumer_start, consumer_done;
    cudaEventCreate(&producer_start);
    cudaEventCreate(&producer_done);
    cudaEventCreate(&consumer_start);
    cudaEventCreate(&consumer_done);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    printf("Launching producer kernel on producer_stream\n");
    cudaEventRecord(producer_start, producer_stream);
    producer_kernel<<<grid, block, 0, producer_stream>>>(d_data, N);
    cudaEventRecord(producer_done, producer_stream);

    printf("Launching consumer kernel on consumer_stream (will wait on producer)\n");
    cudaStreamWaitEvent(consumer_stream, producer_done);
    cudaEventRecord(consumer_start, consumer_stream);
    consumer_kernel<<<grid, block, 0, consumer_stream>>>(d_data, d_result, N);
    cudaEventRecord(consumer_done, consumer_stream);

    /* Wait for consumer to finish */
    cudaEventSynchronize(consumer_done);

    /* Calculate timings */
    float producer_to_consumer_ms;
    float producer_duration_ms, consumer_duration_ms;

    cudaEventElapsedTime(&producer_to_consumer_ms, producer_done, consumer_start);
    cudaEventElapsedTime(&producer_duration_ms, producer_start, producer_done);
    cudaEventElapsedTime(&consumer_duration_ms, consumer_start, consumer_done);

    printf("\n=== Timing Info ===\n");
    printf("Producer kernel duration: %.3f ms\n", producer_duration_ms);
    printf("Consumer kernel duration: %.3f ms\n", consumer_duration_ms);
    printf("Time between producer completion and consumer start: %.3f ms\n",
           producer_to_consumer_ms);

    /* Show first few values of producer output */
    int h_data[10];
    cudaMemcpy(h_data, d_data, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nFirst 10 elements produced: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_data[i]);
    printf("\n");

    /* Show first few values of consumer output */
    int h_result[10];
    cudaMemcpy(h_result, d_result, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("First 10 elements consumed: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_result[i]);
    printf("\n");

    /* Cleanup */
    cudaEventDestroy(producer_start);
    cudaEventDestroy(producer_done);
    cudaEventDestroy(consumer_start);
    cudaEventDestroy(consumer_done);

    cudaStreamDestroy(producer_stream);
    cudaStreamDestroy(consumer_stream);

    cudaFree(d_data);
    cudaFree(d_result);

    printf("\nProgram finished\n");
    return 0;
}
