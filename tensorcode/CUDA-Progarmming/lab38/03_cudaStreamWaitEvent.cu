#include <cuda_runtime.h>
#include <stdio.h>


__global__ void producer_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate some work
        for (int i = 0; i < 1000; i++) {
            data[idx] = idx * 2 + i - i;
        }
    }
}

/* Consumer kernel: reads producer data and processes it */
__global__ void consumer_kernel(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + 1;
    }
}

int main() {
    const int N = 100000;
    int *d_data, *d_result;

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(int));

    /* Create streams */
    cudaStream_t producer_stream, consumer_stream;
    cudaStreamCreate(&producer_stream);
    cudaStreamCreate(&consumer_stream);

    /*
     * API 1: cudaEventCreate(event)
     * Creates a CUDA event object
     */
    printf("=== API 1: cudaEventCreate ===\n");
    cudaEvent_t producer_start, producer_done;
    cudaEvent_t consumer_start, consumer_done;
    cudaEventCreate(&producer_start);
    cudaEventCreate(&producer_done);
    cudaEventCreate(&consumer_start);
    cudaEventCreate(&consumer_done);
    printf("Created 4 events: producer_start, producer_done, consumer_start, consumer_done\n\n");

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    /*
     * API 2: cudaEventRecord(event, stream)
     * Inserts event into stream - triggered when all prior work completes
     */
    printf("=== API 2: cudaEventRecord ===\n");
    printf("Recording producer_start before kernel launch\n");
    cudaEventRecord(producer_start, producer_stream);

    printf("Launching producer kernel on producer_stream\n");
    producer_kernel<<<grid, block, 0, producer_stream>>>(d_data, N);

    printf("Recording producer_done after kernel launch\n");
    cudaEventRecord(producer_done, producer_stream);
    printf("\n");

    /*
     * API 5: cudaEventQuery(event)
     * Non-blocking check - returns cudaSuccess if event completed
     * Returns cudaErrorNotReady if still pending
     */
    printf("=== API 5: cudaEventQuery ===\n");
    int query_count = 0;
    cudaError_t status;

    printf("Polling producer_done event (non-blocking)...\n");
    while ((status = cudaEventQuery(producer_done)) == cudaErrorNotReady) {
        query_count++;
        // Host can do other work here while waiting
    }

    if (status == cudaSuccess) {
        printf("Producer completed! Polled %d times before completion\n", query_count);
    } else {
        printf("Error: %s\n", cudaGetErrorString(status));
    }
    printf("\n");

    /*
     * API 4: cudaStreamWaitEvent(stream, event)
     * Makes stream wait for event - GPU-side sync, host doesn't block!
     */
    printf("=== API 4: cudaStreamWaitEvent ===\n");
    printf("Consumer stream will wait for producer_done event (GPU-side wait)\n");
    cudaStreamWaitEvent(consumer_stream, producer_done, 0);

    cudaEventRecord(consumer_start, consumer_stream);
    printf("Launching consumer kernel on consumer_stream\n");
    consumer_kernel<<<grid, block, 0, consumer_stream>>>(d_data, d_result, N);
    cudaEventRecord(consumer_done, consumer_stream);
    printf("Host continues immediately - no blocking!\n\n");

    /*
     * API 3: cudaEventSynchronize(event)
     * Blocks host until event has occurred
     */
    printf("=== API 3: cudaEventSynchronize ===\n");
    printf("Host waiting for consumer_done event...\n");
    cudaEventSynchronize(consumer_done);
    printf("Consumer completed! Host can now safely access results\n\n");

    /*
     * API 6: cudaEventElapsedTime(time, start, stop)
     * Measures elapsed time between two recorded events
     */
    printf("=== API 6: cudaEventElapsedTime ===\n");
    float producer_duration_ms, consumer_duration_ms;
    float producer_to_consumer_ms, total_time_ms;

    cudaEventElapsedTime(&producer_duration_ms, producer_start, producer_done);
    cudaEventElapsedTime(&consumer_duration_ms, consumer_start, consumer_done);
    cudaEventElapsedTime(&producer_to_consumer_ms, producer_done, consumer_start);
    cudaEventElapsedTime(&total_time_ms, producer_start, consumer_done);

    printf("Producer kernel duration:    %.3f ms\n", producer_duration_ms);
    printf("Consumer kernel duration:    %.3f ms\n", consumer_duration_ms);
    printf("Gap (producer -> consumer):  %.3f ms\n", producer_to_consumer_ms);
    printf("Total time (start to end):   %.3f ms\n", total_time_ms);
    printf("\n");

    /* Verify data correctness */
    printf("=== Data Verification ===\n");
    int h_data[10], h_result[10];
    cudaMemcpy(h_data, d_data, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result, d_result, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Producer output (idx * 2):  ");
    for (int i = 0; i < 10; i++) printf("%d ", h_data[i]);
    printf("\n");

    printf("Consumer output (+ 1):      ");
    for (int i = 0; i < 10; i++) printf("%d ", h_result[i]);
    printf("\n\n");

    /* Cleanup */
    cudaEventDestroy(producer_start);
    cudaEventDestroy(producer_done);
    cudaEventDestroy(consumer_start);
    cudaEventDestroy(consumer_done);

    cudaStreamDestroy(producer_stream);
    cudaStreamDestroy(consumer_stream);

    cudaFree(d_data);
    cudaFree(d_result);

    printf("Program finished\n");
    return 0;
}
