/*
 * cuBLAS with CUDA Streams
 *
 * By default every cuBLAS call goes into stream 0 (the default stream).
 * All operations in stream 0 execute sequentially.
 *
 * Assigning a cuBLAS handle to a non-default stream lets that handle's
 * operations run concurrently with other streams.
 *
 * This file shows:
 *   1. Sequential execution (default stream)
 *   2. Concurrent execution (separate streams)
 *   3. Timing both to show the difference
 *
 * Compile: nvcc -o streams 01_streams.cu -lcublas
 */

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUBLAS(call) { \
    cublasStatus_t _e = (call); \
    if (_e != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %d at line %d\n", _e, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    int n = 512;
    size_t sz = (size_t)n * n * sizeof(float);

    /* Allocate 4 independent matrix pairs */
    float *d_A[4], *d_B[4], *d_C[4];
    for (int i = 0; i < 4; i++) {
        cudaMalloc((void**)&d_A[i], sz);
        cudaMalloc((void**)&d_B[i], sz);
        cudaMalloc((void**)&d_C[i], sz);
        cudaMemset(d_A[i], 1, sz);
        cudaMemset(d_B[i], 1, sz);
        cudaMemset(d_C[i], 0, sz);
    }

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    /* ----------------------------------------------------------------
     * Demo 1: Sequential — one handle, default stream
     *
     * All 4 GEMMs run one after the other.
     * Each must finish before the next starts.
     * ---------------------------------------------------------------- */
    cublasHandle_t h_seq;
    CHECK_CUBLAS(cublasCreate(&h_seq));

    /* Warm up */
    cublasSgemm(h_seq, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                &alpha, d_A[0],n, d_B[0],n, &beta, d_C[0],n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 4; i++)
        cublasSgemm(h_seq, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                    &alpha, d_A[i],n, d_B[i],n, &beta, d_C[i],n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Sequential (1 handle, default stream): %.3f ms\n", ms);

    /* ----------------------------------------------------------------
     * Demo 2: Concurrent — 4 handles, each on its own stream
     *
     * Each handle is assigned a different CUDA stream.
     * The 4 GEMMs can overlap on the GPU.
     *
     * cublasSetStream(handle, stream) assigns the stream.
     * Every subsequent call on that handle uses that stream.
     * ---------------------------------------------------------------- */
    cudaStream_t streams[4];
    cublasHandle_t handles[4];

    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
        CHECK_CUBLAS(cublasCreate(&handles[i]));
        /* cublasSetStream(handle, stream)
         * All operations on this handle now go into this stream.
         */
        CHECK_CUBLAS(cublasSetStream(handles[i], streams[i]));
    }

    cudaEventRecord(start);
    for (int i = 0; i < 4; i++)
        cublasSgemm(handles[i], CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                    &alpha, d_A[i],n, d_B[i],n, &beta, d_C[i],n);

    /* Wait for all streams to finish */
    for (int i = 0; i < 4; i++)
        cudaStreamSynchronize(streams[i]);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Concurrent (4 handles, 4 streams):     %.3f ms\n", ms);

    printf("\nConcurrent is faster when the GPU has spare compute units\n");
    printf("to fill while one stream is stalled on memory access.\n");

    /* Cleanup */
    cublasDestroy(h_seq);
    for (int i = 0; i < 4; i++) {
        cublasDestroy(handles[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(d_A[i]); cudaFree(d_B[i]); cudaFree(d_C[i]);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

/*
 * EXPECTED OUTPUT (approximate):
 * --------------------------------
 * Sequential (1 handle, default stream): ~4.0 ms
 * Concurrent (4 handles, 4 streams):     ~1.5 ms
 *
 * KEY CONCEPTS:
 * cublasCreate  → creates a handle (default stream)
 * cublasSetStream(handle, stream) → redirects all ops on that handle
 * cudaStreamSynchronize(stream)   → wait for one stream to finish
 * cudaDeviceSynchronize()         → wait for ALL streams to finish
 *
 * One handle = one stream. For concurrent execution,
 * create one handle per concurrent operation.
 */
