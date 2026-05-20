/*
 * Stream Synchronization and Dependencies
 *
 * Streams run independently by default.
 * When operation B depends on the result of operation A in a different
 * stream, you must insert a synchronization point.
 *
 * Two tools for cross-stream synchronization:
 *   cudaStreamWaitEvent(stream, event, 0)
 *     — stream waits until event is recorded
 *   cudaEventRecord(event, stream)
 *     — records a marker in a stream
 *
 * This file shows a fork-join pattern:
 *   Stream 0: compute A*B → result R0
 *   Stream 1: compute C*D → result R1
 *   Stream 2: waits for both, then computes R0 + R1
 *
 * Compile: nvcc -o sync 02_stream_sync.cu -lcublas
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

void print_corner(const char *label, float *d_M, int n) {
    float h[4];
    cudaMemcpy(&h[0], d_M,             sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h[1], d_M + 1,         sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h[2], d_M + n,         sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h[3], d_M + n + 1,     sizeof(float), cudaMemcpyDeviceToHost);
    printf("  %-14s top-left 2x2: [[%.1f, %.1f],[%.1f, %.1f]]\n",
           label, h[0], h[1], h[2], h[3]);
}

int main() {
    int n = 256;
    size_t sz = (size_t)n * n * sizeof(float);

    float *d_A, *d_B, *d_C, *d_D, *d_R0, *d_R1, *d_Final;
    cudaMalloc((void**)&d_A,     sz);
    cudaMalloc((void**)&d_B,     sz);
    cudaMalloc((void**)&d_C,     sz);
    cudaMalloc((void**)&d_D,     sz);
    cudaMalloc((void**)&d_R0,    sz);
    cudaMalloc((void**)&d_R1,    sz);
    cudaMalloc((void**)&d_Final, sz);

    /* A = 2*I, B = I, C = 3*I, D = I so R0=2I, R1=3I, Final=5I */
    cudaMemset(d_B, 0, sz); cudaMemset(d_D, 0, sz);
    float h_A = 2.0f, h_C = 3.0f;
    for (int i = 0; i < n; i++) {
        cudaMemcpy(d_A + i*n + i, &h_A, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B + i*n + i, (float[]){1.0f}, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C + i*n + i, &h_C, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_D + i*n + i, (float[]){1.0f}, sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemset(d_R0, 0, sz); cudaMemset(d_R1, 0, sz); cudaMemset(d_Final, 0, sz);

    /* Create 3 streams and 3 handles */
    cudaStream_t s0, s1, s2;
    cublasHandle_t h0, h1, h2;
    cudaStreamCreate(&s0); cudaStreamCreate(&s1); cudaStreamCreate(&s2);
    cublasCreate(&h0); cublasCreate(&h1); cublasCreate(&h2);
    cublasSetStream(h0, s0);
    cublasSetStream(h1, s1);
    cublasSetStream(h2, s2);

    /* Events to signal when R0 and R1 are ready */
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    float alpha=1.0f, beta=0.0f, beta1=1.0f;

    printf("=== Fork-Join Stream Pattern ===\n\n");
    printf("  Stream 0: R0 = A * B  (runs concurrently with stream 1)\n");
    printf("  Stream 1: R1 = C * D  (runs concurrently with stream 0)\n");
    printf("  Stream 2: waits for both, then Final = R0 + R1\n\n");

    /* Fork: launch R0 and R1 in parallel */
    cublasSgemm(h0, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                &alpha, d_A,n, d_B,n, &beta, d_R0,n);

    cublasSgemm(h1, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                &alpha, d_C,n, d_D,n, &beta, d_R1,n);

    /* Record events when each stream finishes its GEMM */
    cudaEventRecord(ev0, s0);
    cudaEventRecord(ev1, s1);

    /* Join: stream 2 must wait for both events before proceeding
     *
     * cudaStreamWaitEvent(stream, event, 0)
     *   — inserts a wait into 'stream'
     *   — stream will not start any new work until 'event' is recorded
     *   — does NOT block the CPU — only the GPU stream waits
     */
    cudaStreamWaitEvent(s2, ev0, 0);
    cudaStreamWaitEvent(s2, ev1, 0);

    /* Final = R0 + R1 via two geam calls in stream 2
     * First: Final = R0
     * Then:  Final = Final + R1
     */
    cublasSgeam(h2, CUBLAS_OP_N, CUBLAS_OP_N, n,n,
                &alpha, d_R0,n, &beta,  d_R0,n, d_Final,n);
    cublasSgeam(h2, CUBLAS_OP_N, CUBLAS_OP_N, n,n,
                &alpha, d_Final,n, &beta1, d_R1,n, d_Final,n);

    /* Wait for stream 2 to complete */
    cudaStreamSynchronize(s2);

    print_corner("R0 = A*B", d_R0, n);
    print_corner("R1 = C*D", d_R1, n);
    print_corner("Final = R0+R1", d_Final, n);
    printf("\n  Expected diagonal: R0=2, R1=3, Final=5\n");

    cublasDestroy(h0); cublasDestroy(h1); cublasDestroy(h2);
    cudaStreamDestroy(s0); cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaFree(d_R0); cudaFree(d_R1); cudaFree(d_Final);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === Fork-Join Stream Pattern ===
 *
 *   Stream 0: R0 = A * B  (runs concurrently with stream 1)
 *   Stream 1: R1 = C * D  (runs concurrently with stream 0)
 *   Stream 2: waits for both, then Final = R0 + R1
 *
 *   R0 = A*B       top-left 2x2: [[2.0, 0.0],[0.0, 2.0]]
 *   R1 = C*D       top-left 2x2: [[3.0, 0.0],[0.0, 3.0]]
 *   Final = R0+R1  top-left 2x2: [[5.0, 0.0],[0.0, 5.0]]
 *
 *   Expected diagonal: R0=2, R1=3, Final=5
 *
 * KEY CONCEPTS:
 * cudaEventRecord(event, stream)      — mark a point in a stream
 * cudaStreamWaitEvent(stream, event)  — stream waits for that mark
 * This is GPU-side sync — the CPU is never blocked during the wait.
 * Only the GPU stream pauses until the event fires.
 */
