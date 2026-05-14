/*
 * Timing: sequential gemm vs gemmBatched vs gemmStridedBatched
 *
 * All three compute the same thing: 16 independent 64x64 multiplies.
 * This file shows how many kernel launches each approach uses
 * and how that affects wall-clock time.
 *
 * Compile: nvcc -o compare 03_compare.cu -lcublas
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
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int batch = 16, m = 64, n = 64, k = 64;
    long long stride = (long long)m * k;
    size_t total = batch * stride * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, total);
    cudaMalloc((void**)&d_B, total);
    cudaMalloc((void**)&d_C, total);
    cudaMemset(d_A, 1, total);
    cudaMemset(d_B, 1, total);
    cudaMemset(d_C, 0, total);

    /* Build pointer arrays for gemmBatched */
    float **h_Ap = (float**)malloc(batch*sizeof(float*));
    float **h_Bp = (float**)malloc(batch*sizeof(float*));
    float **h_Cp = (float**)malloc(batch*sizeof(float*));
    for (int i = 0; i < batch; i++) {
        h_Ap[i] = d_A + i*stride;
        h_Bp[i] = d_B + i*stride;
        h_Cp[i] = d_C + i*stride;
    }
    float **d_Ap, **d_Bp, **d_Cp;
    cudaMalloc((void**)&d_Ap, batch*sizeof(float*));
    cudaMalloc((void**)&d_Bp, batch*sizeof(float*));
    cudaMalloc((void**)&d_Cp, batch*sizeof(float*));
    cudaMemcpy(d_Ap, h_Ap, batch*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bp, h_Bp, batch*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cp, h_Cp, batch*sizeof(float*), cudaMemcpyHostToDevice);

    float alpha=1.0f, beta=0.0f;
    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    /* Warm up */
    for (int i = 0; i < batch; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,n,k,
                    &alpha, d_A+i*stride,m, d_B+i*stride,k,
                    &beta,  d_C+i*stride,m);
    cudaDeviceSynchronize();

    /* ---- Sequential gemm: batch separate calls ---- */
    cudaEventRecord(start);
    for (int i = 0; i < batch; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,n,k,
                    &alpha, d_A+i*stride,m,
                            d_B+i*stride,k,
                    &beta,  d_C+i*stride,m);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Sequential cublasSgemm        — %2d calls : %.3f ms\n", batch, ms);

    /* ---- gemmBatched: one call, pointer array ---- */
    cudaEventRecord(start);
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,n,k,
                       &alpha, (const float**)d_Ap,m,
                               (const float**)d_Bp,k,
                       &beta,  d_Cp,m, batch);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("cublasSgemmBatched             —  1 call  : %.3f ms\n", ms);

    /* ---- gemmStridedBatched: one call, stride ---- */
    cudaEventRecord(start);
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,n,k,
                              &alpha, d_A,m,stride,
                                      d_B,k,stride,
                              &beta,  d_C,m,stride, batch);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("cublasSgemmStridedBatched      —  1 call  : %.3f ms\n", ms);

    printf("\nAll three produce identical results.\n");
    printf("Batched calls avoid per-matrix kernel launch overhead.\n");

    free(h_Ap); free(h_Bp); free(h_Cp);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_Ap); cudaFree(d_Bp); cudaFree(d_Cp);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT (approximate, varies by GPU):
 * ----------------------------------------------
 * Sequential cublasSgemm        — 16 calls : ~2.5 ms
 * cublasSgemmBatched             —  1 call  : ~0.8 ms
 * cublasSgemmStridedBatched      —  1 call  : ~0.7 ms
 *
 * EXPLANATION:
 * Sequential: 16 kernel launches, each with CPU-GPU sync overhead.
 * Batched:    1 kernel launch — GPU handles all 16 internally.
 * Strided:    1 kernel launch — same as batched, slightly less overhead
 *             because matrix addresses are computed (not dereferenced).
 *
 * The difference grows with batch size.
 * At batch=128 the sequential vs batched gap is much larger.
 */
