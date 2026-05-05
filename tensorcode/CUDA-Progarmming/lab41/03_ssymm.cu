#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

    int n    = 16384;  /* A is n x n symmetric — 1 GB, far beyond any L2 cache */
    int nrhs = 2;      /* minimal RHS columns — maximally memory-bandwidth-bound */

    float *h_A = (float*)malloc(n * n    * sizeof(float));
    float *h_B = (float*)malloc(n * nrhs * sizeof(float));

    /* Fill A as a symmetric matrix */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float val = (i == j) ? (float)(n + i) : (float)((i * n + j) % 7 + 1);
            h_A[j * n + i] = val;
            h_A[i * n + j] = val;
        }
    }

    /* Fill B with arbitrary values */
    for (int i = 0; i < n * nrhs; i++)
        h_B[i] = (float)(i % 5 + 1);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * n    * sizeof(float));
    cudaMalloc((void**)&d_B, n * nrhs * sizeof(float));
    cudaMalloc((void**)&d_C, n * nrhs * sizeof(float));

    cublasSetMatrix(n, n,    sizeof(float), h_A, n, d_A, n);
    cublasSetMatrix(n, nrhs, sizeof(float), h_B, n, d_B, n);

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    /* Warm up */
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, nrhs, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);
    cudaDeviceSynchronize();

    /* Time cublasSgemm */
    cudaEventRecord(start);
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, nrhs, n,
                             &alpha, d_A, n,
                             d_B, n,
                             &beta, d_C, n));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("cublasSgemm  — A(%dx%d) x B(%dx%d): %.3f ms\n", n, n, n, nrhs, ms);

    /* Warm up */
    cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                n, nrhs, &alpha, d_A, n, d_B, n, &beta, d_C, n);
    cudaDeviceSynchronize();

    /* Time cublasSsymm */
    cudaEventRecord(start);
    CHECK_CUBLAS(cublasSsymm(handle,
                             CUBLAS_SIDE_LEFT,
                             CUBLAS_FILL_MODE_UPPER,
                             n, nrhs,
                             &alpha, d_A, n,
                             d_B, n,
                             &beta, d_C, n));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("cublasSsymm  — A(%dx%d) x B(%dx%d): %.3f ms\n", n, n, n, nrhs, ms);

    printf("\nssymm reads ~half the elements of A vs gemm.\n");
    printf("On large symmetric matrices the time difference becomes measurable.\n");

    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cublasDestroy(handle);
    return 0;
}