/*
 * Naive GEMM Kernel vs cublasSgemm
 *
 * A custom CUDA kernel implements matrix multiply from scratch.
 * cublasSgemm is then run on the same data.
 * Both results and timings are compared.
 *
 * The naive kernel:
 *   - One thread computes one output element C[i][j]
 *   - No shared memory, no tiling, no vectorization
 *   - Every thread independently reads full rows/columns from DRAM
 *
 * cuBLAS:
 *   - Tiled shared memory loads
 *   - Vectorized memory access
 *   - Tensor Core acceleration
 *   - Auto-tuned algorithm selection
 *
 * Compile: nvcc -O3 -o naive_vs_cublas 04_naive_vs_cublas.cu -lcublas
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUBLAS(call) { \
    cublasStatus_t _e = (call); \
    if (_e != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %d at line %d\n", _e, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

/* ----------------------------------------------------------------
 * Naive GEMM kernel: C = A * B
 *
 * Row-major layout for the naive kernel (simpler to reason about).
 * A[i][k] = A[i*K + k]
 * B[k][j] = B[k*N + j]
 * C[i][j] = C[i*N + j]
 *
 * Each thread (row, col) computes one C element.
 * ---------------------------------------------------------------- */
__global__ void naive_gemm(float *A, float *B, float *C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = sum;
}

float run_naive(float *d_A, float *d_B, float *d_C,
                int M, int N, int K, float *ms_out) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    /* warmup */
    naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(ms_out, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return *ms_out;
}

float run_cublas(cublasHandle_t handle,
                 float *d_A_cm, float *d_B_cm, float *d_C_cm,
                 int M, int N, int K, float *ms_out) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    float alpha = 1.0f, beta = 0.0f;

    /* warmup */
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, &alpha, d_A_cm, M, d_B_cm, K, &beta, d_C_cm, M);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha, d_A_cm, M,
                             d_B_cm, K,
                             &beta, d_C_cm, M));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(ms_out, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return *ms_out;
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int M = 512, N = 512, K = 512;
    size_t sz = (size_t)M * N * sizeof(float);

    printf("Matrix size: %d by %d by %d\n\n", M, N, K);

    /* Host data */
    float *h_A = (float*)malloc(M*K*sizeof(float));
    float *h_B = (float*)malloc(K*N*sizeof(float));
    float *h_C_naive  = (float*)malloc(M*N*sizeof(float));
    float *h_C_cublas = (float*)malloc(M*N*sizeof(float));

    for (int i=0;i<M*K;i++) h_A[i]=(float)(i%8+1)*0.1f;
    for (int i=0;i<K*N;i++) h_B[i]=(float)(i%5+1)*0.1f;

    /* Device arrays for naive kernel (row-major) */
    float *d_A_rm, *d_B_rm, *d_C_naive;
    cudaMalloc((void**)&d_A_rm,   M*K*sizeof(float));
    cudaMalloc((void**)&d_B_rm,   K*N*sizeof(float));
    cudaMalloc((void**)&d_C_naive,M*N*sizeof(float));
    cudaMemcpy(d_A_rm,   h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_rm,   h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C_naive, 0, M*N*sizeof(float));

    /* Device arrays for cuBLAS (column-major — transpose row-major on upload) */
    float *d_A_cm, *d_B_cm, *d_C_cublas;
    cudaMalloc((void**)&d_A_cm,    M*K*sizeof(float));
    cudaMalloc((void**)&d_B_cm,    K*N*sizeof(float));
    cudaMalloc((void**)&d_C_cublas,M*N*sizeof(float));

    /* For cuBLAS col-major: transpose A and B on CPU */
    float *h_A_t = (float*)malloc(M*K*sizeof(float));
    float *h_B_t = (float*)malloc(K*N*sizeof(float));
    for (int i=0;i<M;i++) for (int k=0;k<K;k++) h_A_t[k*M+i]=h_A[i*K+k];
    for (int k=0;k<K;k++) for (int j=0;j<N;j++) h_B_t[j*K+k]=h_B[k*N+j];
    cudaMemcpy(d_A_cm, h_A_t, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_cm, h_B_t, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C_cublas, 0, M*N*sizeof(float));

    /* Run both */
    float ms_naive, ms_cublas;
    run_naive(d_A_rm, d_B_rm, d_C_naive, M, N, K, &ms_naive);
    run_cublas(handle, d_A_cm, d_B_cm, d_C_cublas, M, N, K, &ms_cublas);

    /* Compute GFLOPS: 2*M*N*K FLOPs per GEMM */
    double flops = 2.0 * M * N * K;
    double gflops_naive  = flops / (ms_naive  * 1e6);
    double gflops_cublas = flops / (ms_cublas * 1e6);

    printf("  %-20s  %8s  %10s\n", "Method", "Time(ms)", "GFLOPS");
    printf("  %-20s  %8s  %10s\n",
           "--------------------", "--------", "----------");
    printf("  %-20s  %8.3f  %10.2f\n", "Naive GEMM kernel", ms_naive,  gflops_naive);
    printf("  %-20s  %8.3f  %10.2f\n", "cublasSgemm",       ms_cublas, gflops_cublas);
    printf("\n  cublasSgemm is %.1fx faster than naive kernel.\n",
           ms_naive / ms_cublas);

    /* Verify results match */
    cudaMemcpy(h_C_naive,  d_C_naive,  M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cublas, d_C_cublas, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    /* Compare: naive is row-major, cublas is col-major */
    float max_diff = 0.0f;
    for (int i=0;i<M;i++)
        for (int j=0;j<N;j++) {
            float naive_val  = h_C_naive[i*N+j];
            float cublas_val = h_C_cublas[j*M+i];   /* col-major to row-major */
            float d = fabsf(naive_val - cublas_val);
            if (d > max_diff) max_diff = d;
        }
    printf("  Max diff between results: %.4f  %s\n\n", max_diff,
           max_diff < 0.1f ? "(correct)" : "(mismatch — check layout)");

    printf("  Why cuBLAS is faster:\n");
    printf("    Naive: each thread reads M+N floats from DRAM per output.\n");
    printf("    cuBLAS: tiles data into shared memory, reads each tile once.\n");
    printf("    cuBLAS: vectorized loads (float4), Tensor Core units.\n");
    printf("    cuBLAS: thousands of person-hours of optimization.\n");

    free(h_A); free(h_B); free(h_A_t); free(h_B_t);
    free(h_C_naive); free(h_C_cublas);
    cudaFree(d_A_rm); cudaFree(d_B_rm); cudaFree(d_C_naive);
    cudaFree(d_A_cm); cudaFree(d_B_cm); cudaFree(d_C_cublas);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT (approximate):
 * --------------------------------
 * Matrix size: 512 by 512 by 512
 *
 *   Method                Time(ms)      GFLOPS
 *   --------------------  --------  ----------
 *   Naive GEMM kernel       ~12.0        ~22
 *   cublasSgemm              ~0.3       ~900
 *
 *   cublasSgemm is ~40x faster than naive kernel.
 *   Max diff between results: ~0.0001 (correct)
 */
