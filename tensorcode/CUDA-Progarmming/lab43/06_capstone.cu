/*
 * Capstone: Streams + Mixed Precision Pipeline
 *
 * Combines everything from this lab:
 *   - FP16 GEMMs for compute efficiency
 *   - Multiple streams for concurrency
 *   - Stream events for dependency management
 *
 * Scenario: two independent FP16 matrix products running concurrently
 * in separate streams, results accumulated in FP32, final result
 * converted back to FP16.
 *
 *   Stream A: C_A = A1 * A2   (FP16 in, FP32 accum, FP16 out)
 *   Stream B: C_B = B1 * B2   (FP16 in, FP32 accum, FP16 out)
 *   Stream C: waits for A and B, then Final = C_A + C_B  (FP32)
 *
 * Compile: nvcc -o capstone 06_capstone.cu -lcublas -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_fp16.h>
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
    size_t nf = (size_t)n * n;
    printf("=== Streams + Mixed Precision Pipeline ===\n");
    printf("  Matrix size: %dx%d  |  FP16 compute + FP32 accumulation\n\n", n, n);

    /* Allocate and fill host matrices */
    __half *h_A1 = (__half*)malloc(nf*sizeof(__half));
    __half *h_A2 = (__half*)malloc(nf*sizeof(__half));
    __half *h_B1 = (__half*)malloc(nf*sizeof(__half));
    __half *h_B2 = (__half*)malloc(nf*sizeof(__half));
    for (size_t i=0;i<nf;i++) {
        h_A1[i]=__float2half((float)(i%5+1)*0.1f);
        h_A2[i]=__float2half((float)(i%3+1)*0.2f);
        h_B1[i]=__float2half((float)(i%7+1)*0.1f);
        h_B2[i]=__float2half((float)(i%4+1)*0.2f);
    }

    /* Device allocations — FP16 inputs and outputs, FP32 final */
    __half *d_A1,*d_A2,*d_B1,*d_B2,*d_CA,*d_CB;
    float  *d_Final;
    cudaMalloc((void**)&d_A1,    nf*sizeof(__half));
    cudaMalloc((void**)&d_A2,    nf*sizeof(__half));
    cudaMalloc((void**)&d_B1,    nf*sizeof(__half));
    cudaMalloc((void**)&d_B2,    nf*sizeof(__half));
    cudaMalloc((void**)&d_CA,    nf*sizeof(__half));
    cudaMalloc((void**)&d_CB,    nf*sizeof(__half));
    cudaMalloc((void**)&d_Final, nf*sizeof(float));

    cudaMemcpy(d_A1, h_A1, nf*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A2, h_A2, nf*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B1, nf*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B2, nf*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_CA,    0, nf*sizeof(__half));
    cudaMemset(d_CB,    0, nf*sizeof(__half));
    cudaMemset(d_Final, 0, nf*sizeof(float));

    /* 3 streams, 3 handles */
    cudaStream_t sA, sB, sC;
    cublasHandle_t hA, hB, hC;
    cudaStreamCreate(&sA); cudaStreamCreate(&sB); cudaStreamCreate(&sC);
    cublasCreate(&hA); cublasCreate(&hB); cublasCreate(&hC);
    cublasSetStream(hA, sA);
    cublasSetStream(hB, sB);
    cublasSetStream(hC, sC);

    /* Enable Tensor Cores on all handles */
    cublasSetMathMode(hA, CUBLAS_TENSOR_OP_MATH);
    cublasSetMathMode(hB, CUBLAS_TENSOR_OP_MATH);
    cublasSetMathMode(hC, CUBLAS_TENSOR_OP_MATH);

    /* Events for fork-join synchronization */
    cudaEvent_t evA, evB;
    cudaEventCreate(&evA);
    cudaEventCreate(&evB);

    float alpha=1.0f, beta=0.0f, beta1=1.0f;

    cudaEvent_t wall_start, wall_stop;
    cudaEventCreate(&wall_start); cudaEventCreate(&wall_stop);
    cudaEventRecord(wall_start, 0);

    /* ---- Fork: launch A and B streams concurrently ---- */
    printf("  [stream A] C_A = A1 * A2  (FP16 in, FP32 accum)\n");
    CHECK_CUBLAS(cublasGemmEx(hA,
                              CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                              &alpha,
                              d_A1, CUDA_R_16F, n,
                              d_A2, CUDA_R_16F, n,
                              &beta,
                              d_CA, CUDA_R_16F, n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));

    printf("  [stream B] C_B = B1 * B2  (FP16 in, FP32 accum)\n");
    CHECK_CUBLAS(cublasGemmEx(hB,
                              CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                              &alpha,
                              d_B1, CUDA_R_16F, n,
                              d_B2, CUDA_R_16F, n,
                              &beta,
                              d_CB, CUDA_R_16F, n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));

    /* Record events when each stream is done */
    cudaEventRecord(evA, sA);
    cudaEventRecord(evB, sB);

    /* ---- Join: stream C waits for both ---- */
    cudaStreamWaitEvent(sC, evA, 0);
    cudaStreamWaitEvent(sC, evB, 0);

    printf("  [stream C] Final = C_A + C_B  (FP32, after join)\n");

    /* Promote FP16 C_A to FP32 Final, then add C_B */
    /* First: Final = 1*C_A + 0*C_A  (copy C_A as FP32) */
    /* We use gemmEx to do C_A * I + 0 = C_A in FP32 output */
    /* Simpler: allocate FP32 intermediate, convert C_A and C_B */
    float *d_CA_f, *d_CB_f;
    cudaMalloc((void**)&d_CA_f, nf*sizeof(float));
    cudaMalloc((void**)&d_CB_f, nf*sizeof(float));
    cudaMemset(d_CA_f, 0, nf*sizeof(float));
    cudaMemset(d_CB_f, 0, nf*sizeof(float));

    /* Identity matrix for conversion trick */
    __half *d_I;
    cudaMalloc((void**)&d_I, n*n*sizeof(__half));
    cudaMemset(d_I, 0, n*n*sizeof(__half));
    __half one_h = __float2half(1.0f);
    for (int i=0;i<n;i++)
        cudaMemcpy(d_I+i*n+i, &one_h, sizeof(__half), cudaMemcpyHostToDevice);

    /* C_A_f = C_A * I  (FP16 in, FP32 out via gemmEx) */
    CHECK_CUBLAS(cublasGemmEx(hC, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                              &alpha, d_CA,CUDA_R_16F,n, d_I,CUDA_R_16F,n,
                              &beta, d_CA_f,CUDA_R_32F,n,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    /* C_B_f = C_B * I  (FP16 in, FP32 out) */
    CHECK_CUBLAS(cublasGemmEx(hC, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                              &alpha, d_CB,CUDA_R_16F,n, d_I,CUDA_R_16F,n,
                              &beta, d_CB_f,CUDA_R_32F,n,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    /* Final = C_A_f + C_B_f */
    CHECK_CUBLAS(cublasSgeam(hC, CUBLAS_OP_N, CUBLAS_OP_N, n,n,
                             &alpha, d_CA_f,n, &beta1, d_CB_f,n, d_Final,n));

    cudaStreamSynchronize(sC);
    cudaEventRecord(wall_stop, 0);
    cudaEventSynchronize(wall_stop);

    float wall_ms;
    cudaEventElapsedTime(&wall_ms, wall_start, wall_stop);

    /* Sample output */
    float h_final_corner[4];
    cudaMemcpy(&h_final_corner[0], d_Final,       sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_final_corner[1], d_Final+1,     sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_final_corner[2], d_Final+n,     sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_final_corner[3], d_Final+n+1,   sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n  Final[0][0] = %.4f\n", h_final_corner[0]);
    printf("  Final[0][1] = %.4f\n", h_final_corner[2]);
    printf("  Total wall time: %.3f ms\n\n", wall_ms);

    printf("  Pipeline summary:\n");
    printf("    FP16 storage   — 2 bytes/element vs 4 for FP32\n");
    printf("    FP32 accum     — numerical stability preserved\n");
    printf("    2 concurrent streams — parallel GEMMs\n");
    printf("    Event sync     — correct dependency ordering\n");

    free(h_A1); free(h_A2); free(h_B1); free(h_B2);
    cudaFree(d_A1); cudaFree(d_A2); cudaFree(d_B1); cudaFree(d_B2);
    cudaFree(d_CA); cudaFree(d_CB); cudaFree(d_Final);
    cudaFree(d_CA_f); cudaFree(d_CB_f); cudaFree(d_I);
    cublasDestroy(hA); cublasDestroy(hB); cublasDestroy(hC);
    cudaStreamDestroy(sA); cudaStreamDestroy(sB); cudaStreamDestroy(sC);
    cudaEventDestroy(evA); cudaEventDestroy(evB);
    cudaEventDestroy(wall_start); cudaEventDestroy(wall_stop);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === Streams + Mixed Precision Pipeline ===
 *   Matrix size: 512x512  |  FP16 compute + FP32 accumulation
 *
 *   [stream A] C_A = A1 * A2  (FP16 in, FP32 accum)
 *   [stream B] C_B = B1 * B2  (FP16 in, FP32 accum)
 *   [stream C] Final = C_A + C_B  (FP32, after join)
 *
 *   Final[0][0] = x.xxxx
 *   Final[0][1] = x.xxxx
 *   Total wall time: ~0.8 ms
 *
 *   Pipeline summary:
 *     FP16 storage   — 2 bytes/element vs 4 for FP32
 *     FP32 accum     — numerical stability preserved
 *     2 concurrent streams — parallel GEMMs
 *     Event sync     — correct dependency ordering
 */
