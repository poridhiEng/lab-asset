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

#define CHECK_CUDA(call) { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

float time_gemm(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    int n = 1024;
    size_t nf = (size_t)n * n;

    printf("Matrix: %d by %d\n\n", n, n);
    printf("  %-24s  %8s  %6s  %10s\n", "Config", "Time(ms)", "Mem", "Max err");
    printf("  %-24s  %8s  %6s  %10s\n",
           "------------------------", "--------", "------", "----------");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    /* ---- allocate host data ---- */
    float  *h_Af = (float*)malloc(nf * sizeof(float));
    float  *h_Bf = (float*)malloc(nf * sizeof(float));
    float  *h_Cf = (float*)malloc(nf * sizeof(float));
    __half *h_Ah = (__half*)malloc(nf * sizeof(__half));
    __half *h_Bh = (__half*)malloc(nf * sizeof(__half));
    __half *h_Ch = (__half*)malloc(nf * sizeof(__half));

    for (size_t i = 0; i < nf; i++) {
        h_Af[i] = (float)((i % 8) + 1) * 0.1f;
        h_Bf[i] = (float)((i % 5) + 1) * 0.1f;
        h_Ah[i] = __float2half(h_Af[i]);
        h_Bh[i] = __float2half(h_Bf[i]);
    }

    /* ---- allocate device data ---- */
    float  *d_Af, *d_Bf, *d_Cf;
    __half *d_Ah, *d_Bh, *d_Ch;

    CHECK_CUDA(cudaMalloc((void**)&d_Af, nf * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_Bf, nf * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_Cf, nf * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_Ah, nf * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&d_Bh, nf * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&d_Ch, nf * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_Af, h_Af, nf*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Bf, h_Bf, nf*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Ah, h_Ah, nf*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Bh, h_Bh, nf*sizeof(__half), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    /* ================================================================
     * Run 1: FP32 in, FP32 accumulation
     * ================================================================ */
    cudaMemset(d_Cf, 0, nf * sizeof(float));

    /* warmup */
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                 &alpha, d_Af,CUDA_R_32F,n, d_Bf,CUDA_R_32F,n,
                 &beta,  d_Cf,CUDA_R_32F,n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEventRecord(start);
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                              &alpha, d_Af,CUDA_R_32F,n, d_Bf,CUDA_R_32F,n,
                              &beta,  d_Cf,CUDA_R_32F,n,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ms = time_gemm(start, stop);

    CHECK_CUDA(cudaMemcpy(h_Cf, d_Cf, nf*sizeof(float), cudaMemcpyDeviceToHost));

    size_t mem_fp32 = (3 * nf * sizeof(float)) / (1024 * 1024);
    printf("  %-24s  %8.3f  %4zuMB  %10s\n",
           "FP32 + FP32 accum", ms, mem_fp32, "reference");

    /* ================================================================
     * Run 2: FP16 in, FP32 accumulation
     * ================================================================ */
    cudaMemset(d_Ch, 0, nf * sizeof(__half));

    /* warmup */
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                 &alpha, d_Ah,CUDA_R_16F,n, d_Bh,CUDA_R_16F,n,
                 &beta,  d_Ch,CUDA_R_16F,n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEventRecord(start);
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                              &alpha, d_Ah,CUDA_R_16F,n, d_Bh,CUDA_R_16F,n,
                              &beta,  d_Ch,CUDA_R_16F,n,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ms = time_gemm(start, stop);

    CHECK_CUDA(cudaMemcpy(h_Ch, d_Ch, nf*sizeof(__half), cudaMemcpyDeviceToHost));

    float max_err_32 = 0.0f;
    for (size_t i = 0; i < nf; i++) {
        float d = fabsf(__half2float(h_Ch[i]) - h_Cf[i]);
        if (d > max_err_32) max_err_32 = d;
    }

    size_t mem_fp16 = (3 * nf * sizeof(__half)) / (1024 * 1024);
    printf("  %-24s  %8.3f  %4zuMB  %10.4f\n",
           "FP16 + FP32 accum", ms, mem_fp16, max_err_32);

    /* ================================================================
     * Run 3: FP16 in, FP16 accumulation
     * ================================================================ */
    cudaMemset(d_Ch, 0, nf * sizeof(__half));

    __half ah = __float2half(1.0f), bh = __float2half(0.0f);

    /* warmup */
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                &ah, d_Ah,n, d_Bh,n, &bh, d_Ch,n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEventRecord(start);
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                             &ah, d_Ah,n, d_Bh,n, &bh, d_Ch,n));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ms = time_gemm(start, stop);

    CHECK_CUDA(cudaMemcpy(h_Ch, d_Ch, nf*sizeof(__half), cudaMemcpyDeviceToHost));

    float max_err_16 = 0.0f;
    for (size_t i = 0; i < nf; i++) {
        float d = fabsf(__half2float(h_Ch[i]) - h_Cf[i]);
        if (d > max_err_16) max_err_16 = d;
    }

    printf("  %-24s  %8.3f  %4zuMB  %10.4f\n",
           "FP16 + FP16 accum", ms, mem_fp16, max_err_16);

    printf("\nFP16 with FP32 accumulation is preferred for training.\n");
    printf("Half the memory, faster compute, acceptable precision.\n");

    /* ---- cleanup ---- */
    free(h_Af); free(h_Bf); free(h_Cf);
    free(h_Ah); free(h_Bh); free(h_Ch);
    cudaFree(d_Af); cudaFree(d_Bf); cudaFree(d_Cf);
    cudaFree(d_Ah); cudaFree(d_Bh); cudaFree(d_Ch);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * Matrix: 1024 by 1024
 *
 *   Config                    Time(ms)     Mem     Max err
 *   ------------------------  --------  ------  ----------
 *   FP32 + FP32 accum            0.038    12MB   reference
 *   FP16 + FP32 accum            0.031     6MB    0.0625
 *   FP16 + FP16 accum            0.027     6MB    0.2729
 *
 * Compile: nvcc -O3 -o precision 05_precision.cu -lcublas -lm
 */