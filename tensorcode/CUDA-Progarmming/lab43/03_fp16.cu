/*
 * Half Precision (FP16) — cublasSgemmEx and cublasHgemm
 *
 * FP16 (half) uses 2 bytes per float instead of 4.
 * Tensor Cores on Volta+ GPUs compute FP16 GEMM at much higher throughput.
 *
 * Two APIs for FP16 GEMM:
 *
 *   cublasHgemm  — pure FP16: inputs FP16, output FP16, accumulation FP16
 *                  Fast but lower precision accumulation
 *
 *   cublasSgemmEx — mixed: inputs FP16, output FP16, accumulation FP32
 *                   Better numerical stability than cublasHgemm
 *                   The compute type (accumulation) is set separately
 *
 * Tensor Core activation:
 *   cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)
 *   — enables Tensor Cores when input/output types qualify
 *   — matrix dimensions should be multiples of 8 for best performance
 *
 * Compile: nvcc -o fp16 03_fp16.cu -lcublas
 */

#include <stdio.h>
#include <stdlib.h>
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

/* Convert float array to half on host */
void float_to_half(float *src, __half *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = __float2half(src[i]);
}

/* Convert half array to float on host */
void half_to_float(__half *src, float *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = __half2float(src[i]);
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int m = 4, n = 4, k = 4;

    float h_A_f[] = {
        1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16
    };
    float h_B_f[] = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1   /* identity */
    };

    /* Convert inputs to FP16 */
    __half h_A_h[16], h_B_h[16];
    float_to_half(h_A_f, h_A_h, m*k);
    float_to_half(h_B_f, h_B_h, k*n);

    /* Allocate FP16 device arrays */
    __half *d_A_h, *d_B_h, *d_C_h;
    float  *d_C_f;
    cudaMalloc((void**)&d_A_h, m*k*sizeof(__half));
    cudaMalloc((void**)&d_B_h, k*n*sizeof(__half));
    cudaMalloc((void**)&d_C_h, m*n*sizeof(__half));
    cudaMalloc((void**)&d_C_f, m*n*sizeof(float));
    cudaMemcpy(d_A_h, h_A_h, m*k*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_h, h_B_h, k*n*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C_h, 0, m*n*sizeof(__half));
    cudaMemset(d_C_f, 0, m*n*sizeof(float));

    /* ----------------------------------------------------------------
     * Demo 1: cublasHgemm — pure FP16
     *
     * All inputs, outputs, and accumulation are FP16.
     * alpha and beta are __half values.
     * ---------------------------------------------------------------- */
    printf("=== Demo 1: cublasHgemm (pure FP16) ===\n");

    __half alpha_h = __float2half(1.0f);
    __half beta_h  = __float2half(0.0f);

    /* cublasHgemm(handle,
     *   transa, transb,
     *   m, n, k,
     *   &alpha_h,          — __half scalar
     *   d_A_h, m,          — __half* matrix A
     *   d_B_h, k,          — __half* matrix B
     *   &beta_h,           — __half scalar
     *   d_C_h, m)          — __half* output C
     */
    CHECK_CUBLAS(cublasHgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha_h,
                             d_A_h, m,
                             d_B_h, k,
                             &beta_h,
                             d_C_h, m));

    __half h_C_h[16];
    cudaMemcpy(h_C_h, d_C_h, m*n*sizeof(__half), cudaMemcpyDeviceToHost);

    printf("  C = A * I  (A * identity = A):\n");
    for (int i = 0; i < m; i++) {
        printf("    [");
        for (int j = 0; j < n; j++)
            printf(" %5.1f", __half2float(h_C_h[j*m+i]));
        printf(" ]\n");
    }

    /* ----------------------------------------------------------------
     * Demo 2: cublasSgemmEx — mixed precision
     *         FP16 inputs, FP32 accumulation, FP16 output
     *
     * CUDA_R_16F = FP16 type identifier
     * CUDA_R_32F = FP32 type identifier
     * CUBLAS_COMPUTE_32F = use FP32 for accumulation
     *
     * This avoids FP16 overflow in the accumulator while
     * keeping storage in FP16 for memory bandwidth savings.
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 2: cublasSgemmEx (FP16 in, FP32 accumulate, FP16 out) ===\n");

    cudaMemset(d_C_h, 0, m*n*sizeof(__half));
    float alpha_f = 1.0f, beta_f = 0.0f;

    /* cublasSgemmEx(handle,
     *   transa, transb,
     *   m, n, k,
     *   &alpha_f,          — float scalar
     *   d_A_h, CUDA_R_16F, m,    — FP16 input A + type tag
     *   d_B_h, CUDA_R_16F, k,    — FP16 input B + type tag
     *   &beta_f,           — float scalar
     *   d_C_h, CUDA_R_16F, m,    — FP16 output C + type tag
     *   CUBLAS_COMPUTE_32F,       — accumulate in FP32
     *   CUBLAS_GEMM_DEFAULT)      — algorithm selection
     */
    CHECK_CUBLAS(cublasGemmEx(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, k,
                               &alpha_f,
                               d_A_h, CUDA_R_16F, m,
                               d_B_h, CUDA_R_16F, k,
                               &beta_f,
                               d_C_h, CUDA_R_16F, m,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT));

    cudaMemcpy(h_C_h, d_C_h, m*n*sizeof(__half), cudaMemcpyDeviceToHost);
    printf("  C = A * I  (same result, better accumulation precision):\n");
    for (int i = 0; i < m; i++) {
        printf("    [");
        for (int j = 0; j < n; j++)
            printf(" %5.1f", __half2float(h_C_h[j*m+i]));
        printf(" ]\n");
    }

    /* ----------------------------------------------------------------
     * Demo 3: Enable Tensor Cores
     *
     * CUBLAS_TENSOR_OP_MATH  — allow Tensor Core acceleration
     * Best results when m, n, k are multiples of 8 (or 16 for FP16)
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 3: Tensor Core mode ===\n");

    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    printf("  Math mode set to CUBLAS_TENSOR_OP_MATH\n");
    printf("  Tensor Cores activate automatically when:\n");
    printf("    - Input/output types are FP16\n");
    printf("    - m, n, k are multiples of 8\n");
    printf("    - Compute type is CUBLAS_COMPUTE_16F or CUBLAS_COMPUTE_32F\n");

    /* Reset to default math mode */
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    cudaFree(d_A_h); cudaFree(d_B_h);
    cudaFree(d_C_h); cudaFree(d_C_f);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === Demo 1: cublasHgemm (pure FP16) ===
 *   C = A * I:
 *     [  1.0   5.0   9.0  13.0 ]
 *     [  2.0   6.0  10.0  14.0 ]
 *     [  3.0   7.0  11.0  15.0 ]
 *     [  4.0   8.0  12.0  16.0 ]
 *
 * === Demo 2: cublasSgemmEx (FP16 in, FP32 accumulate, FP16 out) ===
 *   Same result — accumulation in FP32 avoids precision loss
 *
 * === Demo 3: Tensor Core mode ===
 *   Tensor Cores activate automatically when conditions are met
 *
 * PRECISION SUMMARY:
 * FP32 (float):  23-bit mantissa, ~7 decimal digits, 4 bytes
 * FP16 (half):   10-bit mantissa, ~3 decimal digits, 2 bytes
 * BF16:          7-bit mantissa,  ~2 decimal digits, 2 bytes (better range)
 *
 * For training: FP16 compute + FP32 master weights (mixed precision training)
 * For inference: FP16 throughout (2x memory, ~2-8x Tensor Core throughput)
 */
