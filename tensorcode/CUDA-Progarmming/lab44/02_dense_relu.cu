/*
 * Dense Layer with ReLU/GELU via cublasLt Epilogue
 *
 * cublasLt (cuBLAS lightweight) extends GEMM with epilogue operations
 * that fuse post-GEMM steps into the same kernel:
 *   - bias add
 *   - ReLU activation
 *   - GELU activation
 *   - dropout (select GPUs)
 *
 * Without epilogue: GEMM kernel → separate bias kernel → separate ReLU kernel
 * With epilogue:    one fused kernel (less memory traffic, faster)
 *
 * This file shows:
 *   1. Basic cublasLt GEMM (equivalent to cublasSgemm)
 *   2. GEMM + bias epilogue
 *   3. GEMM + bias + ReLU epilogue
 *
 * Compile: nvcc -o dense_relu 02_dense_relu.cu -lcublasLt -lcublas
 */

#include <stdio.h>
#include <stdlib.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUBLAS(call) { \
    cublasStatus_t _e = (call); \
    if (_e != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %d at line %d\n", _e, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void print_vector(const char *label, float *v, int n) {
    printf("  %-20s: [", label);
    for (int i = 0; i < n; i++) printf(" %6.3f", v[i]);
    printf(" ]\n");
}

/* Run a cublasLt GEMM with a given epilogue.
 * C = A * B + bias  (optionally with activation)
 *
 * All matrices column-major. A: m×k, B: k×n, C: m×n, bias: m
 */
void lt_gemm_with_epilogue(
    cublasLtHandle_t lt_handle,
    int m, int n, int k,
    float *d_A, float *d_B, float *d_C, float *d_bias,
    cublasLtEpilogue_t epilogue,
    const char *label)
{
    cublasLtMatmulDesc_t   op_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
    cublasLtMatmulPreference_t pref;

    cublasOperation_t trans_A = CUBLAS_OP_N;
    cublasOperation_t trans_B = CUBLAS_OP_N;

    /* Matrix multiply descriptor */
    cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &trans_A, sizeof(trans_A));
    cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &trans_B, sizeof(trans_B));
    cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

    /* Attach bias pointer when epilogue uses it */
    if (epilogue == CUBLASLT_EPILOGUE_BIAS ||
        epilogue == CUBLASLT_EPILOGUE_RELU_BIAS ||
        epilogue == CUBLASLT_EPILOGUE_GELU_BIAS) {
        cublasLtMatmulDescSetAttribute(op_desc,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias));
    }

    /* Matrix layout descriptors */
    cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_32F, m, k, m);
    cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_32F, k, n, k);
    cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32F, m, n, m);

    /* Algorithm selection */
    cublasLtMatmulPreferenceCreate(&pref);
    size_t workspace_size = 1 << 22;   /* 4MB workspace */
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size));

    int returned;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc,
        A_desc, B_desc, C_desc, C_desc,
        pref, 1, &heuristic, &returned);

    float *workspace = NULL;
    if (returned > 0) cudaMalloc((void**)&workspace, workspace_size);

    float alpha = 1.0f, beta = 0.0f;

    /* cublasLtMatmul(lt_handle,
     *   op_desc,             — operation descriptor (includes epilogue)
     *   &alpha,
     *   d_A, A_desc,         — matrix A with layout
     *   d_B, B_desc,         — matrix B with layout
     *   &beta,
     *   d_C, C_desc,         — input C (for beta scaling)
     *   d_C, C_desc,         — output C
     *   algo,                — selected algorithm
     *   workspace, ws_size,  — temporary workspace
     *   stream)              — CUDA stream (0 = default)
     */
    CHECK_CUBLAS(cublasLtMatmul(lt_handle, op_desc,
                                &alpha,
                                d_A, A_desc,
                                d_B, B_desc,
                                &beta,
                                d_C, C_desc,
                                d_C, C_desc,
                                returned > 0 ? &heuristic.algo : NULL,
                                workspace, returned > 0 ? workspace_size : 0,
                                0));

    cudaDeviceSynchronize();

    float *h_C = (float*)malloc(m * n * sizeof(float));
    cublasGetMatrix(m, n, sizeof(float), d_C, m, h_C, m);

    printf("  %s output (first sample):\n", label);
    print_vector("  Y[:,0]", h_C, m);

    free(h_C);
    if (workspace) cudaFree(workspace);
    cublasLtMatmulDescDestroy(op_desc);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(C_desc);
    cublasLtMatmulPreferenceDestroy(pref);
}

int main() {
    cublasLtHandle_t lt_handle;
    CHECK_CUBLAS(cublasLtCreate(&lt_handle));

    int m = 4, n = 6, k = 3;   /* out_feat=4, batch=6, in_feat=3 */

    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float h_bias[] = { 0.5f, -0.5f, 0.2f, -0.2f };

    /* Fill W with small values */
    for (int i = 0; i < m*k; i++) h_A[i] = (float)(i % 5 + 1) * 0.1f;
    for (int i = 0; i < k*n; i++) h_B[i] = (float)(i % 4 + 1) * 0.2f - 0.5f;

    float *d_A, *d_B, *d_C, *d_bias;
    cudaMalloc((void**)&d_A,    m * k * sizeof(float));
    cudaMalloc((void**)&d_B,    k * n * sizeof(float));
    cudaMalloc((void**)&d_C,    m * n * sizeof(float));
    cudaMalloc((void**)&d_bias, m     * sizeof(float));

    cublasSetMatrix(m, k, sizeof(float), h_A, m, d_A, m);
    cublasSetMatrix(k, n, sizeof(float), h_B, k, d_B, k);
    cublasSetVector(m, sizeof(float), h_bias, 1, d_bias, 1);

    printf("=== cublasLt Epilogue Operations ===\n");
    printf("  Matrix: %d×%d × %d×%d  (out=%d, batch=%d, in=%d)\n\n", m,k, k,n, m,n,k);

    /* 1. Plain GEMM (no epilogue) */
    cudaMemset(d_C, 0, m*n*sizeof(float));
    lt_gemm_with_epilogue(lt_handle, m, n, k, d_A, d_B, d_C, NULL,
                          CUBLASLT_EPILOGUE_DEFAULT, "GEMM only");

    /* 2. GEMM + bias */
    cudaMemset(d_C, 0, m*n*sizeof(float));
    lt_gemm_with_epilogue(lt_handle, m, n, k, d_A, d_B, d_C, d_bias,
                          CUBLASLT_EPILOGUE_BIAS, "GEMM + bias");

    /* 3. GEMM + bias + ReLU */
    cudaMemset(d_C, 0, m*n*sizeof(float));
    lt_gemm_with_epilogue(lt_handle, m, n, k, d_A, d_B, d_C, d_bias,
                          CUBLASLT_EPILOGUE_RELU_BIAS, "GEMM + bias + ReLU");

    /* 4. GEMM + bias + GELU */
    cudaMemset(d_C, 0, m*n*sizeof(float));
    lt_gemm_with_epilogue(lt_handle, m, n, k, d_A, d_B, d_C, d_bias,
                          CUBLASLT_EPILOGUE_GELU_BIAS, "GEMM + bias + GELU");

    printf("\n  ReLU zeroes all negative values.\n");
    printf("  GELU applies smooth gating (non-zero for negative inputs).\n");
    printf("  Both are fused into the GEMM kernel — no separate kernel launch.\n");

    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_bias);
    cublasLtDestroy(lt_handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * GEMM only output:        some values, some negative
 * GEMM + bias output:      same + bias per row
 * GEMM + bias + ReLU:      negative values replaced with 0
 * GEMM + bias + GELU:      negative values smoothly gated near 0
 *
 * KEY CONCEPT:
 * cublasLt epilogue fuses post-GEMM ops into the same kernel.
 * Without epilogue: 3 kernel launches (GEMM, bias, activation).
 * With epilogue:    1 kernel launch — same result, less overhead.
 * The output Y is never written to memory between steps.
 */
