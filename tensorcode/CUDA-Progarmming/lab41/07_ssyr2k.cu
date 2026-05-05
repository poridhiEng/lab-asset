/*
 * cublasSsyr2k — Symmetric Rank-2K Update
 * Operation: C = α(ABᵀ + BAᵀ) + βC  or  C = α(AᵀB + BᵀA) + βC
 *
 * Takes TWO general matrices A and B.
 * The result is always symmetric by construction:
 *   ABᵀ + BAᵀ is symmetric because (ABᵀ + BAᵀ)ᵀ = BAᵀ + ABᵀ
 *
 * syrk vs syr2k:
 *   syrk:  C = αAAᵀ + βC    (one matrix, rank-k update)
 *   syr2k: C = α(ABᵀ+BAᵀ) + βC  (two matrices, rank-2k update)
 *
 * Compile: nvcc -o ssyr2k 07_ssyr2k.cu -lcublas
 */

#include <stdio.h>
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

void print_matrix(const char *label, float *A, int rows, int cols) {
    printf("  %s (%dx%d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("    [");
        for (int j = 0; j < cols; j++)
            printf(" %8.3f", A[j * rows + i]);
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 3, k = 2;

    /* A and B are 3×2 matrices */
    float h_A[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float h_B[] = {
        7.0f, 8.0f, 9.0f,
        1.0f, 2.0f, 3.0f
    };

    /* ----------------------------------------------------------------
     * Demo 1: Basic syr2k  C = ABᵀ + BAᵀ
     * ---------------------------------------------------------------- */
    printf("=== Demo 1: C = ABᵀ + BAᵀ ===\n");
    print_matrix("A (3x2)", h_A, n, k);
    print_matrix("B (3x2)", h_B, n, k);

    float h_C[9] = {0};
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n*k*sizeof(float));
    cudaMalloc((void**)&d_B, n*k*sizeof(float));
    cudaMalloc((void**)&d_C, n*n*sizeof(float));

    cublasSetMatrix(n,k,sizeof(float),h_A,n,d_A,n);
    cublasSetMatrix(n,k,sizeof(float),h_B,n,d_B,n);
    cublasSetMatrix(n,n,sizeof(float),h_C,n,d_C,n);

    float alpha = 1.0f, beta = 0.0f;

    /* cublasSsyr2k(handle,
     *              CUBLAS_FILL_MODE_UPPER,  — update upper triangle only
     *              CUBLAS_OP_N,             — OP_N: C = α(ABᵀ + BAᵀ) + βC
     *              n,                       — rows of A and B, dimension of C
     *              k,                       — cols of A and B
     *              &alpha,                  — scalar
     *              d_A, n,                  — matrix A, lda = n
     *              d_B, n,                  — matrix B, ldb = n
     *              &beta,                   — scalar for existing C
     *              d_C, n)                  — symmetric output C, ldc = n
     */
    CHECK_CUBLAS(cublasSsyr2k(handle,
                              CUBLAS_FILL_MODE_UPPER,
                              CUBLAS_OP_N,
                              n, k,
                              &alpha,
                              d_A, n,
                              d_B, n,
                              &beta,
                              d_C, n));

    cublasGetMatrix(n,n,sizeof(float),d_C,n,h_C,n);
    print_matrix("C = ABᵀ + BAᵀ (upper stored)", h_C, n, n);

    /* ----------------------------------------------------------------
     * Demo 2: Verify symmetry
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 2: Verify result is symmetric ===\n");

    int sym = 1;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            float upper = h_C[j * n + i];
            float lower = h_C[i * n + j];
            printf("  C[%d][%d] = %.3f  vs  C[%d][%d] = %.3f  %s\n",
                   i, j, upper, j, i, lower,
                   (fabsf(upper-lower) < 1e-4f) ? "MATCH" : "MISMATCH");
            if (fabsf(upper-lower) > 1e-4f) sym = 0;
        }
    }
    printf("  Upper triangle is %s\n", sym ? "symmetric." : "NOT symmetric.");

    /* ----------------------------------------------------------------
     * Demo 3: Special case — syr2k(A, A) = 2 * syrk(A)
     *
     * When A == B: ABᵀ + BAᵀ = AAᵀ + AAᵀ = 2AAᵀ
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 3: syr2k(A,A) == 2 * syrk(A) ===\n");

    float h_C_syr2k[9]={0}, h_C_syrk[9]={0};
    float *d_C2, *d_C3;
    cudaMalloc((void**)&d_C2, n*n*sizeof(float));
    cudaMalloc((void**)&d_C3, n*n*sizeof(float));

    cublasSetMatrix(n,n,sizeof(float),h_C_syr2k,n,d_C2,n);
    cublasSetMatrix(n,n,sizeof(float),h_C_syrk, n,d_C3,n);

    CHECK_CUBLAS(cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                              n,k, &alpha, d_A,n, d_A,n, &beta, d_C2,n));

    float alpha2 = 2.0f;
    CHECK_CUBLAS(cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                             n,k, &alpha2, d_A,n, &beta, d_C3,n));

    cublasGetMatrix(n,n,sizeof(float),d_C2,n,h_C_syr2k,n);
    cublasGetMatrix(n,n,sizeof(float),d_C3,n,h_C_syrk, n);

    print_matrix("syr2k(A, A)", h_C_syr2k, n, n);
    print_matrix("2 * syrk(A)", h_C_syrk,  n, n);

    int match = 1;
    for (int i=0;i<n*n;i++)
        if (fabsf(h_C_syr2k[i]-h_C_syrk[i])>1e-3f) match=0;
    printf("  Results are %s\n", match ? "identical." : "different.");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_C2); cudaFree(d_C3);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === Demo 1: C = ABᵀ + BAᵀ ===
 *   C (upper stored) — only upper triangle written, lower = 0 in memory.
 *   Upper triangle values reflect ABᵀ + BAᵀ.
 *
 * === Demo 2: Verify symmetry ===
 *   All pairs MATCH — result is symmetric.
 *
 * === Demo 3: syr2k(A,A) == 2*syrk(A) ===
 *   Results are identical.
 *
 * EXPLANATION:
 * C[i][j] = sum_f (A[i][f]*B[j][f] + B[i][f]*A[j][f])
 * This is inherently symmetric: C[i][j] = C[j][i]
 *
 * When B=A: C[i][j] = 2 * sum_f A[i][f]*A[j][f] = 2*(AAᵀ)[i][j]
 * So syr2k(A,A,alpha=1) = syrk(A,alpha=2)
 *
 * syr2k is used in Cholesky factorization updates and other
 * algorithms that need symmetric rank updates from two matrices.
 */
