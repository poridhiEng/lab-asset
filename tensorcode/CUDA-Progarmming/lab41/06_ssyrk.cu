/*
 * cublasSsyrk — Symmetric Rank-K Update
 * Operation: C = αAAᵀ + βC  or  C = αAᵀA + βC
 *
 * C must be symmetric. Only one triangle of C is updated.
 * A is a general (m×k) matrix.
 *
 * trans parameter:
 *   CUBLAS_OP_N: C = αAAᵀ + βC  (A is n×k, C is n×n)
 *   CUBLAS_OP_T: C = αAᵀA + βC  (A is k×n, C is n×n)
 *
 * Real-world use:
 *   Computing XᵀX is the first step in least-squares regression.
 *   Computing XXᵀ builds a covariance/Gram matrix from data rows.
 *
 * Compile: nvcc -o ssyrk 06_ssyrk.cu -lcublas
 */

#include <stdio.h>
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
            printf(" %7.2f", A[j * rows + i]);
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    /* A is a 3×2 data matrix (3 samples, 2 features), column-major
     * A = [ 1  2 ]
     *     [ 3  4 ]
     *     [ 5  6 ]
     */
    int n = 3, k = 2;

    float h_A[] = {
        1.0f, 3.0f, 5.0f,   /* col 0 */
        2.0f, 4.0f, 6.0f    /* col 1 */
    };

    /* ----------------------------------------------------------------
     * Demo 1: C = AAᵀ  (Gram matrix / covariance-style)
     *
     * A is n×k, AAᵀ is n×n = 3×3
     * C[i][j] = dot product of row i and row j of A
     * ---------------------------------------------------------------- */
    printf("=== Demo 1: C = AAᵀ (OP_N, Gram matrix) ===\n");
    print_matrix("A (3x2, data matrix)", h_A, n, k);

    float h_C[9] = {0};
    float *d_A, *d_C;
    cudaMalloc((void**)&d_A, n*k*sizeof(float));
    cudaMalloc((void**)&d_C, n*n*sizeof(float));

    cublasSetMatrix(n, k, sizeof(float), h_A, n, d_A, n);
    cublasSetMatrix(n, n, sizeof(float), h_C, n, d_C, n);

    float alpha = 1.0f, beta = 0.0f;

    /* cublasSsyrk(handle,
     *             CUBLAS_FILL_MODE_UPPER,  — update upper triangle of C
     *             CUBLAS_OP_N,             — C = αAAᵀ + βC
     *             n,                       — rows of A and dimension of C
     *             k,                       — cols of A (rank of update)
     *             &alpha,                  — scalar for AAᵀ
     *             d_A, n,                  — matrix A, lda = n
     *             &beta,                   — scalar for existing C
     *             d_C, n)                  — symmetric output C, ldc = n
     */
    CHECK_CUBLAS(cublasSsyrk(handle,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             n, k,
                             &alpha, d_A, n,
                             &beta,  d_C, n));

    cublasGetMatrix(n,n,sizeof(float),d_C,n,h_C,n);
    print_matrix("C = AAᵀ (upper triangle stored)", h_C, n, n);
    printf("  C[i][j] = row_i · row_j of A\n");
    printf("  C[0][0] = 1²+2² = 5  C[1][1] = 3²+4² = 25  C[2][2] = 5²+6² = 61\n");

    /* ----------------------------------------------------------------
     * Demo 2: C = AᵀA  (normal equations matrix for least squares)
     *
     * AᵀA is k×k = 2×2
     * This is the matrix you invert in normal equations: x = (AᵀA)⁻¹Aᵀb
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 2: C = AᵀA (OP_T, normal equations matrix) ===\n");

    float h_C2[4] = {0};
    float *d_C2;
    cudaMalloc((void**)&d_C2, k*k*sizeof(float));
    cublasSetMatrix(k,k,sizeof(float),h_C2,k,d_C2,k);

    /* OP_T: C = αAᵀA + βC, A is treated as k×n, C is k×k */
    CHECK_CUBLAS(cublasSsyrk(handle,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_T,
                             k, n,
                             &alpha, d_A, n,
                             &beta,  d_C2, k));

    cublasGetMatrix(k,k,sizeof(float),d_C2,k,h_C2,k);
    print_matrix("C = AᵀA (2x2, normal equations)", h_C2, k, k);
    printf("  This is the first step in computing least-squares: (AᵀA)x = Aᵀb\n");

    /* ----------------------------------------------------------------
     * Demo 3: Accumulated syrk — C = A₁A₁ᵀ + A₂A₂ᵀ
     *
     * Using beta=1 on second call accumulates into C.
     * This builds a covariance matrix from multiple data batches.
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 3: Accumulated C = A1*A1ᵀ + A2*A2ᵀ ===\n");

    float h_A2[] = {
        2.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 2.0f
    };
    float h_Cacc[9] = {0};
    float *d_A2, *d_Cacc;
    cudaMalloc((void**)&d_A2,   n*k*sizeof(float));
    cudaMalloc((void**)&d_Cacc, n*n*sizeof(float));

    cublasSetMatrix(n,k,sizeof(float),h_A2,  n,d_A2,  n);
    cublasSetMatrix(n,n,sizeof(float),h_Cacc,n,d_Cacc,n);

    /* First batch: C = A1*A1ᵀ */
    CHECK_CUBLAS(cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                             n,k, &alpha, d_A,n, &beta, d_Cacc,n));

    /* Second batch: C = C + A2*A2ᵀ (beta=1 preserves existing C) */
    float beta2 = 1.0f;
    CHECK_CUBLAS(cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                             n,k, &alpha, d_A2,n, &beta2, d_Cacc,n));

    cublasGetMatrix(n,n,sizeof(float),d_Cacc,n,h_Cacc,n);
    print_matrix("C = A1*A1ᵀ + A2*A2ᵀ", h_Cacc, n, n);

    cudaFree(d_A); cudaFree(d_C); cudaFree(d_C2);
    cudaFree(d_A2); cudaFree(d_Cacc);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === Demo 1: C = AAᵀ ===
 *   C = AAᵀ (upper triangle stored) (3x3):
 *     [   5.00  11.00  17.00 ]   ← upper filled
 *     [   0.00  25.00  39.00 ]   ← lower = 0 (not written)
 *     [   0.00   0.00  61.00 ]
 *
 * === Demo 2: C = AᵀA ===
 *   C = AᵀA (2x2) (normal equations):
 *     [  35.00  44.00 ]
 *     [   0.00  56.00 ]
 *
 * EXPLANATION:
 * Demo 1: AAᵀ[i][j] = sum_f A[i][f]*A[j][f] (dot product of rows i and j)
 *   [0][0] = 1*1 + 2*2 = 5
 *   [0][1] = 1*3 + 2*4 = 11
 *   [1][1] = 3*3 + 4*4 = 25
 *
 * Demo 2: AᵀA[i][j] = sum_s A[s][i]*A[s][j] (dot product of cols i and j)
 *   [0][0] = 1²+3²+5² = 35
 *   [0][1] = 1*2+3*4+5*6 = 44
 *   [1][1] = 2²+4²+6² = 56
 */
