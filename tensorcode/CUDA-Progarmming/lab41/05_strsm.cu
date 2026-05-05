/*
 * cublasStrsm — Triangular Solve with Multiple Right-Hand Sides
 * Operation: solve AX = αB  or  XA = αB  for X
 *
 * Solves a triangular system for a matrix X (not just a vector).
 * B is overwritten with the solution X.
 * Does NOT invert A — uses back/forward substitution column by column.
 *
 * Level 3 trsm vs Level 2 trsv:
 *   trsv: solves Ax = b  for a single vector x (Level 2)
 *   trsm: solves AX = B  for a matrix X — multiple right-hand sides (Level 3)
 *
 * Compile: nvcc -o strsm 05_strsm.cu -lcublas
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
            printf(" %8.4f", A[j * rows + i]);
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 3, nrhs = 2;

    /* Upper triangular A
     * A = [ 2  3  4 ]
     *     [ 0  5  6 ]
     *     [ 0  0  7 ]
     */
    float h_A[] = {
        2.0f, 0.0f, 0.0f,
        3.0f, 5.0f, 0.0f,
        4.0f, 6.0f, 7.0f
    };

    /* ----------------------------------------------------------------
     * Demo 1: Solve AX = B for X
     *
     * B columns are two different right-hand sides.
     * We know A from the trmm file, so we use trmm result as B
     * and expect to recover the original matrix.
     *
     * B = [ 20  56 ]   (= A * [[1,4],[2,5],[3,6]] from trmm demo)
     *     [ 28  80 ]
     *     [ 21  42 ]
     * Expected X = [ 1  4 ]
     *              [ 2  5 ]
     *              [ 3  6 ]
     * ---------------------------------------------------------------- */
    printf("=== Demo 1: Solve AX = B (two right-hand sides) ===\n");

    float h_B[] = {
        20.0f, 28.0f, 21.0f,
        56.0f, 80.0f, 42.0f
    };

    print_matrix("A (upper triangular)", h_A, n, n);
    print_matrix("B (right-hand side)", h_B, n, nrhs);
    printf("  Expected X: col0=[1,2,3] col1=[4,5,6]\n\n");

    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, n*n*sizeof(float));
    cudaMalloc((void**)&d_B, n*nrhs*sizeof(float));

    cublasSetMatrix(n,n,   sizeof(float), h_A, n, d_A, n);
    cublasSetMatrix(n,nrhs,sizeof(float), h_B, n, d_B, n);

    float alpha = 1.0f;

    /* cublasStrsm(handle,
     *             CUBLAS_SIDE_LEFT,          — solve AX = αB (A on left)
     *             CUBLAS_FILL_MODE_UPPER,    — upper triangular A
     *             CUBLAS_OP_N,               — no transpose
     *             CUBLAS_DIAG_NON_UNIT,      — use actual diagonal
     *             n,                         — rows of X and B
     *             nrhs,                      — cols of X and B
     *             &alpha,                    — scalar (scales B before solving)
     *             d_A, n,                    — triangular matrix A, lda = n
     *             d_B, n)                    — rhs B, overwritten with solution X
     */
    CHECK_CUBLAS(cublasStrsm(handle,
                             CUBLAS_SIDE_LEFT,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, nrhs,
                             &alpha,
                             d_A, n,
                             d_B, n));

    cublasGetMatrix(n,nrhs,sizeof(float),d_B,n,h_B,n);
    print_matrix("X = A⁻¹B (solution)", h_B, n, nrhs);

    /* ----------------------------------------------------------------
     * Demo 2: Round-trip verification — trmm then trsm
     *
     * Start with X_orig, apply trmm to get B = AX_orig,
     * apply trsm to recover X. Must match X_orig.
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 2: Round-trip trmm → trsm recovers X ===\n");

    float h_X_orig[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float h_X_work[6], h_C_trmm[6] = {0};
    for(int i=0;i<6;i++) h_X_work[i]=h_X_orig[i];

    float *d_X, *d_C_trmm;
    cudaMalloc((void**)&d_X,      n*nrhs*sizeof(float));
    cudaMalloc((void**)&d_C_trmm, n*nrhs*sizeof(float));

    cublasSetMatrix(n,nrhs,sizeof(float),h_X_work,n,d_X,n);
    cublasSetMatrix(n,nrhs,sizeof(float),h_C_trmm,n,d_C_trmm,n);

    print_matrix("X_orig", h_X_orig, n, nrhs);

    /* Step 1: B = AX via trmm */
    CHECK_CUBLAS(cublasStrmm(handle,
                             CUBLAS_SIDE_LEFT,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, nrhs,
                             &alpha,
                             d_A, n,
                             d_X, n,
                             d_C_trmm, n));

    cublasGetMatrix(n,nrhs,sizeof(float),d_C_trmm,n,h_C_trmm,n);
    print_matrix("B = AX (after trmm)", h_C_trmm, n, nrhs);

    /* Step 2: solve AX = B via trsm — should recover X_orig */
    CHECK_CUBLAS(cublasStrsm(handle,
                             CUBLAS_SIDE_LEFT,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, nrhs,
                             &alpha,
                             d_A, n,
                             d_C_trmm, n));

    cublasGetMatrix(n,nrhs,sizeof(float),d_C_trmm,n,h_C_trmm,n);
    print_matrix("X_recovered (after trsm)", h_C_trmm, n, nrhs);
    print_matrix("X_orig (should match)", h_X_orig, n, nrhs);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_X); cudaFree(d_C_trmm);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === Demo 1: Solve AX = B ===
 *   X = A⁻¹B (solution) (3x2):
 *     [   1.0000   4.0000 ]
 *     [   2.0000   5.0000 ]
 *     [   3.0000   6.0000 ]
 *
 * === Demo 2: Round-trip ===
 *   X_recovered matches X_orig exactly.
 *
 * EXPLANATION:
 * trsm solves AX = B column by column using back-substitution.
 * Each column of B is treated as a separate right-hand side vector.
 * This is equivalent to calling Level 2 trsv once per column,
 * but trsm does all columns in one optimised GPU call.
 *
 * The round-trip proves trmm and trsm are inverses, just as
 * trmv and trsv were inverses in Level 2.
 */
