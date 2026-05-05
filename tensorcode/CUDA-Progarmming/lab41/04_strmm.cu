
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

    int n = 3, nrhs = 2;

    /* Upper triangular A (3x3), column-major
     * A = [ 2  3  4 ]
     *     [ 0  5  6 ]
     *     [ 0  0  7 ]
     */
    float h_A[] = {
        2.0f, 0.0f, 0.0f,
        3.0f, 5.0f, 0.0f,
        4.0f, 6.0f, 7.0f
    };

    /* General B (3x2) */
    float h_B_orig[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    printf("=== Demo 1: B = AB (upper triangular A) ===\n");

    float h_B[6];
    for (int i=0;i<6;i++) h_B[i]=h_B_orig[i];

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n*n*sizeof(float));
    cudaMalloc((void**)&d_B, n*nrhs*sizeof(float));
    cudaMalloc((void**)&d_C, n*nrhs*sizeof(float));

    cublasSetMatrix(n,n,   sizeof(float), h_A, n, d_A, n);
    cublasSetMatrix(n,nrhs,sizeof(float), h_B, n, d_B, n);
    cublasSetMatrix(n,nrhs,sizeof(float), h_B, n, d_C, n);

    float alpha = 1.0f;

    print_matrix("A (upper triangular)", h_A, n, n);
    print_matrix("B (before)", h_B, n, nrhs);

    
    CHECK_CUBLAS(cublasStrmm(handle,
                             CUBLAS_SIDE_LEFT,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, nrhs,
                             &alpha,
                             d_A, n,
                             d_B, n,
                             d_C, n));

    float h_C[6] = {0};
    cublasGetMatrix(n,nrhs,sizeof(float),d_C,n,h_C,n);
    print_matrix("B = AB (result in C)", h_C, n, nrhs);

    
    printf("\n=== Demo 2: DIAG_UNIT — diagonal 99 is ignored, treated as 1 ===\n");

    float h_A_diag[] = {
        99.0f, 0.0f, 0.0f,
         3.0f, 99.0f, 0.0f,
         4.0f,  6.0f, 99.0f
    };
    float h_B2[6], h_C2[6] = {0};
    for (int i=0;i<6;i++) h_B2[i]=h_B_orig[i];

    float *d_A2, *d_B2, *d_C2;
    cudaMalloc((void**)&d_A2, n*n*sizeof(float));
    cudaMalloc((void**)&d_B2, n*nrhs*sizeof(float));
    cudaMalloc((void**)&d_C2, n*nrhs*sizeof(float));

    cublasSetMatrix(n,n,   sizeof(float),h_A_diag,n,d_A2,n);
    cublasSetMatrix(n,nrhs,sizeof(float),h_B2,    n,d_B2,n);
    cublasSetMatrix(n,nrhs,sizeof(float),h_C2,    n,d_C2,n);

    CHECK_CUBLAS(cublasStrmm(handle,
                             CUBLAS_SIDE_LEFT,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_UNIT,
                             n, nrhs,
                             &alpha,
                             d_A2, n,
                             d_B2, n,
                             d_C2, n));

    cublasGetMatrix(n,nrhs,sizeof(float),d_C2,n,h_C2,n);
    print_matrix("B = AB (DIAG_UNIT, diagonal=99 ignored)", h_C2, n, nrhs);
    printf("  Compare with Demo 1 — off-diagonal elements are same.\n");
    printf("  Diagonal contribution uses 1.0, not 99.0.\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === Demo 1: B = AB ===
 *   B = AB (result in C) (3x2):
 *     [  20.000  56.000 ]
 *     [  28.000  80.000 ]
 *     [  21.000  42.000 ]
 *
 * === Demo 2: DIAG_UNIT ===
 *   Off-diagonal elements same as Demo 1.
 *   Diagonal contribution uses 1.0 not 99.0.
 *
 * EXPLANATION:
 * Demo 1: C[0][0] = 2*1 + 3*2 + 4*3 = 2+6+12 = 20
 *         C[1][0] = 0*1 + 5*2 + 6*3 = 0+10+18 = 28
 *         C[2][0] = 0*1 + 0*2 + 7*3 = 21
 *
 * KEY DIFFERENCE from Level 2 trmv:
 * Level 3 trmm writes to a separate output buffer (d_C).
 * Level 2 trmv overwrites the input vector x in place.
 * Always provide a separate d_C when calling trmm.
 */
