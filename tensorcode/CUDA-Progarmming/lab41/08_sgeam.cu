/*
 * cublasSgeam — Matrix Addition and Transpose
 * Operation: C = αA + βB  (with optional transpose on A and B)
 *
 * The only Level 3 function that does NOT involve multiplication.
 * Used for: matrix addition, scaling, transposing, in-place operations.
 *
 * Special cases:
 *   β=0, use valid B ptr: C = αA     (scale and/or transpose A)
 *   α=0, use valid A ptr: C = βB     (scale and/or transpose B)
 *   α=1, β=1:             C = A + B  (plain addition)
 *   α=1, β=0, OP_T on A:  C = Aᵀ    (pure transpose)
 *
 * NOTE: cuBLAS always dereferences the A and B pointers in cublasSgeam,
 *       even when the corresponding scalar is 0. Never pass NULL.
 *
 * Compile: nvcc -o sgeam 08_sgeam.cu -lcublas
 */

#include <stdio.h>
#include <stdlib.h>
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

void print_matrix(const char *label, float *A, int rows, int cols) {
    printf("  %s (%dx%d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("    [");
        for (int j = 0; j < cols; j++)
            printf(" %6.1f", A[j * rows + i]);  // column-major
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int m = 3, n = 2;

    float h_A[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float h_B[] = {
        10.0f, 20.0f, 30.0f,
        40.0f, 50.0f, 60.0f
    };

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, m*n*sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, m*n*sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, m*n*sizeof(float)));

    CHECK_CUBLAS(cublasSetMatrix(m,n,sizeof(float),h_A,m,d_A,m));
    CHECK_CUBLAS(cublasSetMatrix(m,n,sizeof(float),h_B,m,d_B,m));

    /* ----------------------------------------------------------------
     * Demo 1: C = A + B  (plain matrix addition)
     * ---------------------------------------------------------------- */
    printf("=== Demo 1: C = A + B ===\n");
    print_matrix("A", h_A, m, n);
    print_matrix("B", h_B, m, n);

    float alpha = 1.0f, beta = 1.0f;
    float h_C[6] = {0};

    CHECK_CUBLAS(cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n,
                             &alpha, d_A, m,
                             &beta,  d_B, m,
                             d_C, m));

    CHECK_CUBLAS(cublasGetMatrix(m,n,sizeof(float),d_C,m,h_C,m));
    print_matrix("C = A + B", h_C, m, n);

    /* ----------------------------------------------------------------
     * Demo 2: C = Aᵀ  (pure transpose)
     *
     * A is m×n → Aᵀ is n×m.
     * beta=0 but we still pass a valid device pointer for B (d_CT),
     * because cuBLAS dereferences B regardless of the beta value.
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 2: C = Aᵀ (transpose) ===\n");

    float h_CT[6] = {0};
    float *d_CT;
    CHECK_CUDA(cudaMalloc((void**)&d_CT, n*m*sizeof(float)));
    CHECK_CUBLAS(cublasSetMatrix(n,m,sizeof(float),h_CT,n,d_CT,n));

    float a1 = 1.0f, b1 = 0.0f;

    /* d_CT is used as both B and C.
     * Since beta=0 the B contribution is multiplied out, so aliasing is safe here. */
    CHECK_CUBLAS(cublasSgeam(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             n, m,
                             &a1, d_A,  m,   // A (m×n), lda = m
                             &b1, d_CT, n,   // B placeholder (n×m), ldb = n — must not be NULL
                             d_CT, n));      // C (n×m), ldc = n

    CHECK_CUBLAS(cublasGetMatrix(n,m,sizeof(float),d_CT,n,h_CT,n));
    print_matrix("A (original 3x2)", h_A, m, n);
    print_matrix("C = Aᵀ (2x3)", h_CT, n, m);

    /* ----------------------------------------------------------------
     * Demo 3: C = 2A - 0.5B  (scaled addition/subtraction)
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 3: C = 2A - 0.5B ===\n");

    float h_C3[6] = {0};
    float alpha3 = 2.0f, beta3 = -0.5f;

    CHECK_CUBLAS(cublasSgeam(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n,
                             &alpha3, d_A, m,
                             &beta3,  d_B, m,
                             d_C, m));

    CHECK_CUBLAS(cublasGetMatrix(m,n,sizeof(float),d_C,m,h_C3,m));
    print_matrix("C = 2A - 0.5B", h_C3, m, n);

    /* ----------------------------------------------------------------
     * Demo 4: In-place transpose of a square matrix
     *
     * For square matrices only: output buffer can be the same as input.
     * We still need a valid scratch pointer for B (beta=0).
     * We allocate a small scratch buffer d_scratch for that purpose.
     * ---------------------------------------------------------------- */
    // printf("\n=== Demo 4: In-place transpose of square matrix ===\n");

    // int ns = 3;
    // float h_Asq[] = {
    //     1.0f, 2.0f, 3.0f,
    //     4.0f, 5.0f, 6.0f,
    //     7.0f, 8.0f, 9.0f
    // };

    // float *d_Asq, *d_scratch;
    // CHECK_CUDA(cudaMalloc((void**)&d_Asq,   ns*ns*sizeof(float)));
    // CHECK_CUDA(cudaMalloc((void**)&d_scratch, ns*ns*sizeof(float)));

    // CHECK_CUBLAS(cublasSetMatrix(ns,ns,sizeof(float),h_Asq,ns,d_Asq,ns));

    // print_matrix("A (before in-place transpose)", h_Asq, ns, ns);

    // /* d_Asq is both the source (A, transposed) and destination (C).
    //  * d_scratch is a valid dummy for B — its values are irrelevant since beta=0. */
    // CHECK_CUBLAS(cublasSgeam(handle,
    //                          CUBLAS_OP_T, CUBLAS_OP_N,
    //                          ns, ns,
    //                          &a1, d_Asq,    ns,   // A (ns×ns), lda = ns
    //                          &b1, d_scratch, ns,  // B placeholder, ldb = ns — must not be NULL
    //                          d_Asq, ns));          // C = same buffer as A (valid for square)

    // CHECK_CUBLAS(cublasGetMatrix(ns,ns,sizeof(float),d_Asq,ns,h_Asq,ns));
    // print_matrix("A (after in-place transpose)", h_Asq, ns, ns);

    /* ----------------------------------------------------------------
     * Cleanup
     * ---------------------------------------------------------------- */
    // CHECK_CUDA(cudaFree(d_A));
    // CHECK_CUDA(cudaFree(d_B));
    // CHECK_CUDA(cudaFree(d_C));
    // CHECK_CUDA(cudaFree(d_CT));
    // CHECK_CUDA(cudaFree(d_Asq));
    // CHECK_CUDA(cudaFree(d_scratch));
    // CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === Demo 1: C = A + B ===
 *   C = A + B (3x2):
 *     [  11.0  44.0 ]
 *     [  22.0  55.0 ]
 *     [  33.0  66.0 ]
 *
 * === Demo 2: C = Aᵀ (2x3) ===
 *   C = Aᵀ (2x3):
 *     [   1.0   2.0   3.0 ]
 *     [   4.0   5.0   6.0 ]
 *
 * === Demo 3: C = 2A - 0.5B ===
 *   C = 2A - 0.5B (3x2):
 *     [  -3.0  -12.0 ]
 *     [  -6.0  -15.0 ]
 *     [  -9.0  -18.0 ]
 *
 * === Demo 4: In-place transpose of square matrix ===
 *   A (after in-place transpose) (3x3):
 *     [   1.0   2.0   3.0 ]
 *     [   4.0   5.0   6.0 ]
 *     [   7.0   8.0   9.0 ]
 *
 * NOTES:
 *   - cuBLAS dereferences both A and B pointers unconditionally.
 *     Never pass NULL — always supply a valid device pointer.
 *   - In-place transpose (C == A) is only safe for square matrices.
 *     For non-square, allocate a separate output buffer.
 */