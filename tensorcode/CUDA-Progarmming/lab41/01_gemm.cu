
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
            printf(" %7.3f", A[j * rows + i]);
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    printf("=== Demo 1: C = AB ===\n");

    int m = 2, n = 2, k = 3;

    float h_A[] = {
        1.0f, 4.0f,   /* col 0 */
        2.0f, 5.0f,   /* col 1 */
        3.0f, 6.0f    /* col 2 */
    };
    float h_B[] = {
         7.0f,  9.0f, 11.0f,   /* col 0 */
         8.0f, 10.0f, 12.0f    /* col 1 */
    };
    float h_C[4] = { 0.0f };

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    cublasSetMatrix(m, k, sizeof(float), h_A, m, d_A, m);
    cublasSetMatrix(k, n, sizeof(float), h_B, k, d_B, k);
    cublasSetMatrix(m, n, sizeof(float), h_C, m, d_C, m);

    float alpha = 1.0f, beta = 0.0f;

    /* cublasSgemm(handle,
     *             CUBLAS_OP_N, CUBLAS_OP_N,   — no transpose for A or B
     *             m,                          — rows of A and C
     *             n,                          — cols of B and C
     *             k,                          — cols of A = rows of B
     *             &alpha,                     — scalar for AB
     *             d_A, m,                     — matrix A, lda = rows of A
     *             d_B, k,                     — matrix B, ldb = rows of B
     *             &beta,                      — scalar for existing C
     *             d_C, m)                     — matrix C, ldc = rows of C
     */
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             d_A, m,
                             d_B, k,
                             &beta,
                             d_C, m));

    cublasGetMatrix(m, n, sizeof(float), d_C, m, h_C, m);
    print_matrix("A", h_A, m, k);
    print_matrix("B", h_B, k, n);
    print_matrix("C = AB", h_C, m, n);

    printf("\n=== Demo 2: C = 2AB + C ===\n");

    float h_C2[] = { 1.0f, 0.0f, 0.0f, 1.0f };   /* identity matrix */
    cublasSetMatrix(m, n, sizeof(float), h_C2, m, d_C, m);
    print_matrix("C (before)", h_C2, m, n);

    float alpha2 = 2.0f, beta2 = 1.0f;

    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha2,
                             d_A, m,
                             d_B, k,
                             &beta2,
                             d_C, m));

    cublasGetMatrix(m, n, sizeof(float), d_C, m, h_C2, m);
    print_matrix("C = 2AB + I", h_C2, m, n);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}
