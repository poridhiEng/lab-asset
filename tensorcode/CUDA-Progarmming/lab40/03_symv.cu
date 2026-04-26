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

void print_vector(const char *label, float *v, int n) {
    printf("  %-30s: [", label);
    for (int i = 0; i < n; i++) printf(" %6.2f", v[i]);
    printf(" ]\n");
}

void print_matrix(const char *label, float *A, int n) {
    printf("  %s (%dx%d):\n", label, n, n);
    for (int i = 0; i < n; i++) {
        printf("    [");
        for (int j = 0; j < n; j++)
            printf(" %6.2f", A[j * n + i]);
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 3;

    float h_A[] = {
        4.0f, 2.0f, 1.0f,
        2.0f, 5.0f, 3.0f,
        1.0f, 3.0f, 6.0f
    };
    float h_x[] = { 1.0f, 2.0f, 3.0f };

    printf("=== Demo 1: symv vs gemv on symmetric matrix ===\n");
    print_matrix("A (symmetric)", h_A, n);
    print_vector("x", h_x, n);

    float *d_A, *d_x, *d_y_symv, *d_y_gemv;
    cudaMalloc((void**)&d_A,      n * n * sizeof(float));
    cudaMalloc((void**)&d_x,      n * sizeof(float));
    cudaMalloc((void**)&d_y_symv, n * sizeof(float));
    cudaMalloc((void**)&d_y_gemv, n * sizeof(float));

    cublasSetMatrix(n, n, sizeof(float), h_A, n, d_A, n);
    cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1);

    float alpha = 1.0f, beta = 0.0f;
    float h_y_symv[3] = {0}, h_y_gemv[3] = {0};
    cublasSetVector(n, sizeof(float), h_y_symv, 1, d_y_symv, 1);
    cublasSetVector(n, sizeof(float), h_y_gemv, 1, d_y_gemv, 1);

    CHECK_CUBLAS(cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, n,
                             &alpha, d_A, n,
                             d_x, 1,
                             &beta, d_y_symv, 1));

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, n, n,
                             &alpha, d_A, n,
                             d_x, 1,
                             &beta, d_y_gemv, 1));

    cublasGetVector(n, sizeof(float), d_y_symv, 1, h_y_symv, 1);
    cublasGetVector(n, sizeof(float), d_y_gemv, 1, h_y_gemv, 1);

    print_vector("y via symv (UPPER)", h_y_symv, n);
    print_vector("y via gemv (reference)", h_y_gemv, n);

    printf("\n=== Demo 2: Corrupt lower triangle — symv should not notice ===\n");

    float h_A_corrupt[] = {
        4.0f,  999.0f, 999.0f,
        2.0f,  5.0f,   999.0f,
        1.0f,  3.0f,   6.0f
    };

    print_matrix("A (corrupted lower triangle)", h_A_corrupt, n);

    float *d_A_corrupt, *d_y_corrupt;
    cudaMalloc((void**)&d_A_corrupt, n * n * sizeof(float));
    cudaMalloc((void**)&d_y_corrupt, n * sizeof(float));

    float h_y_corrupt[3] = {0};
    cublasSetMatrix(n, n, sizeof(float), h_A_corrupt, n, d_A_corrupt, n);
    cublasSetVector(n, sizeof(float), h_y_corrupt, 1, d_y_corrupt, 1);

    CHECK_CUBLAS(cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, n,
                             &alpha, d_A_corrupt, n,
                             d_x, 1,
                             &beta, d_y_corrupt, 1));

    cublasGetVector(n, sizeof(float), d_y_corrupt, 1, h_y_corrupt, 1);
    print_vector("y (corrupted lower, UPPER mode)", h_y_corrupt, n);
    print_vector("y (clean matrix, UPPER mode)   ", h_y_symv, n);
    printf("  Results are identical — lower triangle was never read.\n");

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y_symv); cudaFree(d_y_gemv);
    cudaFree(d_A_corrupt); cudaFree(d_y_corrupt);
    cublasDestroy(handle);
    return 0;
}
