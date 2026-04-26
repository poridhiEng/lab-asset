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
    printf("  %-36s: [", label);
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

    float h_U[] = {
        2.0f, 0.0f, 0.0f,
        3.0f, 5.0f, 0.0f,
        4.0f, 6.0f, 7.0f
    };

    float h_L[] = {
        1.0f, 2.0f, 4.0f,
        0.0f, 3.0f, 5.0f,
        0.0f, 0.0f, 6.0f
    };

    float h_x_orig[] = { 1.0f, 2.0f, 3.0f };

    printf("=== Demo 1: Upper triangular  x = Ux ===\n");

    float h_x[3];
    h_x[0]=1; h_x[1]=2; h_x[2]=3;

    float *d_U, *d_x;
    cudaMalloc((void**)&d_U, n * n * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));

    cublasSetMatrix(n, n, sizeof(float), h_U, n, d_U, n);
    cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1);

    print_matrix("U (upper triangular)", h_U, n);
    print_vector("x (before)", h_x, n);

    CHECK_CUBLAS(cublasStrmv(handle,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, d_U, n, d_x, 1));

    cublasGetVector(n, sizeof(float), d_x, 1, h_x, 1);
    print_vector("x (after: Ux)", h_x, n);

    printf("\n=== Demo 2: Lower triangular  x = Lx ===\n");

    h_x[0]=1; h_x[1]=2; h_x[2]=3;

    float *d_L;
    cudaMalloc((void**)&d_L, n * n * sizeof(float));
    cublasSetMatrix(n, n, sizeof(float), h_L, n, d_L, n);
    cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1);

    print_matrix("L (lower triangular)", h_L, n);
    print_vector("x (before)", h_x, n);

    CHECK_CUBLAS(cublasStrmv(handle,
                             CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, d_L, n, d_x, 1));

    cublasGetVector(n, sizeof(float), d_x, 1, h_x, 1);
    print_vector("x (after: Lx)", h_x, n);

    printf("\n=== Demo 3: DIAG_UNIT ignores actual diagonal values ===\n");

    float h_U_diag[] = {
        99.0f, 0.0f, 0.0f,
         3.0f, 99.0f, 0.0f,
         4.0f,  6.0f, 99.0f
    };
    h_x[0]=1; h_x[1]=2; h_x[2]=3;

    float *d_U_diag;
    cudaMalloc((void**)&d_U_diag, n * n * sizeof(float));
    cublasSetMatrix(n, n, sizeof(float), h_U_diag, n, d_U_diag, n);
    cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1);

    print_matrix("U (diagonal=99, off-diag same as Demo1)", h_U_diag, n);
    print_vector("x (before)", h_x, n);

    CHECK_CUBLAS(cublasStrmv(handle,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_UNIT,
                             n, d_U_diag, n, d_x, 1));

    cublasGetVector(n, sizeof(float), d_x, 1, h_x, 1);
    print_vector("x after DIAG_UNIT (diagonal treated as 1)", h_x, n);
    printf("  99.0 on diagonal was completely ignored.\n");

    cudaFree(d_U); cudaFree(d_L); cudaFree(d_x); cudaFree(d_U_diag);
    cublasDestroy(handle);
    return 0;
}
