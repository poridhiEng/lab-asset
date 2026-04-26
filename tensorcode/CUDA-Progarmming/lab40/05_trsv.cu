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
    printf("  %-34s: [", label);
    for (int i = 0; i < n; i++) printf(" %7.3f", v[i]);
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

    printf("=== Demo 1: Solve Ux = b ===\n");

    float h_b[] = { 20.0f, 28.0f, 21.0f };

    print_matrix("U", h_U, n);
    print_vector("b (right-hand side)", h_b, n);
    printf("  Expected solution x: [ 1.000  2.000  3.000 ]\n\n");

    float *d_U, *d_b;
    cudaMalloc((void**)&d_U, n * n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));

    cublasSetMatrix(n, n, sizeof(float), h_U, n, d_U, n);
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

    CHECK_CUBLAS(cublasStrsv(handle,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, d_U, n, d_b, 1));

    cublasGetVector(n, sizeof(float), d_b, 1, h_b, 1);
    print_vector("x (solution)", h_b, n);

    printf("\n=== Demo 2: Round-trip  x → trmv → b → trsv → x_recovered ===\n");

    float h_x_orig[] = { 3.0f, 1.0f, 4.0f };
    float h_x_work[3];
    h_x_work[0]=3; h_x_work[1]=1; h_x_work[2]=4;

    float *d_x;
    cudaMalloc((void**)&d_x, n * sizeof(float));

    print_vector("x (original)", h_x_orig, n);

    cublasSetVector(n, sizeof(float), h_x_work, 1, d_x, 1);

    CHECK_CUBLAS(cublasStrmv(handle,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, d_U, n, d_x, 1));

    cublasGetVector(n, sizeof(float), d_x, 1, h_x_work, 1);
    print_vector("b = Ux (after trmv)", h_x_work, n);

    CHECK_CUBLAS(cublasStrsv(handle,
                             CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, d_U, n, d_x, 1));

    cublasGetVector(n, sizeof(float), d_x, 1, h_x_work, 1);
    print_vector("x_recovered (after trsv)", h_x_work, n);
    print_vector("x (original — should match)", h_x_orig, n);

    printf("\n=== Demo 3: Lower triangular solve Lx = b ===\n");

    float h_L[] = {
        1.0f, 2.0f, 4.0f,
        0.0f, 3.0f, 5.0f,
        0.0f, 0.0f, 6.0f
    };
    float h_b2[] = { 1.0f, 8.0f, 32.0f };

    float *d_L, *d_b2;
    cudaMalloc((void**)&d_L,  n * n * sizeof(float));
    cudaMalloc((void**)&d_b2, n * sizeof(float));

    cublasSetMatrix(n, n, sizeof(float), h_L, n, d_L, n);
    cublasSetVector(n, sizeof(float), h_b2, 1, d_b2, 1);

    print_vector("b", h_b2, n);

    CHECK_CUBLAS(cublasStrsv(handle,
                             CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,
                             n, d_L, n, d_b2, 1));

    cublasGetVector(n, sizeof(float), d_b2, 1, h_b2, 1);
    print_vector("x (solution, expected [1,2,3])", h_b2, n);

    cudaFree(d_U); cudaFree(d_b); cudaFree(d_x);
    cudaFree(d_L); cudaFree(d_b2);
    cublasDestroy(handle);
    return 0;
}
