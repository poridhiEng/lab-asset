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

void print_matrix(const char *label, float *A, int n) {
    printf("  %s (%dx%d):\n", label, n, n);
    for (int i = 0; i < n; i++) {
        printf("    [");
        for (int j = 0; j < n; j++)
            printf(" %7.3f", A[j * n + i]);
        printf(" ]\n");
    }
}

void print_vector(const char *label, float *v, int n) {
    printf("  %-22s: [", label);
    for (int i = 0; i < n; i++) printf(" %6.2f", v[i]);
    printf(" ]\n");
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 3;

    float h_x[] = { 1.0f, 2.0f, 3.0f };
    float h_y[] = { 4.0f, 5.0f, 6.0f };

    printf("=== Demo 1: A = xyᵀ + yxᵀ (from zero) ===\n");

    float h_A[9] = { 0.0f };

    print_vector("x", h_x, n);
    print_vector("y", h_y, n);

    float *d_x, *d_y, *d_A;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_A, n * n * sizeof(float));

    cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1);
    cublasSetVector(n, sizeof(float), h_y, 1, d_y, 1);
    cublasSetMatrix(n, n, sizeof(float), h_A, n, d_A, n);

    float alpha = 1.0f;

    CHECK_CUBLAS(cublasSsyr2(handle, CUBLAS_FILL_MODE_UPPER,
                             n, &alpha,
                             d_x, 1,
                             d_y, 1,
                             d_A, n));

    cublasGetMatrix(n, n, sizeof(float), d_A, n, h_A, n);
    print_matrix("A = xyᵀ + yxᵀ (upper stored)", h_A, n);

    printf("\n=== Demo 2: Verify symmetry A[i][j] == A[j][i] ===\n");

    int symmetric = 1;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            float upper = h_A[j * n + i];
            float lower = h_A[i * n + j];
            printf("  A[%d][%d]=%.3f  vs  A[%d][%d]=%.3f  %s\n",
                   i, j, upper, j, i, lower,
                   (fabsf(upper - lower) < 1e-5f) ? "MATCH" : "MISMATCH");
            if (fabsf(upper - lower) > 1e-5f) symmetric = 0;
        }
    }
    printf("  Result is %s\n", symmetric ? "symmetric." : "NOT symmetric.");

    printf("\n=== Demo 3: syr2(x,x) == 2 * syr(x) ===\n");

    float h_A_syr2[9]  = { 0.0f };
    float h_A_syr_x2[9] = { 0.0f };

    float *d_A2, *d_A3;
    cudaMalloc((void**)&d_A2, n * n * sizeof(float));
    cudaMalloc((void**)&d_A3, n * n * sizeof(float));

    cublasSetMatrix(n, n, sizeof(float), h_A_syr2,   n, d_A2, n);
    cublasSetMatrix(n, n, sizeof(float), h_A_syr_x2, n, d_A3, n);

    CHECK_CUBLAS(cublasSsyr2(handle, CUBLAS_FILL_MODE_UPPER,
                             n, &alpha,
                             d_x, 1,
                             d_x, 1,
                             d_A2, n));

    float alpha2 = 2.0f;
    CHECK_CUBLAS(cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER,
                            n, &alpha2, d_x, 1, d_A3, n));

    cublasGetMatrix(n, n, sizeof(float), d_A2, n, h_A_syr2,   n);
    cublasGetMatrix(n, n, sizeof(float), d_A3, n, h_A_syr_x2, n);

    print_matrix("syr2(x, x)", h_A_syr2, n);
    print_matrix("2 * syr(x)", h_A_syr_x2, n);

    int match = 1;
    for (int k = 0; k < n*n; k++)
        if (fabsf(h_A_syr2[k] - h_A_syr_x2[k]) > 1e-4f) match = 0;
    printf("  Results are %s\n", match ? "identical." : "different.");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_A);
    cudaFree(d_A2); cudaFree(d_A3);
    cublasDestroy(handle);
    return 0;
}
