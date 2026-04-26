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

void print_matrix(const char *label, float *A, int n) {
    printf("  %s (%dx%d):\n", label, n, n);
    for (int i = 0; i < n; i++) {
        printf("    [");
        for (int j = 0; j < n; j++)
            printf(" %6.2f", A[j * n + i]);
        printf(" ]\n");
    }
}

void print_vector(const char *label, float *v, int n) {
    printf("  %-24s: [", label);
    for (int i = 0; i < n; i++) printf(" %6.2f", v[i]);
    printf(" ]\n");
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 3;

    printf("=== Demo 1: A = xxᵀ (start from zero) ===\n");

    float h_x[] = { 1.0f, 2.0f, 3.0f };
    float h_A[9] = { 0.0f };

    print_vector("x", h_x, n);

    float *d_x, *d_A;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_A, n * n * sizeof(float));

    cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1);
    cublasSetMatrix(n, n, sizeof(float), h_A, n, d_A, n);

    float alpha = 1.0f;

    CHECK_CUBLAS(cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER,
                            n, &alpha, d_x, 1, d_A, n));

    cublasGetMatrix(n, n, sizeof(float), d_A, n, h_A, n);
    print_matrix("A after syr (upper triangle updated)", h_A, n);
    printf("  Lower triangle in memory = 0 (not written by syr).\n");
    printf("  When reading A as symmetric, infer lower from upper.\n");

    printf("\n=== Demo 2: Accumulated updates (covariance-style) ===\n");

    float h_cov[9] = { 0.0f };
    float *d_cov;
    cudaMalloc((void**)&d_cov, n * n * sizeof(float));
    cublasSetMatrix(n, n, sizeof(float), h_cov, n, d_cov, n);

    float data[3][3] = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 1.0f, 1.0f, 0.0f }
    };

    for (int k = 0; k < 3; k++) {
        float *d_dk;
        cudaMalloc((void**)&d_dk, n * sizeof(float));
        cublasSetVector(n, sizeof(float), data[k], 1, d_dk, 1);

        CHECK_CUBLAS(cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER,
                                n, &alpha, d_dk, 1, d_cov, n));
        cudaFree(d_dk);
    }

    cublasGetMatrix(n, n, sizeof(float), d_cov, n, h_cov, n);
    printf("  Data points: [1,0,0], [0,1,0], [1,1,0]\n");
    print_matrix("Accumulated A (upper triangle)", h_cov, n);

    printf("\n=== Demo 3: syr (upper only) vs ger (full matrix) ===\n");

    float h_A_syr[9] = {0}, h_A_ger[9] = {0};
    float *d_A_syr, *d_A_ger, *d_y_ger;

    cudaMalloc((void**)&d_A_syr, n * n * sizeof(float));
    cudaMalloc((void**)&d_A_ger, n * n * sizeof(float));
    cudaMalloc((void**)&d_y_ger, n * sizeof(float));

    cublasSetMatrix(n, n, sizeof(float), h_A_syr, n, d_A_syr, n);
    cublasSetMatrix(n, n, sizeof(float), h_A_ger, n, d_A_ger, n);
    cublasSetVector(n, sizeof(float), h_x, 1, d_y_ger, 1);

    CHECK_CUBLAS(cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER,
                            n, &alpha, d_x, 1, d_A_syr, n));

    CHECK_CUBLAS(cublasSger(handle, n, n,
                            &alpha, d_x, 1, d_y_ger, 1, d_A_ger, n));

    cublasGetMatrix(n, n, sizeof(float), d_A_syr, n, h_A_syr, n);
    cublasGetMatrix(n, n, sizeof(float), d_A_ger, n, h_A_ger, n);

    print_matrix("syr result (upper filled, lower=0 in memory)", h_A_syr, n);
    print_matrix("ger result (full matrix written)", h_A_ger, n);

    cudaFree(d_x); cudaFree(d_A); cudaFree(d_cov);
    cudaFree(d_A_syr); cudaFree(d_A_ger); cudaFree(d_y_ger);
    cublasDestroy(handle);
    return 0;
}
