#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUBLAS(call) { 
    cublasStatus_t _e = (call); 
    if (_e != CUBLAS_STATUS_SUCCESS) { 
        printf("cuBLAS error %d at line %d\n", _e, __LINE__); \
        exit(EXIT_FAILURE); 
    } 
}

void print_vector(const char *label, float *v, int n) {
    printf("  %-24s: [", label);
    for (int i = 0; i < n; i++) printf(" %6.2f", v[i]);
    printf(" ]\n");
}

void print_matrix(const char *label, float *A, int rows, int cols) {
    printf("  %s (%dx%d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("    [");
        for (int j = 0; j < cols; j++)
            printf(" %6.2f", A[j * rows + i]);
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    printf("=== Demo 1: y = Ax ===\n");

    int m = 3, n = 3;

    float h_A[] = {
        1.0f, 4.0f, 7.0f,
        2.0f, 5.0f, 8.0f,
        3.0f, 6.0f, 9.0f
    };
    float h_x[] = { 1.0f, 1.0f, 1.0f };
    float h_y[] = { 0.0f, 0.0f, 0.0f };

    print_matrix("A", h_A, m, n);
    print_vector("x", h_x, n);

    float *d_A, *d_x, *d_y;
    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, m * sizeof(float));

    cublasSetMatrix(m, n, sizeof(float), h_A, m, d_A, m);
    cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1);
    cublasSetVector(m, sizeof(float), h_y, 1, d_y, 1);

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n,
                             &alpha, d_A, m,
                             d_x, 1,
                             &beta, d_y, 1));

    cublasGetVector(m, sizeof(float), d_y, 1, h_y, 1);
    print_vector("y = Ax", h_y, m);

    printf("\n=== Demo 2: y = 2*Ax + 3*y ===\n");

    float h_y2[] = { 1.0f, 1.0f, 1.0f };
    cublasSetVector(m, sizeof(float), h_y2, 1, d_y, 1);
    print_vector("y (before)", h_y2, m);

    float alpha2 = 2.0f, beta2 = 3.0f;

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n,
                             &alpha2, d_A, m,
                             d_x, 1,
                             &beta2, d_y, 1));

    cublasGetVector(m, sizeof(float), d_y, 1, h_y2, 1);
    print_vector("y = 2Ax + 3y", h_y2, m);

    printf("\n=== Demo 3: y = Aᵀx (transpose mode) ===\n");

    float h_B[] = {
        1.0f, 0.0f, 0.0f,
        2.0f, 1.0f, 0.0f,
        3.0f, 0.0f, 1.0f
    };
    float h_xb[] = { 1.0f, 2.0f, 3.0f };
    float h_yn[]  = { 0.0f, 0.0f, 0.0f };
    float h_yt[]  = { 0.0f, 0.0f, 0.0f };

    float *d_B, *d_xb, *d_yn, *d_yt;
    cudaMalloc((void**)&d_B,  m * n * sizeof(float));
    cudaMalloc((void**)&d_xb, n * sizeof(float));
    cudaMalloc((void**)&d_yn, m * sizeof(float));
    cudaMalloc((void**)&d_yt, m * sizeof(float));

    cublasSetMatrix(m, n, sizeof(float), h_B, m, d_B, m);
    cublasSetVector(n, sizeof(float), h_xb, 1, d_xb, 1);
    cublasSetVector(m, sizeof(float), h_yn, 1, d_yn, 1);
    cublasSetVector(m, sizeof(float), h_yt, 1, d_yt, 1);

    float a1 = 1.0f, b1 = 0.0f;

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n,
                             &a1, d_B, m, d_xb, 1, &b1, d_yn, 1));

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, m, n,
                             &a1, d_B, m, d_xb, 1, &b1, d_yt, 1));

    cublasGetVector(m, sizeof(float), d_yn, 1, h_yn, 1);
    cublasGetVector(m, sizeof(float), d_yt, 1, h_yt, 1);

    print_matrix("B", h_B, m, n);
    print_vector("x", h_xb, n);
    print_vector("y = Bx  (OP_N)", h_yn, m);
    print_vector("y = Bᵀx (OP_T)", h_yt, m);

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_B); cudaFree(d_xb); cudaFree(d_yn); cudaFree(d_yt);
    cublasDestroy(handle);
    return 0;
}
