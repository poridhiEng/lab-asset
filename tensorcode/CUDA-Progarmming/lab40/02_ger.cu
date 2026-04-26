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
            printf(" %6.2f", A[j * rows + i]);
        printf(" ]\n");
    }
}

void print_vector(const char *label, float *v, int n) {
    printf("  %-20s: [", label);
    for (int i = 0; i < n; i++) printf(" %6.2f", v[i]);
    printf(" ]\n");
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    printf("=== Demo 1: Pure outer product A = xyᵀ ===\n");

    int m = 3, n = 2;

    float h_x[] = { 1.0f, 2.0f, 3.0f };
    float h_y[] = { 4.0f, 5.0f };
    float h_A[6] = { 0.0f };

    print_vector("x", h_x, m);
    print_vector("y", h_y, n);

    float *d_x, *d_y, *d_A;
    cudaMalloc((void**)&d_x, m * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_A, m * n * sizeof(float));

    cublasSetVector(m, sizeof(float), h_x, 1, d_x, 1);
    cublasSetVector(n, sizeof(float), h_y, 1, d_y, 1);
    cublasSetMatrix(m, n, sizeof(float), h_A, m, d_A, m);

    float alpha = 1.0f;

    CHECK_CUBLAS(cublasSger(handle, m, n,
                            &alpha,
                            d_x, 1,
                            d_y, 1,
                            d_A, m));

    cublasGetMatrix(m, n, sizeof(float), d_A, m, h_A, m);
    print_matrix("A = xyᵀ", h_A, m, n);

    printf("\n=== Demo 2: Rank-1 update A = αxyᵀ + A ===\n");

    float h_A2[] = {
        10.0f, 0.0f, 0.0f,
         0.0f, 10.0f, 0.0f
    };

    print_matrix("A (before)", h_A2, m, n);
    cublasSetMatrix(m, n, sizeof(float), h_A2, m, d_A, m);

    float alpha2 = 2.0f;

    CHECK_CUBLAS(cublasSger(handle, m, n,
                            &alpha2,
                            d_x, 1,
                            d_y, 1,
                            d_A, m));

    cublasGetMatrix(m, n, sizeof(float), d_A, m, h_A2, m);
    print_matrix("A = 2xyᵀ + A", h_A2, m, n);

    printf("\n=== Demo 3: Neural network weight update W -= lr * δxᵀ ===\n");

    int input_size = 3, output_size = 2;
    float lr = 0.1f;

    float h_W[] = {
        0.5f, 0.3f, 0.1f,
        0.4f, 0.2f, 0.6f
    };
    float h_input[]  = { 1.0f, 0.5f, 0.2f };
    float h_delta[]  = { 0.3f, -0.1f };

    print_matrix("W (before)", h_W, input_size, output_size);
    print_vector("input", h_input, input_size);
    print_vector("delta", h_delta, output_size);

    float *d_W, *d_input, *d_delta;
    cudaMalloc((void**)&d_W,     input_size * output_size * sizeof(float));
    cudaMalloc((void**)&d_input, input_size * sizeof(float));
    cudaMalloc((void**)&d_delta, output_size * sizeof(float));

    cublasSetMatrix(input_size, output_size, sizeof(float), h_W, input_size, d_W, input_size);
    cublasSetVector(input_size, sizeof(float), h_input, 1, d_input, 1);
    cublasSetVector(output_size, sizeof(float), h_delta, 1, d_delta, 1);

    float neg_lr = -lr;

    CHECK_CUBLAS(cublasSger(handle, input_size, output_size,
                            &neg_lr,
                            d_input, 1,
                            d_delta, 1,
                            d_W, input_size));

    cublasGetMatrix(input_size, output_size, sizeof(float), d_W, input_size, h_W, input_size);
    print_matrix("W (after one update)", h_W, input_size, output_size);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_A);
    cudaFree(d_W); cudaFree(d_input); cudaFree(d_delta);
    cublasDestroy(handle);
    return 0;
}
