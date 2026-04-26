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

void print_vector(const char *label, float *v, int n) {
    printf("  %-22s: [", label);
    for (int i = 0; i < n; i++) printf(" %7.4f", v[i]);
    printf(" ]\n");
}

void print_matrix(const char *label, float *A, int rows, int cols) {
    printf("  %s (%dx%d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("    [");
        for (int j = 0; j < cols; j++)
            printf(" %7.4f", A[j * rows + i]);
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int input_size  = 3;
    int output_size = 2;
    float lr = 0.1f;

    float h_x[]      = { 1.0f, 0.5f, 0.25f };
    float h_target[]  = { 1.0f, 0.0f };

    float h_W[] = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f
    };

    printf("=== Linear Layer Training ===\n");
    printf("  Input size:  %d\n", input_size);
    printf("  Output size: %d\n", output_size);
    printf("  Learning rate: %.2f\n\n", lr);

    print_vector("Input x", h_x, input_size);
    print_vector("Target", h_target, output_size);
    print_matrix("Initial W", h_W, input_size, output_size);
    printf("\n");

    float *d_W, *d_x, *d_y, *d_delta;
    cudaMalloc((void**)&d_W,     input_size * output_size * sizeof(float));
    cudaMalloc((void**)&d_x,     input_size * sizeof(float));
    cudaMalloc((void**)&d_y,     output_size * sizeof(float));
    cudaMalloc((void**)&d_delta, output_size * sizeof(float));

    cublasSetVector(input_size, sizeof(float), h_x, 1, d_x, 1);

    float one = 1.0f, zero = 0.0f;

    for (int iter = 0; iter < 10; iter++) {

        cublasSetMatrix(input_size, output_size, sizeof(float),
                        h_W, input_size, d_W, input_size);

        float zeros[2] = {0.0f, 0.0f};
        cublasSetVector(output_size, sizeof(float), zeros, 1, d_y, 1);

        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N,
                                 input_size, output_size,
                                 &one, d_W, input_size,
                                 d_x, 1,
                                 &zero, d_y, 1));

        float h_y[2];
        cublasGetVector(output_size, sizeof(float), d_y, 1, h_y, 1);

        float loss = 0.0f;
        float h_delta[2];
        for (int i = 0; i < output_size; i++) {
            h_delta[i] = h_y[i] - h_target[i];
            loss += 0.5f * h_delta[i] * h_delta[i];
        }

        printf("  Iter %2d | loss = %.6f | y = [%.4f, %.4f]\n",
               iter+1, loss, h_y[0], h_y[1]);

        cublasSetVector(output_size, sizeof(float), h_delta, 1, d_delta, 1);

        float neg_lr = -lr;

        CHECK_CUBLAS(cublasSger(handle,
                                input_size, output_size,
                                &neg_lr,
                                d_x, 1,
                                d_delta, 1,
                                d_W, input_size));

        cublasGetMatrix(input_size, output_size, sizeof(float),
                        d_W, input_size, h_W, input_size);
    }

    printf("\n");
    print_matrix("Final W (after 10 iterations)", h_W, input_size, output_size);

    cublasSetMatrix(input_size, output_size, sizeof(float),
                    h_W, input_size, d_W, input_size);
    float h_y_final[2] = {0};
    cublasSetVector(output_size, sizeof(float), h_y_final, 1, d_y, 1);

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N,
                             input_size, output_size,
                             &one, d_W, input_size,
                             d_x, 1,
                             &zero, d_y, 1));

    cublasGetVector(output_size, sizeof(float), d_y, 1, h_y_final, 1);
    print_vector("Final prediction y", h_y_final, output_size);
    print_vector("Target            ", h_target, output_size);

    cudaFree(d_W); cudaFree(d_x); cudaFree(d_y); cudaFree(d_delta);
    cublasDestroy(handle);
    return 0;
}
