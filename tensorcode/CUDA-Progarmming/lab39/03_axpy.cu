#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUBLAS(call)                                              \
    {                                                                   \
        cublasStatus_t _err = (call);                                   \
        if (_err != CUBLAS_STATUS_SUCCESS) {                            \
            printf("cuBLAS error %d at line %d\n", _err, __LINE__);    \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    }

void print_vector(const char *label, float *v, int n) {
    printf("  %-22s: ", label);
    for (int i = 0; i < n; i++) printf("%7.3f ", v[i]);
    printf("\n");
}

float* upload(float *h, int n) {
    float *d;
    cudaMalloc((void**)&d, n * sizeof(float));
    cublasSetVector(n, sizeof(float), h, 1, d, 1);
    return d;
}

void download(float *d, float *h, int n) {
    cublasGetVector(n, sizeof(float), d, 1, h, 1);
}

int main() {

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 5;

    printf("=== Demo 1: Basic AXPY (y = 2x + y) ===\n");

    float h_x1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_y1[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float alpha1 = 2.0f;

    print_vector("x", h_x1, n);
    print_vector("y (before)", h_y1, n);

    float *d_x1 = upload(h_x1, n);
    float *d_y1 = upload(h_y1, n);

    CHECK_CUBLAS(cublasSaxpy(handle, n, &alpha1, d_x1, 1, d_y1, 1));

    download(d_y1, h_y1, n);
    print_vector("y (after: 2x + y)", h_y1, n);

    cudaFree(d_x1); cudaFree(d_y1);

    printf("\n=== Demo 2: Vector Addition (α=1.0, y = x + y) ===\n");

    float h_x2[] = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    float h_y2[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float alpha2 = 1.0f;

    print_vector("x", h_x2, n);
    print_vector("y (before)", h_y2, n);

    float *d_x2 = upload(h_x2, n);
    float *d_y2 = upload(h_y2, n);

    CHECK_CUBLAS(cublasSaxpy(handle, n, &alpha2, d_x2, 1, d_y2, 1));

    download(d_y2, h_y2, n);
    print_vector("y (after: x + y)", h_y2, n);

    cudaFree(d_x2); cudaFree(d_y2);

    printf("\n=== Demo 3: Gradient Descent Simulation ===\n");

    float learning_rate = 0.1f;

    float weights[]   = {0.5f,  -0.3f,  0.8f,  0.1f, -0.6f};
    float gradients[] = {0.2f,  -0.1f,  0.4f, -0.3f,  0.5f};

    float alpha3 = -learning_rate;

    printf("  Learning rate: %.2f\n", learning_rate);
    print_vector("weights (before)", weights, n);
    print_vector("gradients", gradients, n);

    float *d_w = upload(weights, n);
    float *d_g = upload(gradients, n);

    CHECK_CUBLAS(cublasSaxpy(handle, n, &alpha3, d_g, 1, d_w, 1));

    download(d_w, weights, n);
    print_vector("weights (after 1 step)", weights, n);

    cudaFree(d_w); cudaFree(d_g);
    cublasDestroy(handle);

    return 0;
}
