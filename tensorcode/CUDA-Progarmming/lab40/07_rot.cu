#include <stdio.h>
#include <math.h>
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

#define PI 3.14159265358979f

void rotate_and_print(cublasHandle_t handle,
                      float *h_x, float *h_y, int n,
                      float c, float s, const char *label) {
    printf("  %s (c=%.4f, s=%.4f):\n", label, c, s);

    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1);
    cublasSetVector(n, sizeof(float), h_y, 1, d_y, 1);

    CHECK_CUBLAS(cublasSrot(handle, n, d_x, 1, d_y, 1, &c, &s));

    float result_x[16], result_y[16];
    cublasGetVector(n, sizeof(float), d_x, 1, result_x, 1);
    cublasGetVector(n, sizeof(float), d_y, 1, result_y, 1);

    printf("    x: [");
    for (int i = 0; i < n; i++) printf("%.3f%s", result_x[i], i<n-1?", ":"");
    printf("]\n");
    printf("    y: [");
    for (int i = 0; i < n; i++) printf("%.3f%s", result_y[i], i<n-1?", ":"");
    printf("]\n\n");

    cudaFree(d_x); cudaFree(d_y);
}

int main() {

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    printf("=== Demo 1: Rotating unit vector (1, 0) ===\n");
    printf("  Starting vector: x=[1.0], y=[0.0]\n\n");

    float h_x1[] = {1.0f};
    float h_y1[] = {0.0f};

    float c90 = cosf(PI / 2.0f);
    float s90 = sinf(PI / 2.0f);
    rotate_and_print(handle, h_x1, h_y1, 1, c90, s90, "90-degree rotation");

    h_x1[0] = 1.0f; h_y1[0] = 0.0f;
    float c45 = cosf(PI / 4.0f);
    float s45 = sinf(PI / 4.0f);
    rotate_and_print(handle, h_x1, h_y1, 1, c45, s45, "45-degree rotation");

    h_x1[0] = 1.0f; h_y1[0] = 0.0f;
    float c0 = 1.0f, s0 = 0.0f;
    rotate_and_print(handle, h_x1, h_y1, 1, c0, s0, "0-degree rotation (identity)");

    printf("=== Demo 2: Batch Rotation (4 pairs, 90-degree) ===\n");

    int n2 = 4;
    float h_bx[] = {1.0f, 0.0f, 2.0f, 4.0f};
    float h_by[] = {0.0f, 1.0f, 3.0f, 0.0f};

    printf("  Before:\n");
    printf("    x: [1.0, 0.0, 2.0, 4.0]\n");
    printf("    y: [0.0, 1.0, 3.0, 0.0]\n\n");

    rotate_and_print(handle, h_bx, h_by, n2, c90, s90, "90-degree rotation (4 pairs)");

    printf("  Manual verification for pair (2.0, 3.0):\n");
    printf("    x' =  cos(90)*2 + sin(90)*3 =  0*2 + 1*3 =  3.000\n");
    printf("    y' = -sin(90)*2 + cos(90)*3 = -1*2 + 0*3 = -2.000\n");

    cublasDestroy(handle);

    return 0;
}
