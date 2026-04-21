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

int main() {

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 5;
    float alpha = 3.0f;
    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    printf("Input  x: ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_x[i]);
    printf("\n");
    printf("Scalar α: %.1f\n", alpha);

    float *d_x;
    cudaMalloc((void**)&d_x, n * sizeof(float));

    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1));

    CHECK_CUBLAS(cublasSscal(handle, n, &alpha, d_x, 1));

    CHECK_CUBLAS(cublasGetVector(n, sizeof(float), d_x, 1, h_x, 1));

    printf("Output x: ");
    for (int i = 0; i < n; i++) printf("%.1f ", h_x[i]);
    printf("\n");

    cudaFree(d_x);
    cublasDestroy(handle);

    return 0;
}
