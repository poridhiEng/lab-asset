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

void print_vector_with_index(float *v, int n) {
    printf("  Index (C):  ");
    for (int i = 0; i < n; i++) printf("[%d]%6.1f  ", i, v[i]);
    printf("\n");
}

int main() {

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    printf("=== Demo 1: Basic isamax and isamin ===\n");

    int n = 5;
    float h_x[] = {3.0f, -8.0f, 1.0f, -2.0f, 6.0f};

    printf("  Vector (with C indices):\n  ");
    print_vector_with_index(h_x, n);
    printf("\n  Absolute values: [3.0, 8.0, 1.0, 2.0, 6.0]\n\n");

    float *d_x;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1));

    int idx_max = 0;
    int idx_min = 0;

    CHECK_CUBLAS(cublasIsamax(handle, n, d_x, 1, &idx_max));

    CHECK_CUBLAS(cublasIsamin(handle, n, d_x, 1, &idx_min));

    printf("  cublasIsamax raw result: %d  (1-based)\n", idx_max);
    printf("  cublasIsamin raw result: %d  (1-based)\n", idx_min);
    printf("\n  Correct C-index for max: %d - 1 = %d  (value: %.1f)\n",
           idx_max, idx_max - 1, h_x[idx_max - 1]);
    printf("  Correct C-index for min: %d - 1 = %d  (value: %.1f)\n",
           idx_min, idx_min - 1, h_x[idx_min - 1]);

    cudaFree(d_x);

    printf("\n=== Demo 2: Off-By-One Bug Demonstration ===\n");

    float h_v[] = {1.0f, 2.0f, 99.0f, 4.0f, 5.0f};

    printf("  Vector: [1.0, 2.0, 99.0, 4.0, 5.0]\n");
    printf("  Max absolute value: 99.0 is at C-index 2\n\n");

    float *d_v;
    cudaMalloc((void**)&d_v, n * sizeof(float));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_v, 1, d_v, 1));

    int raw_idx = 0;
    CHECK_CUBLAS(cublasIsamax(handle, n, d_v, 1, &raw_idx));

    printf("  cuBLAS raw result: %d\n", raw_idx);
    printf("\n  WRONG (using raw index):   h_v[%d] = %.1f  <- incorrect element\n",
           raw_idx, h_v[raw_idx]);
    printf("  CORRECT (subtract 1):      h_v[%d] = %.1f  <- correct element\n",
           raw_idx - 1, h_v[raw_idx - 1]);

    cudaFree(d_v);

    printf("\n=== Demo 3: Absolute Max vs Algebraic Max ===\n");

    float h_signed[] = {-100.0f, 1.0f, 2.0f, 3.0f, 50.0f};

    printf("  Vector: [-100.0, 1.0, 2.0, 3.0, 50.0]\n");
    printf("  Algebraic max: 50.0  (largest value)\n");
    printf("  Absolute max:  100.0 (|-100.0|, largest magnitude)\n\n");

    float *d_s;
    cudaMalloc((void**)&d_s, n * sizeof(float));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_signed, 1, d_s, 1));

    int abs_max_idx = 0;
    CHECK_CUBLAS(cublasIsamax(handle, n, d_s, 1, &abs_max_idx));

    int c_idx = abs_max_idx - 1;
    printf("  cublasIsamax result: %d (1-based)  ->  C-index: %d\n",
           abs_max_idx, c_idx);
    printf("  Value at that index: %.1f\n", h_signed[c_idx]);
    printf("  isamax found -100.0, not 50.0, because |-100| > |50|\n");

    cudaFree(d_s);
    cublasDestroy(handle);

    return 0;
}
