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

void print_vector(const char *label, float *v, int n) {
    printf("  %-24s: [", label);
    for (int i = 0; i < n; i++) {
        printf("%6.2f", v[i]);
        if (i < n-1) printf(", ");
    }
    printf("]\n");
}

int main() {

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    printf("=== Demo 1: L2 and L1 Norms ===\n");

    int n = 3;
    float h_x[] = {3.0f, -4.0f, 0.0f};
    print_vector("x", h_x, n);

    float *d_x;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1));

    float l2_norm = 0.0f;
    float l1_norm = 0.0f;

    CHECK_CUBLAS(cublasSnrm2(handle, n, d_x, 1, &l2_norm));

    CHECK_CUBLAS(cublasSasum(handle, n, d_x, 1, &l1_norm));

    printf("  L2 norm (cublasSnrm2): %.4f  [expected: 5.0000]\n", l2_norm);
    printf("  L1 norm (cublasSasum): %.4f  [expected: 7.0000]\n", l1_norm);

    cudaFree(d_x);

    printf("\n=== Demo 2: Vector Normalization ===\n");

    int n2 = 4;
    float h_v[] = {1.0f, 2.0f, 3.0f, 4.0f};
    print_vector("v (before)", h_v, n2);

    float *d_v;
    cudaMalloc((void**)&d_v, n2 * sizeof(float));
    CHECK_CUBLAS(cublasSetVector(n2, sizeof(float), h_v, 1, d_v, 1));

    float norm = 0.0f;
    CHECK_CUBLAS(cublasSnrm2(handle, n2, d_v, 1, &norm));
    printf("  L2 norm before: %.4f\n", norm);

    float inv_norm = 1.0f / norm;
    CHECK_CUBLAS(cublasSscal(handle, n2, &inv_norm, d_v, 1));

    float norm_after = 0.0f;
    CHECK_CUBLAS(cublasSnrm2(handle, n2, d_v, 1, &norm_after));

    CHECK_CUBLAS(cublasGetVector(n2, sizeof(float), d_v, 1, h_v, 1));
    print_vector("v (after normalization)", h_v, n2);
    printf("  L2 norm after:  %.4f  [expected: 1.0000]\n", norm_after);

    cudaFree(d_v);

    printf("\n=== Demo 3: Outlier Sensitivity (L1 vs L2) ===\n");

    int n3 = 4;
    float h_uniform[] = {1.0f, 1.0f,   1.0f,   1.0f};
    float h_outlier[] = {1.0f, 1.0f,   1.0f, 100.0f};

    float *d_u, *d_o;
    cudaMalloc((void**)&d_u, n3 * sizeof(float));
    cudaMalloc((void**)&d_o, n3 * sizeof(float));
    cublasSetVector(n3, sizeof(float), h_uniform, 1, d_u, 1);
    cublasSetVector(n3, sizeof(float), h_outlier, 1, d_o, 1);

    float l1_u, l2_u, l1_o, l2_o;
    CHECK_CUBLAS(cublasSasum(handle, n3, d_u, 1, &l1_u));
    CHECK_CUBLAS(cublasSnrm2(handle, n3, d_u, 1, &l2_u));
    CHECK_CUBLAS(cublasSasum(handle, n3, d_o, 1, &l1_o));
    CHECK_CUBLAS(cublasSnrm2(handle, n3, d_o, 1, &l2_o));

    print_vector("Uniform [1,1,1,1]", h_uniform, n3);
    printf("    L1 = %.3f,  L2 = %.3f\n", l1_u, l2_u);
    print_vector("Outlier [1,1,1,100]", h_outlier, n3);
    printf("    L1 = %.3f,  L2 = %.3f\n", l1_o, l2_o);
    printf("\n  L1 ratio (outlier/uniform): %.2fx\n", l1_o / l1_u);
    printf("  L2 ratio (outlier/uniform): %.2fx\n", l2_o / l2_u);
    printf("  L2 is %.1fx more sensitive to the outlier than L1.\n",
           (l2_o / l2_u) / (l1_o / l1_u));

    cudaFree(d_u); cudaFree(d_o);
    cublasDestroy(handle);

    return 0;
}
