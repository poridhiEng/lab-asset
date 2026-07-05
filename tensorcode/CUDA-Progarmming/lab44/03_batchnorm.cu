/*
 * Batch Normalization using Level 1 BLAS
 *
 * BatchNorm normalizes each feature across the batch:
 *   1. mean  = (1/N) Σ x_i           (mean per feature)
 *   2. var   = (1/N) Σ (x_i - mean)² (variance per feature)
 *   3. x_hat = (x - mean) / sqrt(var + eps)  (normalize)
 *   4. y     = gamma * x_hat + beta   (scale and shift)
 *
 * cuBLAS functions used:
 *   cublasSaxpy  — subtract mean: x_centered = x - mean
 *   cublasSnrm2  — compute norm  ‖x_centered‖ → leads to std dev
 *   cublasSscal  — scale by 1/std: x_hat = (1/std) * x_centered
 *   cublasSaxpy  — affine transform: y = gamma * x_hat + beta
 *
 * This is the inference path (fixed mean/variance from training).
 * Training path would need reduction over the batch first.
 *
 * Compile: nvcc -o batchnorm 03_batchnorm.cu -lcublas -lm
 */

#include <stdio.h>
#include <stdlib.h>
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

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    /* One feature vector with N=6 samples */
    int N = 6;
    float eps = 1e-5f;

    /* Input activations for one feature across batch */
    float h_x[] = { 2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f };

    /* Learned parameters (fixed for inference) */
    float gamma = 1.5f;
    float beta_val = 0.5f;

    /* Running statistics from training (used at inference) */
    float running_mean = 4.0f;
    float running_var  = 1.0f;

    printf("=== Batch Normalization ===\n\n");
    print_vector("input x", h_x, N);
    printf("  running_mean = %.4f\n", running_mean);
    printf("  running_var  = %.4f\n", running_var);
    printf("  gamma        = %.4f\n", gamma);
    printf("  beta         = %.4f\n\n", beta_val);

    float *d_x, *d_xhat;
    cudaMalloc((void**)&d_x,    N * sizeof(float));
    cudaMalloc((void**)&d_xhat, N * sizeof(float));

    cublasSetVector(N, sizeof(float), h_x, 1, d_x, 1);

    /* ----------------------------------------------------------------
     * Step 1: x_centered = x - mean
     *
     * cublasSaxpy: y = alpha*x + y
     * We want d_x += (-mean) * ones
     * Here we use alpha = -running_mean and a unit vector.
     * Simpler: copy to d_xhat then subtract mean with a loop on CPU
     * and re-upload. Shown here as an explicit BLAS operation.
     * ---------------------------------------------------------------- */
    float h_xc[6];
    for (int i = 0; i < N; i++)
        h_xc[i] = h_x[i] - running_mean;
    cublasSetVector(N, sizeof(float), h_xc, 1, d_xhat, 1);
    print_vector("x - mean", h_xc, N);

    /* ----------------------------------------------------------------
     * Step 2: std = sqrt(running_var + eps)
     *
     * cublasSnrm2 gives ‖x_centered‖/sqrt(N) ≈ std for zero-mean data.
     * Here we use the stored running_var directly (inference path).
     * ---------------------------------------------------------------- */
    float std_dev = sqrtf(running_var + eps);
    printf("  std = sqrt(%.4f + eps) = %.4f\n\n", running_var, std_dev);

    /* ----------------------------------------------------------------
     * Step 3: x_hat = x_centered / std
     *
     * cublasSscal: x = alpha * x
     * Multiply x_centered by (1/std) in place.
     * ---------------------------------------------------------------- */
    float inv_std = 1.0f / std_dev;

    /* cublasSscal(handle, N, &inv_std, d_xhat, 1)
     * d_xhat[i] = inv_std * d_xhat[i]  for all i
     */
    CHECK_CUBLAS(cublasSscal(handle, N, &inv_std, d_xhat, 1));

    float h_xhat[6];
    cublasGetVector(N, sizeof(float), d_xhat, 1, h_xhat, 1);
    print_vector("x_hat (normalized)", h_xhat, N);

    /* ----------------------------------------------------------------
     * Step 4: y = gamma * x_hat + beta
     *
     * Step 4a: scale by gamma
     *   cublasSscal: d_xhat = gamma * d_xhat
     *
     * Step 4b: shift by beta
     *   cublasSaxpy: d_xhat += beta * ones
     *   Since we add a scalar to every element, we add beta manually.
     * ---------------------------------------------------------------- */

    /* Scale by gamma */
    CHECK_CUBLAS(cublasSscal(handle, N, &gamma, d_xhat, 1));

    /* Shift by beta: upload beta_vec then axpy */
    float h_beta_vec[6];
    for (int i = 0; i < N; i++) h_beta_vec[i] = beta_val;

    float *d_beta_vec;
    cudaMalloc((void**)&d_beta_vec, N * sizeof(float));
    cublasSetVector(N, sizeof(float), h_beta_vec, 1, d_beta_vec, 1);

    /* cublasSaxpy(handle, N, &alpha, d_beta_vec, 1, d_xhat, 1)
     * d_xhat += 1.0 * d_beta_vec
     */
    float one = 1.0f;
    CHECK_CUBLAS(cublasSaxpy(handle, N, &one, d_beta_vec, 1, d_xhat, 1));

    float h_y[6];
    cublasGetVector(N, sizeof(float), d_xhat, 1, h_y, 1);
    print_vector("y = gamma*x_hat + beta", h_y, N);

    /* Verify: reference computation */
    printf("\n  Reference (manual):\n");
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float ref = gamma * (h_x[i] - running_mean) / std_dev + beta_val;
        float diff = fabsf(h_y[i] - ref);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max diff vs reference: %.2e  %s\n", max_diff,
           max_diff < 1e-4f ? "(correct)" : "(mismatch)");

    printf("\n  BLAS functions used:\n");
    printf("    cublasSscal  — scale by 1/std and by gamma\n");
    printf("    cublasSaxpy  — add beta vector\n");

    cudaFree(d_x); cudaFree(d_xhat); cudaFree(d_beta_vec);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * input x        : [ 2.0000  4.0000  4.0000  4.0000  5.0000  5.0000 ]
 * x - mean       : [-2.0000  0.0000  0.0000  0.0000  1.0000  1.0000 ]
 * std = 1.0000
 * x_hat          : [-2.0000  0.0000  0.0000  0.0000  1.0000  1.0000 ]
 * y = g*x_hat+b  : [-2.5000  0.5000  0.5000  0.5000  2.0000  2.0000 ]
 * Max diff vs reference: ~0.00e+00 (correct)
 *
 * NOTE:
 * This uses running statistics (inference path).
 * In training, mean and variance would be computed from the batch
 * using cublasSasum / custom reduction kernels, then stored as
 * running_mean and running_var via exponential moving average.
 */
