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
    printf("  %-30s: [", label);
    for (int i = 0; i < n; i++) {
        printf("%.4f%s", v[i], i < n-1 ? ", " : "");
    }
    printf("]\n");
}

int main() {

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 6;
    const char *features[] = {"price", "size", "rating", "age", "distance", "popularity"};

    float h_x[] = {9.0f, 8.0f, 9.5f, 2.0f, 1.0f, 8.0f};
    float h_y[] = {8.5f, 5.0f, 9.0f, 2.5f, 9.0f, 2.0f};

    printf("=== Cosine Similarity Pipeline ===\n\n");
    printf("Feature vector dimensions: %d\n", n);
    printf("Features: [price, size, rating, age, distance, popularity]\n\n");
    print_vector("Item A (x)", h_x, n);
    print_vector("Item B (y)", h_y, n);
    printf("\n");

    float *d_x, *d_y, *d_xn, *d_yn;
    cudaMalloc((void**)&d_x,  n * sizeof(float));
    cudaMalloc((void**)&d_y,  n * sizeof(float));
    cudaMalloc((void**)&d_xn, n * sizeof(float));
    cudaMalloc((void**)&d_yn, n * sizeof(float));

    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_y, 1, d_y, 1));

    printf("--- Step 1: Copy vectors to working buffers ---\n");
    CHECK_CUBLAS(cublasScopy(handle, n, d_x, 1, d_xn, 1));
    CHECK_CUBLAS(cublasScopy(handle, n, d_y, 1, d_yn, 1));
    printf("  Copies created. Originals preserved.\n\n");

    printf("--- Steps 2 & 3: Compute L2 norms ---\n");
    float norm_x = 0.0f, norm_y = 0.0f;
    CHECK_CUBLAS(cublasSnrm2(handle, n, d_x, 1, &norm_x));
    CHECK_CUBLAS(cublasSnrm2(handle, n, d_y, 1, &norm_y));
    printf("  ||x||_2 = %.4f\n", norm_x);
    printf("  ||y||_2 = %.4f\n\n", norm_y);

    printf("--- Steps 4 & 5: Normalize vectors ---\n");
    float inv_norm_x = 1.0f / norm_x;
    float inv_norm_y = 1.0f / norm_y;
    CHECK_CUBLAS(cublasSscal(handle, n, &inv_norm_x, d_xn, 1));
    CHECK_CUBLAS(cublasSscal(handle, n, &inv_norm_y, d_yn, 1));

    float h_xn[6], h_yn[6];
    CHECK_CUBLAS(cublasGetVector(n, sizeof(float), d_xn, 1, h_xn, 1));
    CHECK_CUBLAS(cublasGetVector(n, sizeof(float), d_yn, 1, h_yn, 1));
    print_vector("x_hat (normalized x)", h_xn, n);
    print_vector("y_hat (normalized y)", h_yn, n);

    float verify_xn = 0.0f, verify_yn = 0.0f;
    CHECK_CUBLAS(cublasSnrm2(handle, n, d_xn, 1, &verify_xn));
    CHECK_CUBLAS(cublasSnrm2(handle, n, d_yn, 1, &verify_yn));
    printf("  ||x_hat||_2 = %.4f  [expected: 1.0000]\n", verify_xn);
    printf("  ||y_hat||_2 = %.4f  [expected: 1.0000]\n\n", verify_yn);

    printf("--- Step 6: Dot product of normalized vectors = Cosine Similarity ---\n");
    float cosine_sim = 0.0f;
    CHECK_CUBLAS(cublasSdot(handle, n, d_xn, 1, d_yn, 1, &cosine_sim));
    printf("  Cosine similarity: %.4f\n", cosine_sim);

    if (cosine_sim > 0.9f)
        printf("  Interpretation: Very high similarity (>0.9)\n\n");
    else if (cosine_sim > 0.7f)
        printf("  Interpretation: High similarity (0.7-0.9)\n\n");
    else if (cosine_sim > 0.5f)
        printf("  Interpretation: Moderate similarity (0.5-0.7)\n\n");
    else
        printf("  Interpretation: Low similarity (<0.5)\n\n");

    printf("--- Step 7: Find Dominant Feature (cublasIsamax) ---\n");
    int max_x_raw = 0, max_y_raw = 0;
    CHECK_CUBLAS(cublasIsamax(handle, n, d_x, 1, &max_x_raw));
    CHECK_CUBLAS(cublasIsamax(handle, n, d_y, 1, &max_y_raw));

    int max_x_idx = max_x_raw - 1;
    int max_y_idx = max_y_raw - 1;

    printf("  Item A dominant feature: '%s' (value: %.1f)\n",
           features[max_x_idx], h_x[max_x_idx]);
    printf("  Item B dominant feature: '%s' (value: %.1f)\n\n",
           features[max_y_idx], h_y[max_y_idx]);

    printf("--- Step 8: Compare Total Feature Mass (cublasSasum) ---\n");
    float l1_x = 0.0f, l1_y = 0.0f;
    CHECK_CUBLAS(cublasSasum(handle, n, d_x, 1, &l1_x));
    CHECK_CUBLAS(cublasSasum(handle, n, d_y, 1, &l1_y));
    printf("  Item A total feature mass (L1): %.2f\n", l1_x);
    printf("  Item B total feature mass (L1): %.2f\n", l1_y);
    printf("  Item %s has more total feature weight.\n\n", l1_x > l1_y ? "A" : "B");

    printf("=== Summary ===\n");
    printf("  Cosine similarity:        %.4f\n", cosine_sim);
    printf("  Item A dominant feature:  %s\n", features[max_x_idx]);
    printf("  Item B dominant feature:  %s\n", features[max_y_idx]);
    printf("  Item A total mass (L1):   %.2f\n", l1_x);
    printf("  Item B total mass (L1):   %.2f\n", l1_y);

    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_xn); cudaFree(d_yn);
    cublasDestroy(handle);

    return 0;
}
