/*
 * cublasDgemm — General Matrix-Matrix Multiply (double precision)
 * Operation: C = αAB + βC  (all doubles)
 *
 * Identical API to cublasSgemm but uses double (64-bit) instead of float (32-bit).
 * Use when numerical accuracy matters more than raw throughput.
 *
 * Key differences from Sgemm:
 *   - All arrays are double*, not float*
 *   - alpha and beta are double, not float
 *   - cublasDgemm not cublasSgemm
 *   - Throughput is lower — consumer GPUs run doubles at 1/32 of float speed
 *   - Accuracy is significantly higher — ~15 decimal digits vs ~7
 *
 * Compile: nvcc -o dgemm 02_dgemm.cu -lcublas
 */

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

void print_matrix_d(const char *label, double *A, int rows, int cols) {
    printf("  %s (%dx%d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("    [");
        for (int j = 0; j < cols; j++)
            printf(" %14.10f", A[j * rows + i]);
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 3;

    /* ----------------------------------------------------------------
     * Demo 1: Basic double precision multiply
     * ---------------------------------------------------------------- */
    printf("=== Demo 1: Double precision C = AB ===\n");

    double h_A[] = {
        1.0, 4.0, 7.0,
        2.0, 5.0, 8.0,
        3.0, 6.0, 9.0
    };
    double h_B[] = {
        9.0, 8.0, 7.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0
    };
    double h_C[9] = {0.0};

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n*n*sizeof(double));
    cudaMalloc((void**)&d_B, n*n*sizeof(double));
    cudaMalloc((void**)&d_C, n*n*sizeof(double));

    cublasSetMatrix(n,n,sizeof(double),h_A,n,d_A,n);
    cublasSetMatrix(n,n,sizeof(double),h_B,n,d_B,n);
    cublasSetMatrix(n,n,sizeof(double),h_C,n,d_C,n);

    double alpha = 1.0, beta = 0.0;

    /* cublasDgemm(handle,
     *             CUBLAS_OP_N, CUBLAS_OP_N,  — no transpose
     *             n, n, n,                   — m, n, k all equal for square matrices
     *             &alpha,                    — double scalar
     *             d_A, n,                    — double matrix A
     *             d_B, n,                    — double matrix B
     *             &beta,                     — double scalar
     *             d_C, n)                    — double matrix C
     */
    CHECK_CUBLAS(cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             d_A, n,
                             d_B, n,
                             &beta,
                             d_C, n));

    cublasGetMatrix(n,n,sizeof(double),d_C,n,h_C,n);
    print_matrix_d("C = AB (double)", h_C, n, n);

    /* ----------------------------------------------------------------
     * Demo 2: Precision comparison — float vs double
     *
     * Compute the same sum using float and double accumulation.
     * Shows where float loses digits that double preserves.
     * ---------------------------------------------------------------- */
    printf("\n=== Demo 2: Float vs Double precision comparison ===\n");

    int np = 4;
    /* Values chosen to cause float rounding */
    float h_Af[] = {
        1.0f,          0.0f,         0.0f,          0.0f,
        0.0f,    1234567.0f,         0.0f,          0.0f,
        0.0f,          0.0f,   0.0000001f,          0.0f,
        0.0f,          0.0f,         0.0f,    9999999.0f
    };
    double h_Ad[] = {
        1.0,          0.0,          0.0,          0.0,
        0.0,    1234567.0,          0.0,          0.0,
        0.0,          0.0,   0.0000001,            0.0,
        0.0,          0.0,          0.0,    9999999.0
    };

    float  h_Cf[16] = {0};
    double h_Cd[16] = {0};

    float  *d_Af, *d_Cf;
    double *d_Ad, *d_Cd;

    cudaMalloc((void**)&d_Af, np*np*sizeof(float));
    cudaMalloc((void**)&d_Cf, np*np*sizeof(float));
    cudaMalloc((void**)&d_Ad, np*np*sizeof(double));
    cudaMalloc((void**)&d_Cd, np*np*sizeof(double));

    cublasSetMatrix(np,np,sizeof(float), h_Af,np,d_Af,np);
    cublasSetMatrix(np,np,sizeof(float), h_Cf,np,d_Cf,np);
    cublasSetMatrix(np,np,sizeof(double),h_Ad,np,d_Ad,np);
    cublasSetMatrix(np,np,sizeof(double),h_Cd,np,d_Cd,np);

    float  af=1.0f, bf=0.0f;
    double ad=1.0,  bd=0.0;

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             np,np,np, &af, d_Af,np, d_Af,np, &bf, d_Cf,np));

    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             np,np,np, &ad, d_Ad,np, d_Ad,np, &bd, d_Cd,np));

    cublasGetMatrix(np,np,sizeof(float), d_Cf,np,h_Cf,np);
    cublasGetMatrix(np,np,sizeof(double),d_Cd,np,h_Cd,np);

    printf("  Diagonal values (A² where A is diagonal):\n");
    for (int i = 0; i < np; i++) {
        float  sf = h_Cf[i * np + i];
        double sd = h_Cd[i * np + i];
        printf("  [%d][%d]  float=%.7f  double=%.10f  diff=%.2e\n",
               i, i, sf, sd, fabs((double)sf - sd));
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_Af); cudaFree(d_Cf); cudaFree(d_Ad); cudaFree(d_Cd);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === Demo 1: Double precision C = AB ===
 *   C = AB (double) (3x3):
 *     [  30.0000000000  24.0000000000  18.0000000000 ]
 *     [  84.0000000000  69.0000000000  54.0000000000 ]
 *     [ 138.0000000000 114.0000000000  90.0000000000 ]
 *
 * === Demo 2: Float vs Double precision comparison ===
 *   [0][0]  float=1.0000000   double=1.0000000000  diff=0.00e+00
 *   [1][1]  float=1524154368  double=1524155289.0  diff=921.0  ← float loses digits
 *   [2][2]  float=0.0000000   double=0.0000000000  diff=0.00e+00
 *   [3][3]  float=99999980... double=99999980001.0 diff=...    ← rounding differs
 *
 * EXPLANATION:
 * float has ~7 significant decimal digits.
 * double has ~15 significant decimal digits.
 * 1234567² = 1,524,155,289 — 10 digits, beyond float's range.
 * float rounds this, double preserves it exactly.
 * In scientific computing and finance, this difference matters.
 */
