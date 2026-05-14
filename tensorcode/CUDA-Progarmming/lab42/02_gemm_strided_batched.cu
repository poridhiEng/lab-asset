

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

void print_matrix(float *A, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        printf("    [");
        for (int j = 0; j < cols; j++) printf(" %5.1f", A[j*rows+i]);
        printf(" ]\n");
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int batch = 3, m = 2, n = 2, k = 2;

    /* stride = size of one matrix (packed with no gaps) */
    long long strideA = m * k;
    long long strideB = k * n;
    long long strideC = m * n;

    /* All A matrices packed flat: [A0|A1|A2] */
    float h_A[] = {
        1,2,3,4,    /* A[0] col-major */
        5,6,7,8,    /* A[1] col-major */
        1,0,0,1     /* A[2] identity  */
    };
    float h_B[] = {
        1,0,0,1,    /* B[0] identity  */
        2,2,2,2,    /* B[1] all twos  */
        3,4,5,6     /* B[2]           */
    };
    float h_C[12] = {0};

    /* Single allocation per matrix type */
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, batch*strideA*sizeof(float));
    cudaMalloc((void**)&d_B, batch*strideB*sizeof(float));
    cudaMalloc((void**)&d_C, batch*strideC*sizeof(float));
    cudaMemcpy(d_A, h_A, batch*strideA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, batch*strideB*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0,   batch*strideC*sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    /* cublasSgemmStridedBatched(handle,
     *   transa, transb,
     *   m, n, k,
     *   &alpha,
     *   d_A, lda, strideA,   — base pointer, lda, jump between A matrices
     *   d_B, ldb, strideB,   — base pointer, ldb, jump between B matrices
     *   &beta,
     *   d_C, ldc, strideC,   — base pointer, ldc, jump between C matrices
     *   batchCount)
     */
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           m, n, k,
                                           &alpha,
                                           d_A, m, strideA,
                                           d_B, k, strideB,
                                           &beta,
                                           d_C, m, strideC,
                                           batch));

    cudaMemcpy(h_C, d_C, batch*strideC*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch; i++) {
        printf("  C[%d] = A[%d] * B[%d]:\n", i, i, i);
        print_matrix(h_C + i*strideC, m, n);
    }

    /* ----------------------------------------------------------------
     * Stride = 0 trick: share one B matrix across all batch items
     *
     * Setting strideB = 0 means every batch item reads from B[0].
     * No extra memory — one matrix broadcast to entire batch.
     * ---------------------------------------------------------------- */
    printf("\n  Stride-0 trick: all batches share B[0] (identity)\n");

    float h_B_id[] = {1,0,0,1};   /* single identity matrix */
    float *d_B_id;
    cudaMalloc((void**)&d_B_id, k*n*sizeof(float));
    cudaMemcpy(d_B_id, h_B_id, k*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, batch*strideC*sizeof(float));

    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           m, n, k,
                                           &alpha,
                                           d_A,    m, strideA,
                                           d_B_id, k, 0,        /* strideB=0 */
                                           &beta,
                                           d_C, m, strideC,
                                           batch));

    cudaMemcpy(h_C, d_C, batch*strideC*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch; i++) {
        printf("  C[%d] = A[%d] * I:\n", i, i);
        print_matrix(h_C + i*strideC, m, n);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_B_id);
    cublasDestroy(handle);
    return 0;
}
