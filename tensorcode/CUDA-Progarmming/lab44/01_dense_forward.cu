#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUBLAS(call) {                                        \
    cublasStatus_t _e = (call);                                     \
    if (_e != CUBLAS_STATUS_SUCCESS) {                              \
        printf("cuBLAS error %d at line %d\n", _e, __LINE__);      \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

void print_matrix(const char *label, float *A, int rows, int cols) {
    printf("  %s:\n", label);
    for (int i = 0; i < rows; i++) {
        printf("    [");
        for (int j = 0; j < cols; j++)
            printf(" %6.3f", A[j * rows + i]);
        printf(" ]\n");
    }
}

int main() {

    /* Network dimensions */
    int batch    = 4;   /* number of input samples    */
    int in_feat  = 3;   /* input  features per sample */
    int out_feat = 2;   /* output features per sample */

    /*
     * X: shape (in_feat x batch) = (3 x 4), stored column-major.
     * Each COLUMN is one sample's feature vector.
     *
     *        sample0  sample1  sample2  sample3
     * feat0 [  1.0     0.3      0.9      0.2  ]
     * feat1 [  0.5     0.8      0.4      0.6  ]
     * feat2 [  0.2     0.1      0.7      0.3  ]
     *
     * Column-major storage: read down col0, then col1, ...
     */
    float h_X[] = {
        1.0f, 0.5f, 0.2f,   /* col 0: sample 0 */
        0.3f, 0.8f, 0.1f,   /* col 1: sample 1 */
        0.9f, 0.4f, 0.7f,   /* col 2: sample 2 */
        0.2f, 0.6f, 0.3f    /* col 3: sample 3 */
    };

    /*
     * W: shape (out_feat x in_feat) = (2 x 3), stored column-major.
     * Each COLUMN holds the weights for one input feature across all neurons.
     *
     *          feat0  feat1  feat2
     * neuron0 [ 0.1    0.2    0.3 ]
     * neuron1 [ 0.4    0.5    0.6 ]
     *
     * Column-major storage: read down col0, then col1, ...
     */
    float h_W[] = {
        0.1f, 0.4f,   /* col 0: weights for feat0 — neuron0, neuron1 */
        0.2f, 0.5f,   /* col 1: weights for feat1 */
        0.3f, 0.6f    /* col 2: weights for feat2 */
    };

    /* Bias: one value per output neuron */
    float h_b[] = { 0.1f, -0.1f };

    /* Output buffer: (out_feat x batch) = (2 x 4) */
    float h_Y[8] = { 0 };

    /* Create cuBLAS handle */
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    /* Allocate GPU memory */
    float *d_X, *d_W, *d_Y, *d_b;
    cudaMalloc((void**)&d_X, in_feat  * batch   * sizeof(float));
    cudaMalloc((void**)&d_W, out_feat * in_feat  * sizeof(float));
    cudaMalloc((void**)&d_Y, out_feat * batch    * sizeof(float));
    cudaMalloc((void**)&d_b, out_feat            * sizeof(float));

    /*
     * cublasSetMatrix(rows, cols, elemSize, src, src_ld, dst, dst_ld)
     * X is (in_feat x batch):   rows=in_feat,  cols=batch,   ld=in_feat
     * W is (out_feat x in_feat): rows=out_feat, cols=in_feat, ld=out_feat
     */
    cublasSetMatrix(in_feat,  batch,    sizeof(float), h_X, in_feat,  d_X, in_feat);
    cublasSetMatrix(out_feat, in_feat,  sizeof(float), h_W, out_feat, d_W, out_feat);
    cublasSetVector(out_feat, sizeof(float), h_b, 1, d_b, 1);
    cudaMemset(d_Y, 0, out_feat * batch * sizeof(float));

    float alpha = 1.0f;
    float beta  = 0.0f;

    /* ----------------------------------------------------------------
     * Step 1: Y = W * X
     *
     * Math:  (out_feat x batch) = (out_feat x in_feat) * (in_feat x batch)
     *
     * cublasSgemm computes: C = alpha * op(A) * op(B) + beta * C
     *
     *   op(A) = W, no transpose needed (CUBLAS_OP_N), shape (out_feat x in_feat)
     *   op(B) = X, no transpose needed (CUBLAS_OP_N), shape (in_feat x batch)
     *   C     = Y,                                     shape (out_feat x batch)
     *
     *   m = out_feat  (rows of op(A) and C)
     *   n = batch     (cols of op(B) and C)
     *   k = in_feat   (inner dimension)
     *
     *   lda = out_feat  (leading dim of W in col-major)
     *   ldb = in_feat   (leading dim of X in col-major)
     *   ldc = out_feat  (leading dim of Y in col-major)
     */
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             out_feat, batch, in_feat,
                             &alpha,
                             d_W, out_feat,
                             d_X, in_feat,
                             &beta,
                             d_Y, out_feat));

    /* ----------------------------------------------------------------
     * Step 2: Y += b   (broadcast bias to every sample/column)
     *
     * One cublasSaxpy call per sample; each advances out_feat elements
     * into d_Y to reach the next sample's column.
     */
    for (int s = 0; s < batch; s++)
        CHECK_CUBLAS(cublasSaxpy(handle, out_feat,
                                 &alpha,
                                 d_b, 1,
                                 d_Y + s * out_feat, 1));

    /* Copy result back to CPU */
    cublasGetMatrix(out_feat, batch, sizeof(float), d_Y, out_feat, h_Y, out_feat);

    /* Print results */
    printf("Dense Layer: Y = W * X + b\n\n");
    print_matrix("X  (in_feat x batch)",    h_X, in_feat,  batch);
    print_matrix("W  (out_feat x in_feat)", h_W, out_feat, in_feat);
    printf("\n  bias: [%.3f, %.3f]\n\n", h_b[0], h_b[1]);
    print_matrix("Y  (out_feat x batch)",   h_Y, out_feat, batch);

    printf("\n  Sample 0: output[0]=%.3f  output[1]=%.3f\n",
           h_Y[0], h_Y[1]);

    /* Cleanup */
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);
    cudaFree(d_b);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 *
 * Sample 0 output:
 *   output[0] = 1.0*0.1 + 0.5*0.2 + 0.2*0.3 + 0.1 = 0.360
 *   output[1] = 1.0*0.4 + 0.5*0.5 + 0.2*0.6 - 0.1 = 0.770
 *
 * Compile:
 *   nvcc -o dense main.cu -lcublas
 */