/*
 * Capstone: 2-Layer Neural Network Training
 *
 * Implements a fully connected 2-layer network using Level 3 BLAS only.
 *
 * Architecture:
 *   Input x: (batch=4, features=3)
 *   Layer 1: W1 (3→4), ReLU activation
 *   Layer 2: W2 (4→2), output
 *   Loss:    MSE
 *
 * cuBLAS functions used:
 *   cublasSgemm  — forward pass: Y = XW (batched matrix-matrix multiply)
 *   cublasSgemm  — backward pass: dW = Xᵀδ (weight gradients)
 *   cublasSgemm  — backward pass: δ_prev = δW ᵀ (propagate error)
 *   cublasSgeam  — weight update: W = W - lr*dW (matrix addition with scaling)
 *
 * This is the GEMM equivalent of the gemv+ger pipeline from Level 2.
 * The key upgrade: processing a BATCH of samples in one GEMM call
 * instead of one sample at a time with gemv.
 *
 * Compile: nvcc -o pipeline 09_pipeline.cu -lcublas -lm
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

void print_matrix(const char *label, float *A, int rows, int cols) {
    printf("  %s (%dx%d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("    [");
        for (int j = 0; j < cols; j++)
            printf(" %8.4f", A[j * rows + i]);
        printf(" ]\n");
    }
}

/* ReLU on CPU — applied to hidden layer output */
void relu(float *A, int size) {
    for (int i = 0; i < size; i++)
        A[i] = A[i] > 0.0f ? A[i] : 0.0f;
}

/* ReLU gradient mask */
void relu_grad(float *delta, float *forward, int size) {
    for (int i = 0; i < size; i++)
        delta[i] = forward[i] > 0.0f ? delta[i] : 0.0f;
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    /* Network dimensions */
    int batch  = 4;   /* samples processed together */
    int in_dim = 3;   /* input features */
    int h_dim  = 4;   /* hidden layer size */
    int out_dim = 2;  /* output size */
    float lr   = 0.05f;

    /* Input X: batch×in_dim (stored col-major: in_dim rows, batch cols)
     * Each column is one sample.
     * X = [ 1  0  1  0 ]   (in_dim=3, batch=4)
     *     [ 0  1  1  0 ]
     *     [ 1  1  0  1 ]
     */
    float h_X[] = {
        1.0f, 0.0f, 1.0f,   /* col 0 (sample 0) */
        0.0f, 1.0f, 1.0f,   /* col 1 (sample 1) */
        1.0f, 1.0f, 0.0f,   /* col 2 (sample 2) */
        0.0f, 0.0f, 1.0f    /* col 3 (sample 3) */
    };

    /* Targets: out_dim × batch */
    float h_Target[] = {
        1.0f, 0.0f,   /* col 0 */
        0.0f, 1.0f,   /* col 1 */
        1.0f, 0.0f,   /* col 2 */
        0.0f, 0.0f    /* col 3 */
    };

    /* W1: in_dim × h_dim (col-major: in_dim rows, h_dim cols) */
    float h_W1[] = {
        0.1f,-0.1f, 0.2f,
        0.3f, 0.1f,-0.2f,
       -0.1f, 0.2f, 0.1f,
        0.2f,-0.3f, 0.1f
    };

    /* W2: h_dim × out_dim */
    float h_W2[] = {
        0.1f, 0.2f,-0.1f, 0.3f,
       -0.2f, 0.1f, 0.3f,-0.1f
    };

    printf("=== 2-Layer Neural Network Training (Level 3 BLAS) ===\n");
    printf("  Architecture: %d → %d (ReLU) → %d\n", in_dim, h_dim, out_dim);
    printf("  Batch size: %d  |  Learning rate: %.3f\n\n", batch, lr);

    /* Device allocations */
    float *d_X, *d_W1, *d_W2, *d_Target;
    float *d_H, *d_H_relu, *d_Y;
    float *d_dY, *d_dH, *d_dW1, *d_dW2;
    float *d_XT, *d_H_reluT;

    cudaMalloc((void**)&d_X,       in_dim * batch  * sizeof(float));
    cudaMalloc((void**)&d_W1,      in_dim * h_dim  * sizeof(float));
    cudaMalloc((void**)&d_W2,      h_dim  * out_dim* sizeof(float));
    cudaMalloc((void**)&d_Target,  out_dim* batch  * sizeof(float));
    cudaMalloc((void**)&d_H,       h_dim  * batch  * sizeof(float));
    cudaMalloc((void**)&d_H_relu,  h_dim  * batch  * sizeof(float));
    cudaMalloc((void**)&d_Y,       out_dim* batch  * sizeof(float));
    cudaMalloc((void**)&d_dY,      out_dim* batch  * sizeof(float));
    cudaMalloc((void**)&d_dH,      h_dim  * batch  * sizeof(float));
    cudaMalloc((void**)&d_dW1,     in_dim * h_dim  * sizeof(float));
    cudaMalloc((void**)&d_dW2,     h_dim  * out_dim* sizeof(float));
    cudaMalloc((void**)&d_XT,      batch  * in_dim * sizeof(float));
    cudaMalloc((void**)&d_H_reluT, batch  * h_dim  * sizeof(float));

    cublasSetMatrix(in_dim, batch,   sizeof(float), h_X,      in_dim,  d_X,      in_dim);
    cublasSetMatrix(in_dim, h_dim,   sizeof(float), h_W1,     in_dim,  d_W1,     in_dim);
    cublasSetMatrix(h_dim,  out_dim, sizeof(float), h_W2,     h_dim,   d_W2,     h_dim);
    cublasSetMatrix(out_dim,batch,   sizeof(float), h_Target, out_dim, d_Target, out_dim);

    float one=1.0f, zero=0.0f, neg_lr=-lr;

    /* Training loop */
    for (int iter = 0; iter < 20; iter++) {

        /* ============================================================
         * FORWARD PASS
         * ============================================================ */

        /* Step 1: H = W1ᵀ × X
         *
         * W1 is in_dim × h_dim (lda = in_dim).
         * X  is in_dim × batch.
         * H  is h_dim  × batch.
         *
         * We want H = W1ᵀ × X, so we use OP_T on W1.
         *
         * cublasSgemm: C = α op(A) op(B) + β C
         *   op(W1) = W1ᵀ  → m=h_dim, k=in_dim, lda=in_dim
         *   op(X)  = X    → n=batch,  k=in_dim, ldb=in_dim
         */
        CHECK_CUBLAS(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 h_dim, batch, in_dim,
                                 &one,
                                 d_W1, in_dim,
                                 d_X,  in_dim,
                                 &zero,
                                 d_H, h_dim));

        /* Step 2: ReLU activation on CPU */
        float h_H[16], h_H_relu[16];
        cublasGetMatrix(h_dim,batch,sizeof(float),d_H,h_dim,h_H,h_dim);
        for (int i=0;i<h_dim*batch;i++) h_H_relu[i]=h_H[i];
        relu(h_H_relu, h_dim * batch);
        cublasSetMatrix(h_dim,batch,sizeof(float),h_H_relu,h_dim,d_H_relu,h_dim);

        /* Step 3: Y = W2ᵀ × H_relu */
        CHECK_CUBLAS(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 out_dim, batch, h_dim,
                                 &one,
                                 d_W2,     h_dim,
                                 d_H_relu, h_dim,
                                 &zero,
                                 d_Y, out_dim));

        /* Compute loss on CPU */
        float h_Y[8], h_dY[8];
        cublasGetMatrix(out_dim,batch,sizeof(float),d_Y,out_dim,h_Y,out_dim);

        float loss = 0.0f;
        for (int i=0;i<out_dim*batch;i++) {
            h_dY[i] = (h_Y[i] - h_Target[i]) / batch;
            loss += 0.5f * h_dY[i] * h_dY[i];
        }

        if (iter==0 || (iter+1)%5==0)
            printf("  Iter %2d | loss = %.6f\n", iter+1, loss);

        /* ============================================================
         * BACKWARD PASS
         * ============================================================ */
        cublasSetMatrix(out_dim,batch,sizeof(float),h_dY,out_dim,d_dY,out_dim);

        /* Step 4: dW2 = H_reluᵀ × dYᵀ ... equivalently dW2 = H_relu × dYᵀ
         *
         * dW2 is h_dim × out_dim.
         * We compute dW2 = H_relu × dYᵀ
         * H_relu: h_dim × batch
         * dY:     out_dim × batch → dYᵀ: batch × out_dim
         */
        CHECK_CUBLAS(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 h_dim, out_dim, batch,
                                 &one,
                                 d_H_relu, h_dim,
                                 d_dY,     out_dim,
                                 &zero,
                                 d_dW2, h_dim));

        /* Step 5: Propagate error to hidden layer
         * dH = W2 × dY
         * W2:  h_dim × out_dim
         * dY:  out_dim × batch
         * dH:  h_dim × batch
         */
        CHECK_CUBLAS(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 h_dim, batch, out_dim,
                                 &one,
                                 d_W2,  h_dim,
                                 d_dY,  out_dim,
                                 &zero,
                                 d_dH, h_dim));

        /* Step 6: Apply ReLU gradient mask on CPU */
        float h_dH[16];
        cublasGetMatrix(h_dim,batch,sizeof(float),d_dH,h_dim,h_dH,h_dim);
        relu_grad(h_dH, h_H, h_dim*batch);
        cublasSetMatrix(h_dim,batch,sizeof(float),h_dH,h_dim,d_dH,h_dim);

        /* Step 7: dW1 = X × dHᵀ
         * dW1: in_dim × h_dim
         * X:   in_dim × batch
         * dH:  h_dim × batch → dHᵀ: batch × h_dim
         */
        CHECK_CUBLAS(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 in_dim, h_dim, batch,
                                 &one,
                                 d_X,   in_dim,
                                 d_dH,  h_dim,
                                 &zero,
                                 d_dW1, in_dim));

        /* ============================================================
         * WEIGHT UPDATE using cublasSgeam: W = W + (-lr) * dW
         * ============================================================ */

        /* W1 = 1*W1 + (-lr)*dW1 */
        CHECK_CUBLAS(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 in_dim, h_dim,
                                 &one,    d_W1,  in_dim,
                                 &neg_lr, d_dW1, in_dim,
                                 d_W1, in_dim));

        /* W2 = 1*W2 + (-lr)*dW2 */
        CHECK_CUBLAS(cublasSgeam(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 h_dim, out_dim,
                                 &one,    d_W2,  h_dim,
                                 &neg_lr, d_dW2, h_dim,
                                 d_W2, h_dim));
    }

    printf("\n");

    /* Download final weights */
    cublasGetMatrix(in_dim, h_dim,  sizeof(float), d_W1, in_dim, h_W1, in_dim);
    cublasGetMatrix(h_dim,  out_dim,sizeof(float), d_W2, h_dim,  h_W2, h_dim);
    print_matrix("Final W1", h_W1, in_dim, h_dim);
    print_matrix("Final W2", h_W2, h_dim,  out_dim);

    printf("\n  cuBLAS functions used per iteration:\n");
    printf("    cublasSgemm  — forward W1ᵀX, W2ᵀH, backward dW2, dH, dW1\n");
    printf("    cublasSgeam  — weight update W1 and W2\n");
    printf("  Total: 5 × cublasSgemm + 2 × cublasSgeam per iteration\n");

    cudaFree(d_X); cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_Target);
    cudaFree(d_H); cudaFree(d_H_relu); cudaFree(d_Y);
    cudaFree(d_dY); cudaFree(d_dH); cudaFree(d_dW1); cudaFree(d_dW2);
    cudaFree(d_XT); cudaFree(d_H_reluT);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * === 2-Layer Neural Network Training (Level 3 BLAS) ===
 *   Architecture: 3 → 4 (ReLU) → 2
 *   Batch size: 4  |  Learning rate: 0.050
 *
 *   Iter  1 | loss = 0.0xxxxx
 *   Iter  5 | loss = 0.0xxxxx  (decreasing)
 *   Iter 10 | loss = 0.0xxxxx  (decreasing)
 *   Iter 15 | loss = 0.0xxxxx  (decreasing)
 *   Iter 20 | loss = 0.0xxxxx  (decreasing)
 *
 *   cuBLAS functions used per iteration:
 *     cublasSgemm — forward W1ᵀX, W2ᵀH, backward dW2, dH, dW1
 *     cublasSgeam — weight update W1 and W2
 *   Total: 5 × cublasSgemm + 2 × cublasSgeam per iteration
 *
 * KEY DIFFERENCE FROM LEVEL 2 PIPELINE:
 * Level 2 used gemv (matrix × vector) — one sample at a time.
 * Level 3 uses gemm (matrix × matrix) — entire batch at once.
 * Processing 4 samples takes 1 GEMM call instead of 4 GEMV calls.
 * At batch=1000, this is 1000x fewer kernel launches.
 */
