/*
 * Capstone: Train 4 independent networks in parallel
 * using gemmStridedBatched for forward and backward passes.
 *
 * Each network: input(3) → output(2), no hidden layer, no activation.
 * Loss: MSE.  Update: gradient descent via cublasSgeam.
 *
 * Forward:   Y[t] = W[t]ᵀ * X[t]           — 1 strided batched call
 * Backward:  dW[t] = X[t] * δ[t]ᵀ          — 1 strided batched call
 * Update:    W[t] -= lr * dW[t]             — geam per task (no batched geam)
 *
 * Compile: nvcc -o batched_train 05_batched_training.cu -lcublas -lm
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

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int T=4, in=3, out=2, bs=6;
    float lr = 0.05f;

    /* strides */
    long long sW=in*out, sX=in*bs, sY=out*bs;

    float *h_W  = (float*)calloc(T*sW, sizeof(float));
    float *h_X  = (float*)malloc(T*sX*sizeof(float));
    float *h_Tgt= (float*)malloc(T*sY*sizeof(float));

    /* different starting weights per task */
    for (int t=0;t<T;t++)
        for (int i=0;i<sW;i++)
            h_W[t*sW+i] = 0.1f*(t+1)*((i%5)-2);

    /* different inputs per task */
    for (int t=0;t<T;t++)
        for (int i=0;i<sX;i++)
            h_X[t*sX+i] = ((t*sX+i)%5-2)*0.3f;

    /* different targets per task */
    for (int t=0;t<T;t++)
        for (int i=0;i<sY;i++)
            h_Tgt[t*sY+i] = (i%out==t%out) ? 1.0f : -1.0f;

    float *d_W, *d_X, *d_Y, *d_dW, *d_dY;
    cudaMalloc((void**)&d_W,  T*sW*sizeof(float));
    cudaMalloc((void**)&d_X,  T*sX*sizeof(float));
    cudaMalloc((void**)&d_Y,  T*sY*sizeof(float));
    cudaMalloc((void**)&d_dW, T*sW*sizeof(float));
    cudaMalloc((void**)&d_dY, T*sY*sizeof(float));

    cudaMemcpy(d_W, h_W, T*sW*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, T*sX*sizeof(float), cudaMemcpyHostToDevice);

    float one=1.0f, zero=0.0f, neg_lr=-lr;

    for (int iter=0; iter<20; iter++) {

        /* Forward: Y[t] = W[t]ᵀ * X[t]  for all T tasks */
        cudaMemset(d_Y, 0, T*sY*sizeof(float));
        CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            out, bs, in,
            &one,
            d_W, in, sW,
            d_X, in, sX,
            &zero,
            d_Y, out, sY,
            T));

        /* Compute δ on CPU */
        float *h_Y  = (float*)malloc(T*sY*sizeof(float));
        float *h_dY = (float*)malloc(T*sY*sizeof(float));
        cudaMemcpy(h_Y, d_Y, T*sY*sizeof(float), cudaMemcpyDeviceToHost);

        float loss=0;
        for (int t=0;t<T;t++)
            for (int i=0;i<sY;i++) {
                float d = h_Y[t*sY+i]-h_Tgt[t*sY+i];
                h_dY[t*sY+i] = d/bs;
                loss += 0.5f*d*d;
            }

        if (iter==0||(iter+1)%5==0)
            printf("  Iter %2d | loss = %.4f\n", iter+1, loss);

        /* Backward: dW[t] = X[t] * δ[t]ᵀ  for all T tasks */
        cudaMemcpy(d_dY, h_dY, T*sY*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_dW, 0, T*sW*sizeof(float));
        CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            in, out, bs,
            &one,
            d_X,  in,  sX,
            d_dY, out, sY,
            &zero,
            d_dW, in,  sW,
            T));

        /* Update: W[t] -= lr * dW[t]  — geam has no batched variant */
        for (int t=0;t<T;t++)
            cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        in, out,
                        &one,    d_W  +t*sW, in,
                        &neg_lr, d_dW +t*sW, in,
                        d_W+t*sW, in);

        free(h_Y); free(h_dY);
    }

    printf("\n  cuBLAS calls per iteration:\n");
    printf("    1 × gemmStridedBatched  — forward  (all %d tasks)\n", T);
    printf("    1 × gemmStridedBatched  — backward (all %d tasks)\n", T);
    printf("    %d × cublasSgeam        — weight update (no batched geam)\n", T);
    printf("    Without batching: %d gemm calls needed instead of 2\n", 2*T);

    free(h_W); free(h_X); free(h_Tgt);
    cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
    cudaFree(d_dW); cudaFree(d_dY);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 *   Iter  1 | loss = X.XXXX
 *   Iter  5 | loss = X.XXXX  (decreasing)
 *   Iter 10 | loss = X.XXXX  (decreasing)
 *   Iter 15 | loss = X.XXXX  (decreasing)
 *   Iter 20 | loss = X.XXXX  (decreasing)
 *
 *   cuBLAS calls per iteration:
 *     1 × gemmStridedBatched — forward  (all 4 tasks)
 *     1 × gemmStridedBatched — backward (all 4 tasks)
 *     4 × cublasSgeam        — weight update (no batched geam)
 *     Without batching: 8 gemm calls needed instead of 2
 *
 * KEY POINTS:
 * Loss decreases — all 4 networks are learning simultaneously.
 * Each network sees different data and has different weights.
 * Forward + backward each use ONE batched call for all T=4 tasks.
 * cublasSgeam has no batched equivalent — weight updates still loop.
 * This is a real cuBLAS limitation worth remembering.
 */
