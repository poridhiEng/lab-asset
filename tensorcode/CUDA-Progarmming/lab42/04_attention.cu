/*
 * Real-world use: Multi-Head Attention
 *
 * Attention per head:  scores = Q * Kᵀ / sqrt(d_k)
 *                      output = softmax(scores) * V
 *
 * H heads → H independent GEMMs → perfect for gemmStridedBatched.
 * Two batched calls replace what would be 2*H sequential calls.
 *
 * Shapes per head (col-major):
 *   Q, K : seq_len × d_k
 *   V    : seq_len × d_v
 *   scores: seq_len × seq_len
 *   output: seq_len × d_v
 *
 * Compile: nvcc -o attention 04_attention.cu -lcublas -lm
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

/* Row-wise softmax on CPU */
void softmax(float *A, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float mx = A[i*cols];
        for (int j = 1; j < cols; j++) if (A[i*cols+j] > mx) mx = A[i*cols+j];
        float s = 0;
        for (int j = 0; j < cols; j++) { A[i*cols+j] = expf(A[i*cols+j]-mx); s += A[i*cols+j]; }
        for (int j = 0; j < cols; j++) A[i*cols+j] /= s;
    }
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int H = 4, seq = 6, d_k = 8, d_v = 8;

    printf("=== Multi-Head Attention ===\n");
    printf("  H=%d heads  seq=%d  d_k=%d  d_v=%d\n\n", H, seq, d_k, d_v);

    long long sQ = seq*d_k, sK = seq*d_k, sV = seq*d_v;
    long long sS = seq*seq, sO = seq*d_v;

    float *h_Q = (float*)malloc(H*sQ*sizeof(float));
    float *h_K = (float*)malloc(H*sK*sizeof(float));
    float *h_V = (float*)malloc(H*sV*sizeof(float));
    float *h_S = (float*)malloc(H*sS*sizeof(float));
    float *h_O = (float*)malloc(H*sO*sizeof(float));

    for (int i=0;i<H*sQ;i++) h_Q[i]=(i%7-3)*0.1f;
    for (int i=0;i<H*sK;i++) h_K[i]=(i%5-2)*0.1f;
    for (int i=0;i<H*sV;i++) h_V[i]=(i%6-1)*0.2f;

    float *d_Q,*d_K,*d_V,*d_S,*d_O;
    cudaMalloc((void**)&d_Q,H*sQ*sizeof(float));
    cudaMalloc((void**)&d_K,H*sK*sizeof(float));
    cudaMalloc((void**)&d_V,H*sV*sizeof(float));
    cudaMalloc((void**)&d_S,H*sS*sizeof(float));
    cudaMalloc((void**)&d_O,H*sO*sizeof(float));
    cudaMemcpy(d_Q,h_Q,H*sQ*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,h_K,H*sK*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,h_V,H*sV*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_S,0,H*sS*sizeof(float));
    cudaMemset(d_O,0,H*sO*sizeof(float));

    float scale = 1.0f/sqrtf((float)d_k), zero=0.0f;

    /* Step 1: scores[h] = Q[h] * K[h]ᵀ / sqrt(d_k)  — all H heads, 1 call */
    printf("Step 1: scores = Q * Kᵀ / sqrt(d_k)  [%d heads in 1 call]\n", H);
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        seq, seq, d_k,
        &scale,
        d_Q, seq, sQ,
        d_K, seq, sK,
        &zero,
        d_S, seq, sS,
        H));

    /* Step 2: softmax on CPU */
    printf("Step 2: softmax(scores)  [CPU]\n");
    cudaMemcpy(h_S,d_S,H*sS*sizeof(float),cudaMemcpyDeviceToHost);
    for (int h=0;h<H;h++) softmax(h_S+h*sS, seq, seq);
    cudaMemcpy(d_S,h_S,H*sS*sizeof(float),cudaMemcpyHostToDevice);

    /* Step 3: output[h] = weights[h] * V[h]  — all H heads, 1 call */
    printf("Step 3: output = weights * V  [%d heads in 1 call]\n\n", H);
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        seq, d_v, seq,
        &scale,
        d_S, seq, sS,
        d_V, seq, sV,
        &zero,
        d_O, seq, sO,
        H));

    cudaMemcpy(h_O,d_O,H*sO*sizeof(float),cudaMemcpyDeviceToHost);

    printf("  Output head 0  (first 3 tokens, first 4 dims):\n");
    for (int t=0;t<3;t++) {
        printf("    token %d: [", t);
        for (int d=0;d<4;d++) printf(" %6.3f", h_O[d*seq+t]);
        printf(" ...]\n");
    }

    printf("\n  Sequential equivalent: %d gemm calls\n", 2*H);
    printf("  Batched:               2 gemmStridedBatched calls\n");

    free(h_Q);free(h_K);free(h_V);free(h_S);free(h_O);
    cudaFree(d_Q);cudaFree(d_K);cudaFree(d_V);cudaFree(d_S);cudaFree(d_O);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT:
 * ----------------
 * Step 1: scores = Q * Kᵀ / sqrt(d_k)  [4 heads in 1 call]
 * Step 2: softmax(scores)  [CPU]
 * Step 3: output = weights * V  [4 heads in 1 call]
 *
 *   Output head 0 (first 3 tokens, first 4 dims):
 *     token 0: [ x.xxx  x.xxx  x.xxx  x.xxx ...]
 *     token 1: [ x.xxx  x.xxx  x.xxx  x.xxx ...]
 *     token 2: [ x.xxx  x.xxx  x.xxx  x.xxx ...]
 *
 *   Sequential equivalent: 8 gemm calls
 *   Batched:               2 gemmStridedBatched calls
 *
 * KEY INSIGHT:
 * H independent attention heads = H independent GEMMs.
 * gemmStridedBatched computes all H in one kernel launch.
 * This is how transformer libraries use cuBLAS internally.
 */
