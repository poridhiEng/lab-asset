

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

    /* Three pairs of 2x2 matrices (column-major) */
    float h_A[3][4] = {
        { 1, 2, 3, 4 },          /* A[0] = [[1,3],[2,4]] */
        { 5, 6, 7, 8 },          /* A[1] = [[5,7],[6,8]] */
        { 1, 0, 0, 1 }           /* A[2] = identity      */
    };
    float h_B[3][4] = {
        { 1, 0, 0, 1 },          /* B[0] = identity  */
        { 2, 2, 2, 2 },          /* B[1] = all twos  */
        { 3, 4, 5, 6 }           /* B[2]             */
    };
    float h_C[3][4] = { {0},{0},{0} };

    /* Each matrix gets its own allocation */
    float *d_A[3], *d_B[3], *d_C[3];
    for (int i = 0; i < batch; i++) {
        cudaMalloc((void**)&d_A[i], m*k*sizeof(float));
        cudaMalloc((void**)&d_B[i], k*n*sizeof(float));
        cudaMalloc((void**)&d_C[i], m*n*sizeof(float));
        cublasSetMatrix(m,k,sizeof(float),h_A[i],m,d_A[i],m);
        cublasSetMatrix(k,n,sizeof(float),h_B[i],k,d_B[i],k);
        cublasSetMatrix(m,n,sizeof(float),h_C[i],m,d_C[i],m);
    }

    /* Build device array of device pointers */
    float **d_Aarray, **d_Barray, **d_Carray;
    cudaMalloc((void**)&d_Aarray, batch*sizeof(float*));
    cudaMalloc((void**)&d_Barray, batch*sizeof(float*));
    cudaMalloc((void**)&d_Carray, batch*sizeof(float*));
    cudaMemcpy(d_Aarray, d_A, batch*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Barray, d_B, batch*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Carray, d_C, batch*sizeof(float*), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUBLAS(cublasSgemmBatched(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, n, k,
                                    &alpha,
                                    (const float**)d_Aarray, m,
                                    (const float**)d_Barray, k,
                                    &beta,
                                    d_Carray, m,
                                    batch));

    for (int i = 0; i < batch; i++) {
        cublasGetMatrix(m,n,sizeof(float),d_C[i],m,h_C[i],m);
        printf("  C[%d] = A[%d] * B[%d]:\n", i, i, i);
        print_matrix(h_C[i], m, n);
    }

    for (int i = 0; i < batch; i++) {
        cudaFree(d_A[i]); cudaFree(d_B[i]); cudaFree(d_C[i]);
    }
    cudaFree(d_Aarray); cudaFree(d_Barray); cudaFree(d_Carray);
    cublasDestroy(handle);
    return 0;
}
