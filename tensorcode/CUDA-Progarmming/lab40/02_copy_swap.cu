#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUBLAS(call)                                              
    {                                                                   
        cublasStatus_t _err = (call);                                   
        if (_err != CUBLAS_STATUS_SUCCESS) {                            
            printf("cuBLAS error %d at line %d\n", _err, __LINE__);    
            exit(EXIT_FAILURE);                                         
        }                                                               
    }

void print_vector(const char *label, float *v, int n) {
    printf("%-12s: ", label);
    for (int i = 0; i < n; i++) printf("%5.1f ", v[i]);
    printf("\n");
}

int main() {

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int n = 6;

    float h_x[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    float h_y[] = { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f};

    printf("=== cublasScopy Demo ===\n");
    print_vector("Before: x", h_x, n);
    print_vector("Before: y", h_y, n);

    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_x, 1, d_x, 1));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_y, 1, d_y, 1));

    CHECK_CUBLAS(cublasScopy(handle, n, d_x, 1, d_y, 1));

    CHECK_CUBLAS(cublasGetVector(n, sizeof(float), d_x, 1, h_x, 1));
    CHECK_CUBLAS(cublasGetVector(n, sizeof(float), d_y, 1, h_y, 1));

    printf("After copy:\n");
    print_vector("x (unchanged)", h_x, n);
    print_vector("y (copy of x)", h_y, n);

    printf("\n=== cublasSswap Demo ===\n");

    float h_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float h_b[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    print_vector("Before: a", h_a, n);
    print_vector("Before: b", h_b, n);

    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_a, 1, d_x, 1));
    CHECK_CUBLAS(cublasSetVector(n, sizeof(float), h_b, 1, d_y, 1));

    CHECK_CUBLAS(cublasSswap(handle, 3, d_x, 2, d_y, 2));

    CHECK_CUBLAS(cublasGetVector(n, sizeof(float), d_x, 1, h_a, 1));
    CHECK_CUBLAS(cublasGetVector(n, sizeof(float), d_y, 1, h_b, 1));

    printf("After swap (stride=2, 3 elements swapped):\n");
    print_vector("a", h_a, n);
    print_vector("b", h_b, n);
    printf("  Note: odd-indexed elements {2.0,4.0,6.0} and {8.0,10.0,12.0} are unchanged.\n");

    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);

    return 0;
}
