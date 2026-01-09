#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void kernel_code()
{
    int c=0;
    if(threadIdx.x ==0) printf("Kernel code started \n");
    for(int i = 0; i<100000000; i++) c = 0;
    if(threadIdx.x ==0) printf("Kernel code completed \n");
}

void gold_code()
{
    printf("gold code started \n");
    
    printf("gold code completed \n");
}

int main()
{
    const int N = 1024;
    const size_t bytes = N * sizeof(int);

    // Host memory
    int *h_u1 = (int*)malloc(bytes);
    int *h_u2 = (int*)malloc(bytes);
    int *h_ref = (int*)malloc(bytes);

    for (int i = 0; i < N; i++)
        h_u1[i] = i;

    // Device memory
    int *d_u1, *d_u2;
    cudaMalloc(&d_u1, bytes);
    cudaMalloc(&d_u2, bytes);

    // H2D copy (blocking on default stream)
    printf("Memory1 start \n");
    cudaMemcpy(d_u1, h_u1, bytes, cudaMemcpyHostToDevice);
    printf("Memory1 end \n");
    // Kernel launch (async)
    kernel_code<<<1,4>>>();

    // CPU reference
    gold_code();

    // D2H copy (implicit sync)
    printf("Memory2 start \n");
    cudaMemcpy(h_u2, d_u2, bytes, cudaMemcpyDeviceToHost);
    printf("Memory2 end \n");
    
    printf("Success \n");

    cudaFree(d_u1);
    cudaFree(d_u2);
    free(h_u1);
    free(h_u2);
    free(h_ref);

    return 0;
}
