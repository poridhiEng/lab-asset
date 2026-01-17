#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>

#define N 256


__global__ void tiny_default()
{
    printf("[default stream] thread %d\n", threadIdx.x);
}


__global__ void worker(float *data, int cpu_thread_id)
{
    int tid = threadIdx.x;


    if (tid == 0)
        printf("[worker %d] block start\n", cpu_thread_id);


    // example work
    for (int i = 0; i < 10000000; i++) {
        data[tid % N] += 1.0f;
    }


    if (tid == 0)
        printf("[worker %d] block end\n", cpu_thread_id);
}


int main()
{
    // Correct OpenMP function
    omp_set_num_threads(4);


    float *data[4];
    for (int i = 0; i < 4; i++)
        cudaMalloc(&data[i], N * sizeof(float));


    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        printf("CPU thread ID = %d\n", omp_get_thread_num());
        int cpu_thread_id = omp_get_thread_num();
        // Launch the GPU kernel from each CPU thread
        worker<<<1, 64>>>(data[i], cpu_thread_id);
    }


    cudaDeviceSynchronize();


    return 0;
}
