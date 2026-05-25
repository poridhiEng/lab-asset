/*
 * Performance Profiling — GFLOPS, Bandwidth, Efficiency
 *
 * Metrics:
 *   GFLOPS    = (2 × M × N × K) / (time_s × 10⁹)
 *   Bandwidth = bytes_transferred / (time_s × 10⁹)  GB/s
 *   Efficiency = achieved_GFLOPS / peak_GFLOPS × 100%
 *
 * Peak GFLOPS is read from device properties.
 * Peak bandwidth is also read from device properties.
 *
 * Runs GEMM at multiple matrix sizes to show how efficiency
 * scales — small matrices are memory-bound, large ones are
 * compute-bound and approach theoretical peak.
 *
 * Compile: nvcc -O3 -o perf 05_perf.cu -lcublas -lm
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

/* Time a single GEMM with warmup, return ms */
float time_gemm(cublasHandle_t handle, float *d_A, float *d_B, float *d_C,
                int M, int N, int K) {
    float alpha=1.0f, beta=0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    /* warmup */
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M,N,K, &alpha, d_A,M, d_B,K, &beta, d_C,M);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M,N,K, &alpha, d_A,M, d_B,K, &beta, d_C,M));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    /* Get device properties for peak metrics */
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    /* Peak FP32 GFLOPS = 2 × SM_count × cores_per_SM × clock_GHz */
    double peak_gflops = 2.0 * prop.multiProcessorCount
                       * 128          /* approximate cores per SM */
                       * prop.clockRate * 1e-6;   /* MHz to GHz */

    /* Peak bandwidth = memory clock × bus width / 8 × 2 (DDR) */
    double peak_bw = (double)prop.memoryClockRate * 1e3
                   * prop.memoryBusWidth / 8.0 * 2.0 / 1e9;

    printf("Device: %s\n", prop.name);
    printf("Peak FP32 GFLOPS (estimated): %.0f\n", peak_gflops);
    printf("Peak memory bandwidth:        %.0f GB/s\n\n", peak_bw);
    printf("  %-6s  %8s  %10s  %10s  %10s  %8s\n",
           "Size", "Time(ms)", "GFLOPS", "BW(GB/s)", "Efficiency", "Bound");
    printf("  %-6s  %8s  %10s  %10s  %10s  %8s\n",
           "------", "--------", "----------",
           "----------", "----------", "--------");

    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int nsizes  = 7;

    int max_n = 4096;
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, (size_t)max_n*max_n*sizeof(float));
    cudaMalloc((void**)&d_B, (size_t)max_n*max_n*sizeof(float));
    cudaMalloc((void**)&d_C, (size_t)max_n*max_n*sizeof(float));
    cudaMemset(d_A, 1, (size_t)max_n*max_n*sizeof(float));
    cudaMemset(d_B, 1, (size_t)max_n*max_n*sizeof(float));

    for (int s = 0; s < nsizes; s++) {
        int n = sizes[s];

        float ms = time_gemm(handle, d_A, d_B, d_C, n, n, n);
        double time_s = ms * 1e-3;

        /* GFLOPS: 2*n³ floating point operations */
        double flops     = 2.0 * n * n * n;
        double gflops    = flops / (time_s * 1e9);

        /* Bandwidth: read A + B, write C = 3 matrices */
        double bytes     = 3.0 * n * n * sizeof(float);
        double bw        = bytes / (time_s * 1e9);

        /* Efficiency vs peak */
        double efficiency = gflops / peak_gflops * 100.0;

        /* Roofline: is this kernel compute or memory bound?
         * Arithmetic intensity = FLOPs / bytes
         * If intensity > peak_gflops/peak_bw → compute bound
         *                   else              → memory bound
         */
        double intensity = flops / bytes;
        double ridge     = peak_gflops / peak_bw;
        const char *bound = intensity > ridge ? "compute" : "memory";

        printf("  %-6d  %8.3f  %10.1f  %10.1f  %9.1f%%  %8s\n",
               n, ms, gflops, bw, efficiency, bound);
    }

    printf("\n  Arithmetic intensity = 2*n³ / (3*n²*4) = n/6  (grows with n)\n");
    printf("  Small n: memory bound — not enough work per byte fetched.\n");
    printf("  Large n: compute bound — approaches theoretical peak GFLOPS.\n");

    /* ----------------------------------------------------------------
     * Roofline summary
     * ---------------------------------------------------------------- */
    printf("\n  Roofline model:\n");
    printf("    Ridge point = peak_GFLOPS / peak_BW = %.1f FLOP/byte\n",
           peak_gflops / peak_bw);
    printf("    n=64:  intensity ~ %.1f  FLOP/byte  (well below ridge)\n", 64.0/6.0);
    printf("    n=4096: intensity ~ %.1f FLOP/byte  (above ridge)\n",   4096.0/6.0);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}

/*
 * EXPECTED OUTPUT (varies by GPU):
 * ----------------------------------
 * Device: NVIDIA ...
 * Peak FP32 GFLOPS: ~10000
 * Peak bandwidth:   ~900 GB/s
 *
 *   Size    Time(ms)      GFLOPS    BW(GB/s)  Efficiency     Bound
 *   ------  --------  ----------  ----------  ----------  --------
 *      64      0.020         3.3       247.0        0.0%    memory
 *     128      0.022        19.0       356.0        0.2%    memory
 *     256      0.030       112.0       524.0        1.1%    memory
 *     512      0.060       450.0       704.0        4.5%    memory
 *    1024      0.300      ~7000       ~820.0       70.0%   compute
 *    2048      1.800      ~9000       ~840.0       90.0%   compute
 *    4096     13.000      ~9400       ~850.0       94.0%   compute
 *
 * KEY INSIGHTS:
 * Small matrices are memory bound — the GPU fetches data faster
 * than it can compute. Efficiency is low because the kernel
 * spends most time waiting for memory.
 *
 * Large matrices are compute bound — the GPU is saturating
 * its FP32 units. Efficiency approaches 90%+ of peak GFLOPS.
 *
 * The crossover (ridge point) depends on the GPU's ratio of
 * compute power to memory bandwidth.
 */
