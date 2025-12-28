# Lab 28: Warp-Level Optimization

## Introduction

In Lab 27, we optimized our parallel reduction by performing the first add during the global memory load, achieving a 2.99x speedup over the original divergent approach. However, there's still significant instruction overhead in the reduction loop that we can eliminate.

In this lab, we'll introduce **warp-level optimization** by unrolling the last warp of the reduction, eliminating unnecessary synchronization and branch instructions.

## The Problem: Instruction Overhead

Even with our previous optimizations, the kernel is not fully memory-bound. Looking at our Lab 27 performance:

| Metric | Value |
|--------|-------|
| Achieved bandwidth | 284.94 GB/s |
| Theoretical peak | ~432 GB/s |
| Efficiency | 65.9% of peak |

The remaining bottleneck is **instruction overhead**. In each iteration of the reduction loop, we have:

- **Conditional branch**: `if (tid < s)`
- **Synchronization**: `__syncthreads()`
- **Loop control**: Counter update and exit condition check
- **Address calculation**: Computing `tid + s`

These instructions consume GPU cycles without contributing to the actual reduction computation.

## Understanding Warp Execution

A **warp** is a group of 32 threads that execute instructions in lockstep (SIMD). This has important implications:

| Property | Implication |
|----------|-------------|
| All threads execute same instruction | No divergence within a warp doing the same operation |
| Implicit synchronization | Threads in a warp are always synchronized |
| Automatic masking | Inactive threads are automatically masked out |

### The Key Insight

When the reduction stride `s` becomes 32 or less, only a single warp (32 threads) remains active. At this point:

1. **`__syncthreads()` is unnecessary** - Threads in a warp are implicitly synchronized
2. **Conditional branches are redundant** - Warp-level masking handles inactive threads automatically
3. **The loop can be fully unrolled** - We know exactly how many iterations remain (6 iterations for strides 32, 16, 8, 4, 2, 1)

## The Solution: Unrolling the Last Warp

Instead of continuing the loop for the last 6 iterations, we write them out explicitly in a separate device function.

### The warpReduce Function

```cpp
__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
```

### Why `volatile` Is Critical

The `volatile` keyword is essential here. It tells the compiler:

| Without `volatile` | With `volatile` |
|-------------------|-----------------|
| Compiler may cache `sdata[tid]` in a register | Every read/write goes to shared memory |
| Other threads' updates may not be visible | All updates are immediately visible to all threads |
| **Incorrect results** | **Correct results** |

Without `volatile`, the compiler might optimize by keeping intermediate values in registers, preventing threads from seeing each other's updates.

### Warp Unrolling Visualization

```
Before (Loop with s = 32, 16, 8, 4, 2, 1):
┌─────────────────────────────────────────────────────┐
│ for (s = 32; s > 0; s >>= 1) {                      │
│     if (tid < s)                    <- Branch       │
│         sdata[tid] += sdata[tid+s]; <- Add          │
│     __syncthreads();                <- Sync         │
│ }                                   <- Loop control │
└─────────────────────────────────────────────────────┘
6 iterations × (branch + add + sync + loop) = High overhead

After (Unrolled warpReduce):
┌─────────────────────────────────────────────────────┐
│ sdata[tid] += sdata[tid + 32];  <- Just add         │
│ sdata[tid] += sdata[tid + 16];  <- Just add         │
│ sdata[tid] += sdata[tid + 8];   <- Just add         │
│ sdata[tid] += sdata[tid + 4];   <- Just add         │
│ sdata[tid] += sdata[tid + 2];   <- Just add         │
│ sdata[tid] += sdata[tid + 1];   <- Just add         │
└─────────────────────────────────────────────────────┘
6 additions only = Minimal overhead
```

### Step-by-Step Parallel Execution

This is the key to understanding warp-level reduction: **all 32 threads execute each instruction in parallel**, but only certain results are meaningful for the final sum. Let's trace through each time step.

**Initial State** (after the loop reduces to 64 elements in `sdata[0..63]`):

```
sdata: [s0, s1, s2, s3, ... s31, s32, s33, s34, ... s63]
        ↑________________________↑   ↑_____________________↑
              First 32 elements          Next 32 elements
```

---

**T=0: `sdata[tid] += sdata[tid + 32];`**

All 32 threads (tid = 0 to 31) execute this instruction **simultaneously**:

| Thread | Operation | Result |
|--------|-----------|--------|
| tid=0  | s[0] = s[0] + s[32] | ✓ Needed |
| tid=1  | s[1] = s[1] + s[33] | ✓ Needed |
| tid=2  | s[2] = s[2] + s[34] | ✓ Needed |
| ... | ... | ... |
| tid=31 | s[31] = s[31] + s[63] | ✓ Needed |

**All 32 results are meaningful** - we've reduced 64 elements to 32.

```
sdata: [s0+s32, s1+s33, s2+s34, ... s31+s63, s32, s33, ... s63]
        ↑_________________________________↑
              32 partial sums (needed)
```

---

**T=1: `sdata[tid] += sdata[tid + 16];`**

All 32 threads execute this instruction **simultaneously**:

| Thread | Operation | Result |
|--------|-----------|--------|
| tid=0  | s[0] = s[0] + s[16] | ✓ Needed |
| tid=1  | s[1] = s[1] + s[17] | ✓ Needed |
| ... | ... | ... |
| tid=15 | s[15] = s[15] + s[31] | ✓ Needed |
| tid=16 | s[16] = s[16] + s[32] | ✗ Not needed (but harmless) |
| tid=17 | s[17] = s[17] + s[33] | ✗ Not needed (but harmless) |
| ... | ... | ... |
| tid=31 | s[31] = s[31] + s[47] | ✗ Not needed (but harmless) |

**Only s[0..15] matter now** - threads 16-31 execute but their results are irrelevant.

```
sdata: [sum0, sum1, ... sum15, (garbage), (garbage), ...]
        ↑__________________↑
         16 partial sums (needed)
```

---

**T=2: `sdata[tid] += sdata[tid + 8];`**

All 32 threads execute **simultaneously**:

| Thread | Operation | Result |
|--------|-----------|--------|
| tid=0  | s[0] = s[0] + s[8] | ✓ Needed |
| tid=1  | s[1] = s[1] + s[9] | ✓ Needed |
| ... | ... | ... |
| tid=7  | s[7] = s[7] + s[15] | ✓ Needed |
| tid=8 to tid=31 | s[tid] = s[tid] + s[tid+8] | ✗ Not needed |

**Only s[0..7] matter now**.

```
sdata: [sum0, sum1, ... sum7, (garbage), ...]
        ↑________________↑
         8 partial sums (needed)
```

---

**T=3: `sdata[tid] += sdata[tid + 4];`**

All 32 threads execute **simultaneously**:

| Thread | Operation | Result |
|--------|-----------|--------|
| tid=0  | s[0] = s[0] + s[4] | ✓ Needed |
| tid=1  | s[1] = s[1] + s[5] | ✓ Needed |
| tid=2  | s[2] = s[2] + s[6] | ✓ Needed |
| tid=3  | s[3] = s[3] + s[7] | ✓ Needed |
| tid=4 to tid=31 | ... | ✗ Not needed |

**Only s[0..3] matter now**.

```
sdata: [sum0, sum1, sum2, sum3, (garbage), ...]
        ↑__________________↑
         4 partial sums (needed)
```

---

**T=4: `sdata[tid] += sdata[tid + 2];`**

All 32 threads execute **simultaneously**:

| Thread | Operation | Result |
|--------|-----------|--------|
| tid=0  | s[0] = s[0] + s[2] | ✓ Needed |
| tid=1  | s[1] = s[1] + s[3] | ✓ Needed |
| tid=2 to tid=31 | ... | ✗ Not needed |

**Only s[0..1] matter now**.

```
sdata: [sum0, sum1, (garbage), ...]
        ↑________↑
         2 partial sums (needed)
```

---

**T=5: `sdata[tid] += sdata[tid + 1];`**

All 32 threads execute **simultaneously**:

| Thread | Operation | Result |
|--------|-----------|--------|
| tid=0  | s[0] = s[0] + s[1] | ✓ **FINAL SUM** |
| tid=1 to tid=31 | ... | ✗ Not needed |

**Only s[0] matters** - this is the final reduction result!

```
sdata: [FINAL_SUM, (garbage), ...]
        ↑________↑
         The answer!
```

---

### Why This Works Without Branches

| Aspect | Explanation |
|--------|-------------|
| **All threads execute** | SIMT model requires all 32 threads to run the same instruction |
| **Extra work is harmless** | Threads 16-31 at T=1 compute s[16]+s[32], but we never read s[16] again |
| **No branch overhead** | Instead of `if (tid < 16)`, we let all threads compute and ignore unwanted results |
| **No sync needed** | All threads finish the instruction before moving to the next (warp-synchronous) |

### Visual Summary

```
T=0: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] + [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] → 32 sums
T=1: [■■■■■■■■■■■■■■■■ ················] + [■■■■■■■■■■■■■■■■ ················] → 16 sums
T=2: [■■■■■■■■ ························] + [■■■■■■■■ ························] → 8 sums
T=3: [■■■■ ····························] + [■■■■ ····························] → 4 sums
T=4: [■■ ······························] + [■■ ······························] → 2 sums
T=5: [■ ·······························] + [■ ·······························] → 1 sum

■ = meaningful computation
· = executes but result ignored
```

## Implementation

### The Complete Optimized Kernel

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 24)          // 16,777,216 elements
#define BLOCK_SIZE 1024

// Warp-level reduction for the last 32 threads
__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduceWarpUnroll(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + tid;

    // First add during global load
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // Reduction in shared memory (sequential addressing)
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level reduction for last 32 threads
    if (tid < 32)
        warpReduce(sdata, tid);

    // Write result for this block to global memory
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

int main() {
    // Host allocations
    int *h_idata = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        h_idata[i] = 1;

    int numBlocks = N / (BLOCK_SIZE * 2);
    int *h_odata = (int *)malloc(numBlocks * sizeof(int));

    // Device allocations
    int *d_idata, *d_odata;
    cudaMalloc(&d_idata, N * sizeof(int));
    cudaMalloc(&d_odata, numBlocks * sizeof(int));

    cudaMemcpy(d_idata, h_idata, N * sizeof(int), cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    reduceWarpUnroll<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    reduceWarpUnroll<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_idata, d_odata);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // Copy back partial sums
    cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    long long total = 0;
    for (int i = 0; i < numBlocks; i++)
        total += h_odata[i];

    printf("Reduction result: %lld (expected: %d)\n", total, N);
    printf("Kernel time: %.3f ms\n", kernel_ms);

    // Bandwidth calculation (global reads only)
    double bytes_moved = N * sizeof(int);
    double bandwidth_GBs = (bytes_moved / 1e9) / (kernel_ms / 1000.0);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_GBs);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}
```

### Key Code Changes from Lab 27

```cpp
// Lab 27: Loop continues all the way to s = 1
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}

// Lab 28: Loop stops at s = 32, then unrolls
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {  // Note: s > 32
    if (tid < s)
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}

// Warp-level reduction (no sync, no branches)
if (tid < 32)
    warpReduce(sdata, tid);
```

### Iteration Comparison

| Lab 27 | Lab 28 |
|--------|--------|
| Loop: s = 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 | Loop: s = 512, 256, 128, 64 |
| 10 iterations with sync + branch | 4 iterations with sync + branch |
| - | warpReduce: 6 unrolled additions |

## Output

![Output](https://raw.githubusercontent.com/poridhiEng/lab-asset/refs/heads/main/tensorcode/CUDA-Progarmming/lab28/output.png)

## Performance Comparison

| Version | Kernel Time | Bandwidth | Description |
|---------|-------------|-----------|-------------|
| Divergent Interleaved (Lab 25) | 0.706 ms | 95.12 GB/s | Scattered threads cause warp divergence |
| Non-Divergent Interleaved (Lab 25) | 0.446 ms | 150.31 GB/s | Contiguous threads, but bank conflicts |
| Sequential Addressing (Lab 26) | 0.434 ms | 154.50 GB/s | No divergence, no bank conflicts |
| First Add During Load (Lab 27) | 0.236 ms | 284.94 GB/s | 2x elements per thread |
| **Warp-Level Optimization (Lab 28)** | **0.175 ms** | **383.39 GB/s** | Unrolled last warp, no sync overhead |

### Speedup Analysis (Baseline: Divergent Interleaved)

Using the **Divergent Interleaved** version as our baseline (0.706 ms, 95.12 GB/s):

| Version | Time Speedup | Bandwidth Speedup |
|---------|--------------|-------------------|
| Divergent Interleaved (Baseline) | 1.00x | 1.00x |
| Non-Divergent Interleaved (Lab 25) | **1.58x** | **1.58x** |
| Sequential Addressing (Lab 26) | **1.63x** | **1.62x** |
| First Add During Load (Lab 27) | **2.99x** | **3.00x** |
| Warp-Level Optimization (Lab 28) | **4.03x** | **4.03x** |

### Detailed Speedup Calculations

**Warp-Level Optimization vs Baseline (all previous optimizations + warp unrolling):**
```
Time Speedup    = 0.706 ms / 0.175 ms = 4.03x faster
Bandwidth Gain  = 383.39 GB/s / 95.12 GB/s = 4.03x higher throughput
```

### Incremental Improvement from Lab 27 to Lab 28

The additional gain from warp-level optimization (Lab 28 vs Lab 27):
```
Additional Time Speedup    = 0.236 ms / 0.175 ms = 1.35x faster
Additional Bandwidth Gain  = 383.39 GB/s / 284.94 GB/s = 1.35x higher throughput
Additional Time Saved      = 0.236 ms - 0.175 ms = 0.061 ms (25.8% reduction)
```

### Why Such a Big Improvement?

1. **Eliminated 6 `__syncthreads()` calls**: Each sync has overhead even within a warp
2. **Removed 6 conditional branches**: `if (tid < s)` checks are no longer needed
3. **No loop control overhead**: No counter updates, no exit condition checks
4. **Better instruction-level parallelism**: Compiler can schedule unrolled instructions more efficiently

### Current Efficiency

| Metric | Value |
|--------|-------|
| Achieved bandwidth | 383.39 GB/s |
| Theoretical peak (typical GPU) | ~432 GB/s |
| Efficiency | **88.7% of peak** |

We've improved from 65.9% to 88.7% of peak bandwidth - approaching the theoretical limit!

## What Makes Warp-Level Optimization Work

### Warp Synchronization

| Property | Effect |
|----------|--------|
| SIMT execution | All 32 threads execute the same instruction together |
| Implicit sync | No explicit `__syncthreads()` needed within a warp |
| Memory visibility | With `volatile`, shared memory updates are immediately visible |

### Why Only the Last Warp?

| Stride | Active Threads | Can Use Warp Optimization? |
|--------|---------------|---------------------------|
| 512 | 512 | No - Multiple warps active |
| 256 | 256 | No - Multiple warps active |
| 128 | 128 | No - Multiple warps active |
| 64 | 64 | No - 2 warps active |
| 32 | 32 | **Yes - Exactly 1 warp** |
| 16 | 16 | **Yes - Within single warp** |
| 8 | 8 | **Yes - Within single warp** |
| 4 | 4 | **Yes - Within single warp** |
| 2 | 2 | **Yes - Within single warp** |
| 1 | 1 | **Yes - Within single warp** |

Once we're down to 32 or fewer active threads, we're guaranteed to be within a single warp.

## Summary

1. **The problem**: Instruction overhead from synchronization, branches, and loop control in the reduction loop

2. **The solution**: Unroll the last 6 iterations (strides 32 to 1) into explicit additions, eliminating all overhead

3. **The result**: 1.35x speedup over Lab 27, achieving 383.39 GB/s (88.7% of peak bandwidth)

4. **Key insight**: Threads within a warp are implicitly synchronized, so `__syncthreads()` and conditional branches become unnecessary

5. **Cumulative improvement**: 4.03x faster than the original divergent interleaved approach from Lab 25

6. **Critical detail**: The `volatile` keyword ensures memory visibility between warp threads

7. **Near-optimal performance**: At 88.7% of theoretical peak bandwidth, we're approaching the hardware limits
