#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

// Tiled Matrix Multiplication Kernel using Shared Memory
__global__ void tiledMatrixMul(float *A, float *B, float *C,
                                int M, int K, int N) {
    // Shared memory for tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Calculate row and column index for this thread
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Number of tiles needed to cover the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A into shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded the tile
        __syncthreads();

        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Initialize matrix with random values
void initMatrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (float)(rand() % 10);
    }
}

// Print matrix
void printMatrix(float *mat, int rows, int cols, const char *name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Matrix dimensions: A(M x K) * B(K x N) = C(M x N)
    int M = 4;   // rows of A and C
    int K = 3;   // cols of A, rows of B
    int N = 5;   // cols of B and C

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);

    // Initialize matrices
    srand(42);
    initMatrix(h_A, M * K);
    initMatrix(h_B, K * N);

    // Print input matrices
    printMatrix(h_A, M, K, "Matrix A");
    printMatrix(h_B, K, N, "Matrix B");

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    printf("Grid: (%d, %d), Block: (%d, %d)\n\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Launch tiled matrix multiplication kernel
    tiledMatrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    // Print result
    printMatrix(h_C, M, N, "Result Matrix C (A x B)");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
