#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <iostream>
#include <omp.h>

// 4D Tensor Layout GEMM Kernel
// Matrices stored as 4D tensors for improved memory locality
// A: [M, K] -> [M/BM, K/BK, BM, BK]
// B: [K, N] -> [K/BK, N/BN, BK, BN]
// C: [M, N] -> [M/BM, N/BN, BM, BN]

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":"       \
                      << __LINE__ << " -> "                        \
                      << cudaGetErrorString(err)                   \
                      << std::endl;                                \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                         \
    do {                                                           \
        cublasStatus_t s = (call);                                 \
        if (s != CUBLAS_STATUS_SUCCESS) {                          \
            std::cerr << "cuBLAS error at " << __FILE__ << ":"     \
                      << __LINE__ << " -> " << s << std::endl;     \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    } while (0)

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#ifndef _BM
#define _BM 64
#endif

#ifndef _BN
#define _BN 64
#endif

#ifndef _BK
#define _BK 16
#endif

#ifndef _TM
#define _TM 8
#endif

#ifndef _TN
#define _TN 8
#endif

/**
 * Convert column-major 2D matrix to 4D tensor layout
 * Input: column-major matrix of size (rows x cols)
 * Output: 4D tensor [rows/block_rows, cols/block_cols, block_rows, block_cols]
 * 
 * For A (M x K): [M/BM, K/BK, BM, BK]
 * For B (K x N): [K/BK, N/BN, BK, BN]
 * For C (M x N): [M/BM, N/BN, BM, BN]
 */
void convert_to_4d_tensor(const double* src_colmajor, double* dst_4d,
                          int rows, int cols, int block_rows, int block_cols)
{
    int num_block_rows = CEIL_DIV(rows, block_rows);
    int num_block_cols = CEIL_DIV(cols, block_cols);
    
    #pragma omp parallel for collapse(2)
    for (int br = 0; br < num_block_rows; br++) {
        for (int bc = 0; bc < num_block_cols; bc++) {
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    int global_row = br * block_rows + i;
                    int global_col = bc * block_cols + j;
                    
                    double val = 0.0;
                    if (global_row < rows && global_col < cols) {
                        // Read from column-major: col * rows + row
                        val = src_colmajor[global_col * (size_t)rows + global_row];
                    }
                    
                    // Write to 4D layout: [br, bc, i, j]
                    // Linear index: ((br * num_block_cols + bc) * block_rows + i) * block_cols + j
                    size_t idx_4d = ((br * (size_t)num_block_cols + bc) * block_rows + i) * block_cols + j;
                    dst_4d[idx_4d] = val;
                }
            }
        }
    }
}

/**
 * Convert 4D tensor layout back to column-major 2D matrix
 */
void convert_from_4d_tensor(const double* src_4d, double* dst_colmajor,
                            int rows, int cols, int block_rows, int block_cols)
{
    int num_block_rows = CEIL_DIV(rows, block_rows);
    int num_block_cols = CEIL_DIV(cols, block_cols);
    
    #pragma omp parallel for collapse(2)
    for (int br = 0; br < num_block_rows; br++) {
        for (int bc = 0; bc < num_block_cols; bc++) {
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    int global_row = br * block_rows + i;
                    int global_col = bc * block_cols + j;
                    
                    if (global_row < rows && global_col < cols) {
                        // Read from 4D layout
                        size_t idx_4d = ((br * (size_t)num_block_cols + bc) * block_rows + i) * block_cols + j;
                        double val = src_4d[idx_4d];
                        
                        // Write to column-major: col * rows + row
                        dst_colmajor[global_col * (size_t)rows + global_row] = val;
                    }
                }
            }
        }
    }
}

/**
 * 4D Tensor GEMM Kernel
 * 
 * A is stored as [M/BM, K/BK, BM, BK]
 * B is stored as [K/BK, N/BN, BK, BN]
 * C is stored as [M/BM, N/BN, BM, BN]
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void dgemm_4d_tensor(int M, int N, int K,
                                double alpha, const double* A_4d,
                                const double* B_4d,
                                double beta, double* C_4d)
{
    __shared__ double As[BM * BK];
    __shared__ double Bs[BK * BN];

    const int blockRow = blockIdx.y; // which BM-block of rows
    const int blockCol = blockIdx.x; // which BN-block of cols

    const int tid = threadIdx.x;
    const int nThreads = blockDim.x;

    // Thread tile coordinates inside block
    const int tilesPerRow = BN / TN;
    const int threadRow = tid / tilesPerRow;
    const int threadCol = tid % tilesPerRow;

    // Local register tiles
    double threadResults[TM * TN];
    #pragma unroll
    for (int i = 0; i < TM * TN; ++i) threadResults[i] = 0.0;

    double regM[TM];
    double regN[TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) regM[i] = 0.0;
    #pragma unroll
    for (int i = 0; i < TN; ++i) regN[i] = 0.0;

    // Dimensions for 4D indexing
    // const int num_block_M = CEIL_DIV(M, BM);
    const int num_block_K = CEIL_DIV(K, BK);
    const int num_block_N = CEIL_DIV(N, BN);

    // Loop over K dimension in blocks of BK
    for (int bkIdx = 0; bkIdx < num_block_K; bkIdx++) {
        // Load A tile [blockRow, bkIdx, :, :] which is shape [BM, BK]
        // A_4d layout: [M/BM, K/BK, BM, BK]
        // Base index for this tile: (blockRow * num_block_K + bkIdx) * BM * BK
        size_t A_tile_base = (blockRow * (size_t)num_block_K + bkIdx) * BM * BK;
        
        const int A_tile_elems = BM * BK;
        #pragma unroll
        for (int idx = tid; idx < A_tile_elems; idx += nThreads) {
            int local_r = idx / BK; // 0..BM-1
            int local_c = idx % BK; // 0..BK-1
            
            // Bounds check
            int global_row = blockRow * BM + local_r;
            int global_col = bkIdx * BK + local_c;
            
            double val = 0.0;
            if (global_row < M && global_col < K) {
                // Index within the 4D tile: [local_r, local_c] in row-major
                size_t idx_4d = A_tile_base + local_r * BK + local_c;
                val = A_4d[idx_4d];
            } else {
                val = 0.0;
            }
            As[local_r * BK + local_c] = val;
        }

        // Load B tile [bkIdx, blockCol, :, :] which is shape [BK, BN]
        // B_4d layout: [K/BK, N/BN, BK, BN]
        // Base index for this tile: (bkIdx * num_block_N + blockCol) * BK * BN
        size_t B_tile_base = (bkIdx * (size_t)num_block_N + blockCol) * BK * BN;
        
        const int B_tile_elems = BK * BN;
        #pragma unroll
        for (int idx = tid; idx < B_tile_elems; idx += nThreads) {
            int local_r = idx / BN; // 0..BK-1
            int local_c = idx % BN; // 0..BN-1
            
            // Bounds check
            int global_row = bkIdx * BK + local_r;
            int global_col = blockCol * BN + local_c;
            
            double val = 0.0;
            if (global_row < K && global_col < N) {
                // Index within the 4D tile: [local_r, local_c] in row-major
                size_t idx_4d = B_tile_base + local_r * BN + local_c;
                val = B_4d[idx_4d];
            } else {
                val = 0.0;
            }
            Bs[local_r * BN + local_c] = val;
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            // Load TM elements from As
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int local_row = threadRow * TM + i;
                regM[i] = As[local_row * BK + dot];
            }

            // Load TN elements from Bs
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int local_col = threadCol * TN + j;
                regN[j] = Bs[dot * BN + local_col];
            }

            // Accumulate outer product
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results back to C_4d
    // C_4d layout: [M/BM, N/BN, BM, BN]
    // Base index for this tile: (blockRow * num_block_N + blockCol) * BM * BN
    size_t C_tile_base = (blockRow * (size_t)num_block_N + blockCol) * BM * BN;
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int local_row = threadRow * TM + i;
        int global_row = blockRow * BM + local_row;
        if (global_row >= M) continue;
        
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int local_col = threadCol * TN + j;
            int global_col = blockCol * BN + local_col;
            if (global_col >= N) continue;

            double acc = threadResults[i * TN + j];
            
            // Index in 4D tile: [local_row, local_col] in row-major
            size_t idx_4d = C_tile_base + local_row * BN + local_col;
            
            double old = C_4d[idx_4d];
            C_4d[idx_4d] = alpha * acc + beta * old;
        }
    }
}

/**
 * Host function to launch 4D tensor GEMM kernel
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
void launch_dgemm_4d(int M, int N, int K,
                     double alpha, const double* A_4d,
                     const double* B_4d,
                     double beta, double* C_4d)
{
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM / TM) * (BN / TN));

    dgemm_4d_tensor<BM,BN,BK,TM,TN><<<gridDim, blockDim>>>(
        M, N, K, alpha, A_4d, B_4d, beta, C_4d
    );
}

/**
 * Verify correctness by comparing with CPU reference
 */
bool verify_result(const double* gpu_result, const double* ref_result,
                   int M, int N, double tolerance = 1e-6) {
    for (int i = 0; i < M * N; i++) {
        double diff = fabs(gpu_result[i] - ref_result[i]);
        double rel_error = diff / (fabs(ref_result[i]) + 1e-10);
        if (rel_error > tolerance) {
            printf("Mismatch at index %d: GPU=%f, REF=%f, rel_error=%e\n",
                   i, gpu_result[i], ref_result[i], rel_error);
            return false;
        }
    }
    return true;
}

/**
 * Benchmark 4D tensor kernel
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
double benchmark_kernel_4d(int M, int N, int K,
                           double alpha, const double* d_A_4d,
                           const double* d_B_4d,
                           double beta, double* d_C_4d,
                           int num_runs = 10)
{
    // Warmup
    launch_dgemm_4d<BM,BN,BK,TM,TN>(M, N, K, alpha, d_A_4d, d_B_4d, beta, d_C_4d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double total_ms = 0.0;
    for (int i = 0; i < num_runs; i++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        launch_dgemm_4d<BM,BN,BK,TM,TN>(M, N, K, alpha, d_A_4d, d_B_4d, beta, d_C_4d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_ms += milliseconds;

        std::cout << "4D Tensor GEMM (BM=" << BM << ", BN=" << BN
                  << ", BK=" << BK << ", TM=" << TM << ", TN=" << TN
                  << "): " << milliseconds << " ms (" << i << "/" << num_runs << ")"
                  << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    return total_ms / num_runs;
}

/**
 * cuBLAS benchmark using column-major layout
 */
double benchmark_cublas(int M, int N, int K,
                        double alpha, const double* d_A,
                        const double* d_B,
                        double beta, double* d_C,
                        int num_runs = 10)
{
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Warmup
    CUBLAS_CHECK(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_A, M,
        d_B, K,
        &beta,
        d_C, M
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    double total_ms = 0.0;
    for (int i = 0; i < num_runs; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK(cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M
        ));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_ms += milliseconds;

        std::cout << "cuBLAS DGEMM (column-major): "
                  << milliseconds << " ms (" << i << "/" << num_runs << ")"
                  << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return total_ms / num_runs;
}

extern "C" {
    // Fast initialization using LCG
    void fast_init_matrix(double* matrix, int size, unsigned int base_seed) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            unsigned int seed = base_seed + i;
            seed = seed * 1664525u + 1013904223u;
            matrix[i] = (double)(seed & 0xFFFF) / 65536.0;
        }
    }

    double run_gemm_4d_with_params(int M, int N, int K,
                                   int BM, int BN, int BK, int TM, int TN) {
        // Validate parameters
        int threads_per_block = (BM * BN) / (TM * TN);
        if (threads_per_block > 1024 || threads_per_block <= 0) {
            printf("Error: Invalid thread count %d (must be 1-1024)\n", threads_per_block);
            return -1.0;
        }

        int smem_size = (BM * BK + BK * BN) * (int)sizeof(double);
        if (smem_size > 48 * 1024) {
            printf("Error: Shared memory %d bytes exceeds 48KB limit\n", smem_size);
            return -1.0;
        }

        if (BM % TM != 0 || BN % TN != 0) {
            printf("Error: BM must be divisible by TM, BN by TN\n");
            return -1.0;
        }

        // Allocate host memory for column-major matrices
        size_t size_A = (size_t)M * K * sizeof(double);
        size_t size_B = (size_t)K * N * sizeof(double);
        size_t size_C = (size_t)M * N * sizeof(double);

        double *h_A = (double*)malloc(size_A);
        double *h_B = (double*)malloc(size_B);
        double *h_C = (double*)malloc(size_C);
        double *h_C_ref = (double*)malloc(size_C);

        if (!h_A || !h_B || !h_C || !h_C_ref) {
            printf("Host allocation failed\n");
            return -1.0;
        }

        // Initialize column-major matrices
        fast_init_matrix(h_A, M * K, 42);
        fast_init_matrix(h_B, K * N, 4242);

        #pragma omp parallel for
        for (int i = 0; i < M * N; ++i) h_C[i] = 0.0;
        #pragma omp parallel for
        for (int i = 0; i < M * N; ++i) h_C_ref[i] = 0.0;

        // Convert to 4D tensor layout
        printf("Converting matrices to 4D tensor layout...\n");
        
        int num_blocks_M = CEIL_DIV(M, BM);
        int num_blocks_K = CEIL_DIV(K, BK);
        int num_blocks_N = CEIL_DIV(N, BN);
        
        size_t size_A_4d = (size_t)num_blocks_M * num_blocks_K * BM * BK * sizeof(double);
        size_t size_B_4d = (size_t)num_blocks_K * num_blocks_N * BK * BN * sizeof(double);
        size_t size_C_4d = (size_t)num_blocks_M * num_blocks_N * BM * BN * sizeof(double);
        
        double *h_A_4d = (double*)malloc(size_A_4d);
        double *h_B_4d = (double*)malloc(size_B_4d);
        double *h_C_4d = (double*)malloc(size_C_4d);
        
        if (!h_A_4d || !h_B_4d || !h_C_4d) {
            printf("4D tensor host allocation failed\n");
            return -1.0;
        }

        convert_to_4d_tensor(h_A, h_A_4d, M, K, BM, BK);
        convert_to_4d_tensor(h_B, h_B_4d, K, N, BK, BN);
        convert_to_4d_tensor(h_C, h_C_4d, M, N, BM, BN);

        // Device allocations
        double *d_A = nullptr, *d_B = nullptr, *d_C_ref = nullptr;
        double *d_A_4d = nullptr, *d_B_4d = nullptr, *d_C_4d = nullptr;
        
        // Column-major for cuBLAS reference
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C_ref, size_C));
        
        // 4D tensors for our kernel
        CUDA_CHECK(cudaMalloc(&d_A_4d, size_A_4d));
        CUDA_CHECK(cudaMalloc(&d_B_4d, size_B_4d));
        CUDA_CHECK(cudaMalloc(&d_C_4d, size_C_4d));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_ref, h_C_ref, size_C, cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(d_A_4d, h_A_4d, size_A_4d, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_4d, h_B_4d, size_B_4d, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_4d, h_C_4d, size_C_4d, cudaMemcpyHostToDevice));

        double alpha = 1.0;
        double beta = 0.0;
        double kernel_time_ms = -1.0;

        // Benchmark 4D tensor kernel
        printf("\nBenchmarking 4D Tensor Kernel...\n");
        if (BM == _BM && BN == _BN && BK == _BK && TM == _TM && TN == _TN) {
            kernel_time_ms = benchmark_kernel_4d<_BM,_BN,_BK,_TM,_TN>(
                M, N, K, alpha, d_A_4d, d_B_4d, beta, d_C_4d, 10
            );
        } else {
            printf("Error: runtime BM/BN/BK/TM/TN must match compile-time _BM/_BN/_BK/_TM/_TN\n");
            kernel_time_ms = -1.0;
        }

        // Copy 4D result back and convert to column-major
        CUDA_CHECK(cudaMemcpy(h_C_4d, d_C_4d, size_C_4d, cudaMemcpyDeviceToHost));
        convert_from_4d_tensor(h_C_4d, h_C, M, N, BM, BN);

        // Reference using cuBLAS
        printf("\nBenchmarking cuBLAS Reference...\n");
        double cublas_time_ms = benchmark_cublas(M, N, K, alpha, d_A, d_B, beta, d_C_ref, 10);
        CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost));

        // Verify correctness
        printf("\nVerifying correctness...\n");
        bool ok = verify_result(h_C, h_C_ref, M, N);
        if (!ok) {
            printf("Verification FAILED for 4D tensor kernel\n");
        } else {
            printf("Verification PASSED\n");
        }

        // Performance comparison
        if (kernel_time_ms > 0) {
            double gflops_4d = (2.0 * (double)M * (double)N * (double)K) / (kernel_time_ms * 1e6);
            double gflops_cublas = (2.0 * (double)M * (double)N * (double)K) / (cublas_time_ms * 1e6);
            
            printf("\n=== Performance Summary ===\n");
            printf("4D Tensor Kernel: %.3f ms (%.2f GFLOPS)\n", kernel_time_ms, gflops_4d);
            printf("cuBLAS Reference: %.3f ms (%.2f GFLOPS)\n", cublas_time_ms, gflops_cublas);
            printf("Speedup vs cuBLAS: %.2fx\n", cublas_time_ms / kernel_time_ms);
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C_ref));
        CUDA_CHECK(cudaFree(d_A_4d));
        CUDA_CHECK(cudaFree(d_B_4d));
        CUDA_CHECK(cudaFree(d_C_4d));
        
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_ref);
        free(h_A_4d);
        free(h_B_4d);
        free(h_C_4d);

        return kernel_time_ms;
    }
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    const int BM = _BM;
    const int BN = _BN;
    const int BK = _BK;
    const int TM = _TM;
    const int TN = _TN;

    printf("Testing 4D Tensor GEMM with parameters:\n");
    printf("BM=%d BN=%d BK=%d TM=%d TN=%d\n", BM, BN, BK, TM, TN);
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("4D Tensor shapes:\n");
    printf("  A: [%d, %d, %d, %d]\n", CEIL_DIV(M, BM), CEIL_DIV(K, BK), BM, BK);
    printf("  B: [%d, %d, %d, %d]\n", CEIL_DIV(K, BK), CEIL_DIV(N, BN), BK, BN);
    printf("  C: [%d, %d, %d, %d]\n\n", CEIL_DIV(M, BM), CEIL_DIV(N, BN), BM, BN);

    double time_ms = run_gemm_4d_with_params(M, N, K, BM, BN, BK, TM, TN);

    if (time_ms <= 0) {
        printf("Kernel execution failed\n");
        return 1;
    }

    return 0;
}
