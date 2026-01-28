#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <iostream>
#include <omp.h>

// All matrices are in column-major format now.
// Matrix dimensions (M x K) * (K x N) = (M x N)
// C = alpha * A * B + beta * C
// A: M x K (column-major), B: K x N (column-major), C: M x N (column-major)

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
 * 2D Block-Tiled Matrix Multiplication Kernel for FP64 (column-major)
 *
 * Template Parameters:
 * - BM: Block size M dimension (rows of C per block)
 * - BN: Block size N dimension (cols of C per block)
 * - BK: Block size K dimension (dot product chunk size)
 * - TM: Thread tile size M dimension (rows per thread)
 * - TN: Thread tile size N dimension (cols per thread)
 *
 * All matrices are column-major:
 *   element (r, c) -> base[c * M + r]
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void dgemm_2d_blocktiling(int M, int N, int K,
                                     double alpha, const double* A,
                                     const double* B,
                                     double beta, double* C)
{
    extern __shared__ double shared_mem[]; // optional if you prefer dynamic SHM
    // We'll use statically-sized arrays declared from templates (as before):
    __shared__ double As[BM * BK];  // BM x BK tile of A (row-major inside shared)
    __shared__ double Bs[BK * BN];  // BK x BN tile of B (row-major inside shared)

    const int blockRow = blockIdx.y; // which BM-block of rows
    const int blockCol = blockIdx.x; // which BN-block of cols

    const int row0 = blockRow * BM;  // global row start
    const int col0 = blockCol * BN;  // global col start

    const int tid = threadIdx.x;
    const int nThreads = blockDim.x;

    // Thread tile coordinates inside block
    const int tilesPerRow = BN / TN;    // number of thread columns
    const int threadRow = tid / tilesPerRow;
    const int threadCol = tid % tilesPerRow;

    // Local register tiles (TM x TN)
    double threadResults[TM * TN];
    #pragma unroll
    for (int i = 0; i < TM * TN; ++i) threadResults[i] = 0.0;

    double regM[TM];
    double regN[TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) regM[i] = 0.0;
    #pragma unroll
    for (int i = 0; i < TN; ++i) regN[i] = 0.0;

    // Loop over K in blocks of BK
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // -- Load A tile (BM x BK) into shared memory
        // Each thread loads multiple elements in a strided fashion
        const int A_tile_elems = BM * BK;
        #pragma unroll
        for (int idx = tid; idx < A_tile_elems; idx += nThreads) {
            int local_r = idx / BK; // 0..BM-1
            int local_c = idx % BK; // 0..BK-1
            // global A element at (row = row0 + local_r, col = bkIdx + local_c)
            // column-major: A[ (bkIdx + local_c) * M + (row0 + local_r) ]
            int global_col = bkIdx + local_c;
            int global_row = row0 + local_r;
            // bounds check (for edge tiles)
            double val = 0.0;
            if (global_row < M && global_col < K) {
                val = A[ global_col * (size_t)M + global_row ];
            } else {
                val = 0.0;
            }
            As[ local_r * BK + local_c ] = val; // tile kept row-major inside SHM
        }

        // -- Load B tile (BK x BN) into shared memory
        const int B_tile_elems = BK * BN;
        #pragma unroll
        for (int idx = tid; idx < B_tile_elems; idx += nThreads) {
            int local_r = idx / BN; // 0..BK-1 (k dimension)
            int local_c = idx % BN; // 0..BN-1 (n dimension)
            // global B element at (row = bkIdx + local_r, col = col0 + local_c)
            // column-major: B[ (col0 + local_c) * K + (bkIdx + local_r) ]
            int global_col = col0 + local_c;
            int global_row = bkIdx + local_r;
            double val = 0.0;
            if (global_row < K && global_col < N) {
                val = B[ global_col * (size_t)K + global_row ];
            } else {
                val = 0.0;
            }
            Bs[ local_r * BN + local_c ] = val;
        }

        __syncthreads();

        // Compute partial products using the loaded tile
        #pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            // load TM elements from As column 'dot' for this thread's rows
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int local_row = threadRow * TM + i; // 0..BM-1
                regM[i] = As[ local_row * BK + dot ];
            }

            // load TN elements from Bs row 'dot' for this thread's columns
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int local_col = threadCol * TN + j; // 0..BN-1
                regN[j] = Bs[ dot * BN + local_col ];
            }

            // accumulate TM x TN outer product
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }

        __syncthreads();
    } // end bk loop

    // Write results back to global memory (column-major)
    // Each thread writes TM x TN values
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int local_row = threadRow * TM + i; // within block row
        int global_row = row0 + local_row;
        if (global_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int local_col = threadCol * TN + j; // within block col
            int global_col = col0 + local_col;
            if (global_col >= N) continue;
            // C[ global_col * M + global_row ] = alpha * acc + beta * old
            double acc = threadResults[i * TN + j];
            double old = C[ global_col * (size_t)M + global_row ];
            C[ global_col * (size_t)M + global_row ] = alpha * acc + beta * old;
        }
    }
}

/**
 * Host function to launch the GEMM kernel with specified tiling parameters
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
void launch_dgemm(int M, int N, int K,
                  double alpha, const double* A,
                  const double* B,
                  double beta, double* C)
{
    dim3 gridDim( CEIL_DIV(N, BN), CEIL_DIV(M, BM) );
    dim3 blockDim( (BM / TM) * (BN / TN) );

    // optional: compute dynamic shared memory size (not used here)
    size_t shm = 0;
    // launch
    dgemm_2d_blocktiling<BM,BN,BK,TM,TN><<<gridDim, blockDim, shm>>>(
        M, N, K, alpha, A, B, beta, C
    );
}

/**
 * Verify correctness by comparing with CPU reference
 * The inputs are raw linear arrays that are in COLUMN-MAJOR layout.
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
 * Benchmark templated kernel; returns average milliseconds (excluding warmup).
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
double benchmark_kernel(int M, int N, int K,
                        double alpha, const double* d_A,
                        const double* d_B,
                        double beta, double* d_C,
                        int num_runs = 10)
{
    // Warmup
    launch_dgemm<BM,BN,BK,TM,TN>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double total_ms = 0.0;
    for (int i = 0; i < num_runs; i++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        launch_dgemm<BM,BN,BK,TM,TN>(M, N, K, alpha, d_A, d_B, beta, d_C);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_ms += milliseconds;

        std::cout << "2D Block-Tiled GEMM (BM=" << BM << ", BN=" << BN
                  << ", BK=" << BK << ", TM=" << TM << ", TN=" << TN
                  << "): " << milliseconds << " ms (" << i << "/" << num_runs << ")"
                  << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    return total_ms / num_runs;
}

/**
 * cuBLAS benchmark using column-major layout (no transposes).
 * Returns average ms over runs.
 */
double benchmark_cublas(int M, int N, int K,
                        double alpha, const double* d_A,
                        const double* d_B,
                        double beta, double* d_C,
                        int num_runs = 10)
{
    cublasHandle_t handle;
    CUBLAS_CHECK( cublasCreate(&handle) );

    // Warmup
    CUBLAS_CHECK( cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_A, M,
        d_B, K,
        &beta,
        d_C, M
    ) );
    CUDA_CHECK(cudaDeviceSynchronize());

    double total_ms = 0.0;
    for (int i = 0; i < num_runs; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK( cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M
        ) );
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

    CUBLAS_CHECK( cublasDestroy(handle) );
    return total_ms / num_runs;
}

/**
 * External C interface for Python to call with specific parameters
 * Returns: time in milliseconds (kernel avg), or -1 on error
 */
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

    double run_gemm_with_params(int M, int N, int K,
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

        // Allocate host memory (column-major layout)
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

        // Initialize (column-major layout) using the same generator
        // fast_init_matrix fills linear storage; interpreting as column-major is fine.
        fast_init_matrix(h_A, M * K, 42);
        fast_init_matrix(h_B, K * N, 4242);

        // zero C
        #pragma omp parallel for
        for (int i = 0; i < M * N; ++i) h_C[i] = 0.0;
        #pragma omp parallel for
        for (int i = 0; i < M * N; ++i) h_C_ref[i] = 0.0;

        // Device allocations
        double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C_ref = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C, size_C));
        CUDA_CHECK(cudaMalloc(&d_C_ref, size_C));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_ref, h_C_ref, size_C, cudaMemcpyHostToDevice));

        double alpha = 1.0;
        double beta = 0.0;
        double kernel_time_ms = -1.0;

        // Dispatch templated kernel (choose the template instance that matches compile-time constants)
        // For simplicity we dispatch using the compile-time defaults; if runtime BM/BN != compile-time
        // defaults you should dispatch a fallback or generate code for those sizes.
        if (BM == _BM && BN == _BN && BK == _BK && TM == _TM && TN == _TN) {
            kernel_time_ms = benchmark_kernel<_BM,_BN,_BK,_TM,_TN>(
                M, N, K, alpha, d_A, d_B, beta, d_C, 10
            );
        } else {
            printf("Error: runtime BM/BN/BK/TM/TN must match compile-time _BM/_BN/_BK/_TM/_TN\n");
            kernel_time_ms = -1.0;
        }

        // copy kernel result back
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

        // Reference using cuBLAS (column-major)
        double cublas_time_ms = benchmark_cublas(M, N, K, alpha, d_A, d_B, beta, d_C_ref, 10);
        CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost));

        // verify
        bool ok = verify_result(h_C, h_C_ref, M, N);
        if (!ok) {
            printf("Verification FAILED for BM=%d BN=%d BK=%d TM=%d TN=%d\n", BM, BN, BK, TM, TN);
        } else {
            printf("Verification PASSED\n");
        }

        // cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_C_ref));
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_ref);

        return kernel_time_ms;
    }
} // extern "C"

/**
 * Example usage - Test a specific configuration
 */
int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    // Test configuration (must match compile-time _BM/_BN/_BK/_TM/_TN)
    const int BM = _BM;
    const int BN = _BN;
    const int BK = _BK;
    const int TM = _TM;
    const int TN = _TN;

    printf("Testing GEMM with parameters: BM=%d BN=%d BK=%d TM=%d TN=%d\n",
           BM, BN, BK, TM, TN);
    printf("Matrix size: %d x %d x %d\n", M, N, K);

    double time_ms = run_gemm_with_params(M, N, K, BM, BN, BK, TM, TN);

    if (time_ms > 0) {
        double gflops = (2.0 * (double)M * (double)N * (double)K) / (time_ms * 1e6);
        printf("Kernel Time: %.3f ms\n", time_ms);
        printf("Performance: %.2f GFLOPS\n", gflops);
    } else {
        printf("Kernel execution failed or returned error\n");
    }

    return 0;
}
