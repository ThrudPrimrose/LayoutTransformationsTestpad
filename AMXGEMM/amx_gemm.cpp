//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>

#include <mkl.h>

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

// Define tile config data structure
typedef struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __tilecfg;

// Timer utility
double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Initialize tile config
static void init_tile_config(__tilecfg* tileinfo) {
  int i;
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;

  for (i = 0; i < 8; i++) {
    tileinfo->colsb[i] = 64;  // 64 bytes per row
    tileinfo->rows[i] = 16;   // 16 rows
  }

  _tile_loadconfig(tileinfo);
}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use() {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
    return false;
  } else {
    printf("\n TILE DATA USE SET - OK \n\n");
    return true;
  }
  return true;
}

// Sequential scalar GEMM reference implementation (no vectorization)
void sequential_gemm(const __bf16* A, const __bf16* B, float* C, int M, int N,
                     int K) {
  // Zero initialize output
  for (int i = 0; i < M * N; i++) {
    C[i] = 0.0f;
  }
  
  // Basic triple-nested loop: C = A * B
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum +=
            static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]);
      }
      C[i * N + j] = sum;
    }
  }
}

// Alias for reference computation
void naive_gemm(const __bf16* A, const __bf16* B, float* C, int M, int N,
                int K) {
  sequential_gemm(A, B, C, M, N, K);
}

void AMX_gemm(const __bf16* A, const __bf16* B, float* C, int M, int N, int K) {
  // Load tile configuration
  __tilecfg tile_data = {0};
  init_tile_config(&tile_data);

  // Loop over C tiles
  constexpr int ACC_M = 2, ACC_N = 2;
  constexpr int TILE_M = 16;
  constexpr int TILE_N = 16;
  // K-block size matching tile depth
  constexpr int K_TILE = 32;

  // Iterate over output matrix tiles
  for (int m = 0; m < M; m += TILE_M * ACC_M) {
    for (int n = 0; n < N; n += TILE_N * ACC_N) {
      
      // Zero accumulator tiles at the start
      _tile_zero(0);  // C[m:m+16, n:n+16]
      _tile_zero(1);  // C[m:m+16, n+16:n+32]
      _tile_zero(2);  // C[m+16:m+32, n:n+16]
      _tile_zero(3);  // C[m+16:m+32, n+16:n+32]
      
      // Iterate over K dimension in blocks
      for (int k = 0; k < K; k += K_TILE) {
        // Pack a [K_TILE x (TILE_N*ACC_N)] block of B into buffer
        // VNNI layout: interleaved pairs of rows for BF16
        __bf16 b_pack[K_TILE * (TILE_N * ACC_N)];
        
        // Pack B matrix for current K-block
        for (int bk = 0; bk < K_TILE && (k + bk) < K; bk += 2) {
          for (int bn = 0; bn < TILE_N * ACC_N && (n + bn) < N; bn++) {
            // VNNI format: pairs of consecutive K elements
            int dest_idx = (bk / 2) * (TILE_N * ACC_N * 2) + bn * 2;
            b_pack[dest_idx] = B[(k + bk) * N + (n + bn)];
            // Handle edge case where K is odd
            if (k + bk + 1 < K) {
              b_pack[dest_idx + 1] = B[(k + bk + 1) * N + (n + bn)];
            } else {
              b_pack[dest_idx + 1] = 0.0f;  // Pad with zero
            }
          }
        }

        // Load A tiles for current M-block
        _tile_loadd(4, &A[m * K + k], K * sizeof(__bf16));
        if (m + TILE_M < M) {
          _tile_loadd(5, &A[(m + TILE_M) * K + k], K * sizeof(__bf16));
        }

        // Load B tiles from packed buffer
        // Stride is the number of bytes between row-pairs
        int stride = (TILE_N * ACC_N) * 2 * sizeof(__bf16);
        
        _tile_loadd(6, &b_pack[0], stride);  // Cols n:n+16
        _tile_loadd(7, &b_pack[TILE_N * 2], stride);  // Cols n+16:n+32

        // Perform matrix multiplication: C += A * B
        _tile_dpbf16ps(0, 4, 6);  // C[m:m+16, n:n+16] += A[m:m+16, k:k+32] * B[k:k+32, n:n+16]
        _tile_dpbf16ps(1, 4, 7);  // C[m:m+16, n+16:n+32] += A[m:m+16, k:k+32] * B[k:k+32, n+16:n+32]
        
        if (m + TILE_M < M) {
          _tile_dpbf16ps(2, 5, 6);  // C[m+16:m+32, n:n+16] += A[m+16:m+32, k:k+32] * B[k:k+32, n:n+16]
          _tile_dpbf16ps(3, 5, 7);  // C[m+16:m+32, n+16:n+32] += A[m+16:m+32, k:k+32] * B[k:k+32, n+16:n+32]
        }
      }
      
      // Store accumulated results
      _tile_stored(0, &C[m * N + n], N * sizeof(float));
      _tile_stored(1, &C[m * N + n + TILE_N], N * sizeof(float));
      
      if (m + TILE_M < M) {
        _tile_stored(2, &C[(m + TILE_M) * N + n], N * sizeof(float));
        _tile_stored(3, &C[(m + TILE_M) * N + n + TILE_N], N * sizeof(float));
      }
    }
  }
  
  // Release tile resources
  _tile_release();
}

void MKL_gemm(const __bf16* A, const __bf16* B, float* C, int M, int N, int K) {
  // Convert BF16 to FP32 for MKL (MKL doesn't have direct BF16 GEMM in older versions)
  float* A_fp32 = (float*)mkl_malloc(M * K * sizeof(float), 64);
  float* B_fp32 = (float*)mkl_malloc(K * N * sizeof(float), 64);
  
  // Convert BF16 to FP32
  for (int i = 0; i < M * K; i++) {
    A_fp32[i] = static_cast<float>(A[i]);
  }
  for (int i = 0; i < K * N; i++) {
    B_fp32[i] = static_cast<float>(B[i]);
  }
  
  // MKL SGEMM: C = alpha * A * B + beta * C
  float alpha = 1.0f;
  float beta = 0.0f;
  
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K, alpha, A_fp32, K, B_fp32, N, beta, C, N);
  
  mkl_free(A_fp32);
  mkl_free(B_fp32);
}

int main(int argc, char** argv) {
  // Define matrix sizes - using larger dimensions for realistic benchmarking
  // Default: M=1024, N=1024, K=1024 (can be overridden via command line)
  int M = 1024, N = 1024, K = 1024;
  
  // Parse command line arguments if provided
  if (argc >= 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }
  
  // Ensure dimensions are multiples of 32 for optimal AMX performance
  M = ((M + 31) / 32) * 32;
  N = ((N + 31) / 32) * 32;
  K = ((K + 31) / 32) * 32;
  
  __bf16 *A, *B;
  float *C_sequential, *C_amx, *C_mkl, *C_ref;
  
  // Allocate aligned memory
  A = (__bf16*)mkl_malloc(M * K * sizeof(__bf16), 64);
  B = (__bf16*)mkl_malloc(K * N * sizeof(__bf16), 64);
  C_sequential = (float*)mkl_malloc(M * N * sizeof(float), 64);
  C_amx = (float*)mkl_malloc(M * N * sizeof(float), 64);
  C_mkl = (float*)mkl_malloc(M * N * sizeof(float), 64);
  C_ref = (float*)mkl_malloc(M * N * sizeof(float), 64);

  if (!A || !B || !C_sequential || !C_amx || !C_mkl || !C_ref) {
    printf("ERROR: Memory allocation failed!\n");
    printf("Requested: A(%ldMB) + B(%ldMB) + C matrices(%ldMB x 4)\n",
           (M * K * sizeof(__bf16)) / (1024*1024),
           (K * N * sizeof(__bf16)) / (1024*1024),
           (M * N * sizeof(float)) / (1024*1024));
    return -1;
  }

  // Print memory usage
  size_t total_memory = M * K * sizeof(__bf16) + K * N * sizeof(__bf16) + 
                        4 * M * N * sizeof(float);
  printf("\n========================================\n");
  printf("MATRIX MULTIPLICATION BENCHMARK\n");
  printf("========================================\n");
  printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
  printf("Total memory allocated: %.2f MB\n", total_memory / (1024.0 * 1024.0));
  printf("  A matrix (BF16):  %6ld MB\n", (M * K * sizeof(__bf16)) / (1024*1024));
  printf("  B matrix (BF16):  %6ld MB\n", (K * N * sizeof(__bf16)) / (1024*1024));
  printf("  C matrices (FP32): %6ld MB x 4\n", (M * N * sizeof(float)) / (1024*1024));
  printf("========================================\n");

  // Initialize matrices
  srand(0);
  for (int i = 0; i < M * K; i++) {
    A[i] = static_cast<__bf16>(rand() / (RAND_MAX / 2.0f) - 1);
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = static_cast<__bf16>(rand() / (RAND_MAX / 2.0f) - 1);
  }

  // Request permission to linux kernel to run AMX (only once)
  if (!set_tiledata_use()) {
    printf("AMX not available, will only run MKL\n");
  }

  // For large matrices, skip naive GEMM (would be extremely slow)
  // Only verify a small subset of results
  bool run_naive = (M <= 128 && N <= 128 && K <= 128);
  
  if (run_naive) {
    printf("Running naive GEMM for reference (small matrix)...\n");
    memset(C_ref, 0, M * N * sizeof(float));
    naive_gemm(A, B, C_ref, M, N, K);
  } else {
    printf("Skipping naive GEMM (matrix too large, would be extremely slow)\n");
    printf("Will verify correctness by comparing AMX vs MKL results\n");
  }

  printf("\n========================================\n");
  printf("PERFORMANCE BENCHMARK\n");
  printf("Matrix Size: M=%d, N=%d, K=%d\n", M, N, K);
  printf("Total operations: %.2f GFLOPS\n", (2.0 * M * N * K) / 1e9);
  printf("========================================\n\n");

  // Warmup runs (important for large matrices to ensure caches are loaded)
  printf("Running warmup iterations...\n");
  
  // Warmup Sequential GEMM (only for small matrices)
  bool run_sequential = (M <= 256 && N <= 256 && K <= 256);
  if (run_sequential) {
    printf("  Sequential GEMM warmup...\n");
    for (int iter = 0; iter < 3; iter++) {
      memset(C_sequential, 0, M * N * sizeof(float));
      sequential_gemm(A, B, C_sequential, M, N, K);
    }
  }
  
  printf("  AMX GEMM warmup...\n");
  for (int iter = 0; iter < 3; iter++) {
    memset(C_amx, 0, M * N * sizeof(float));
    AMX_gemm(A, B, C_amx, M, N, K);
  }
  
  printf("  MKL GEMM warmup...\n");
  for (int iter = 0; iter < 3; iter++) {
    memset(C_mkl, 0, M * N * sizeof(float));
    MKL_gemm(A, B, C_mkl, M, N, K);
  }
  printf("Warmup complete.\n\n");

  // Benchmark Sequential GEMM (only for small matrices)
  int num_iterations = 10;
  double seq_avg_time = 0.0;
  double seq_gflops = 0.0;
  double seq_bandwidth_gb = 0.0;
  
  if (run_sequential) {
    printf("Benchmarking Sequential (Scalar) GEMM (%d iterations)...\n", num_iterations);
    double seq_total_time = 0.0;
    for (int iter = 0; iter < num_iterations; iter++) {
      memset(C_sequential, 0, M * N * sizeof(float));
      double start = get_time();
      sequential_gemm(A, B, C_sequential, M, N, K);
      double end = get_time();
      seq_total_time += (end - start);
    }
    seq_avg_time = seq_total_time / num_iterations;
    
    // Calculate GFLOPS for sequential
    double gflops = (2.0 * M * N * K) / 1e9;
    seq_gflops = gflops / seq_avg_time;
    
    // Calculate bandwidth for sequential
    double memory_bytes = (M * K * 2.0 + K * N * 2.0 + M * N * 4.0);
    seq_bandwidth_gb = (memory_bytes / 1e9) / seq_avg_time;
  } else {
    printf("Skipping Sequential GEMM benchmark (matrix too large, would be extremely slow)\n");
  }

  // Benchmark AMX GEMM
  printf("Benchmarking AMX GEMM (%d iterations)...\n", num_iterations);
  double amx_total_time = 0.0;
  for (int iter = 0; iter < num_iterations; iter++) {
    memset(C_amx, 0, M * N * sizeof(float));
    double start = get_time();
    AMX_gemm(A, B, C_amx, M, N, K);
    double end = get_time();
    amx_total_time += (end - start);
  }
  double amx_avg_time = amx_total_time / num_iterations;
  
  // Benchmark MKL GEMM
  printf("Benchmarking MKL GEMM (%d iterations)...\n", num_iterations);
  double mkl_total_time = 0.0;
  for (int iter = 0; iter < num_iterations; iter++) {
    memset(C_mkl, 0, M * N * sizeof(float));
    double start = get_time();
    MKL_gemm(A, B, C_mkl, M, N, K);
    double end = get_time();
    mkl_total_time += (end - start);
  }
  double mkl_avg_time = mkl_total_time / num_iterations;

  // Calculate GFLOPS (2*M*N*K operations)
  double gflops = (2.0 * M * N * K) / 1e9;
  double amx_gflops = gflops / amx_avg_time;
  double mkl_gflops = gflops / mkl_avg_time;
  
  // Calculate memory bandwidth (bytes read + written)
  // Read: A (M*K*2 bytes) + B (K*N*2 bytes), Write: C (M*N*4 bytes)
  double memory_bytes = (M * K * 2.0 + K * N * 2.0 + M * N * 4.0);
  double amx_bandwidth_gb = (memory_bytes / 1e9) / amx_avg_time;
  double mkl_bandwidth_gb = (memory_bytes / 1e9) / mkl_avg_time;

  // Print performance results
  printf("\n========================================\n");
  printf("PERFORMANCE RESULTS\n");
  printf("========================================\n\n");
  
  if (run_sequential) {
    printf("Sequential (Scalar Core) GEMM Performance:\n");
    printf("  Average time:  %8.3f ms\n", seq_avg_time * 1000);
    printf("  Performance:   %8.2f GFLOPS\n", seq_gflops);
    printf("  Bandwidth:     %8.2f GB/s\n\n", seq_bandwidth_gb);
  }
  
  printf("AMX GEMM Performance:\n");
  printf("  Average time:  %8.3f ms\n", amx_avg_time * 1000);
  printf("  Performance:   %8.2f GFLOPS\n", amx_gflops);
  printf("  Bandwidth:     %8.2f GB/s\n", amx_bandwidth_gb);
  printf("  Efficiency:    %8.2f%% of peak\n\n", (amx_gflops / 1000.0) * 100); // Rough estimate

  printf("MKL GEMM Performance:\n");
  printf("  Average time:  %8.3f ms\n", mkl_avg_time * 1000);
  printf("  Performance:   %8.2f GFLOPS\n", mkl_gflops);
  printf("  Bandwidth:     %8.2f GB/s\n", mkl_bandwidth_gb);
  printf("  Efficiency:    %8.2f%% of peak\n\n", (mkl_gflops / 1000.0) * 100);

  printf("Comparison:\n");
  if (run_sequential) {
    printf("  Speedup (AMX vs Sequential):  %.2fx\n", seq_avg_time / amx_avg_time);
    printf("  Speedup (MKL vs Sequential):  %.2fx\n", seq_avg_time / mkl_avg_time);
  }
  printf("  Speedup (AMX vs MKL):         %.2fx\n", mkl_avg_time / amx_avg_time);
  printf("  Winner: %s\n\n", amx_avg_time < mkl_avg_time ? "AMX" : "MKL");

  printf("========================================\n");
  printf("CORRECTNESS VERIFICATION\n");
  printf("========================================\n");
  
  if (run_sequential) {
    // Verify against sequential implementation (small matrices only)
    float max_err_amx = 0.0f;
    float max_err_mkl = 0.0f;
    
    for (int i = 0; i < M * N; i++) {
      float err_amx = fabs(C_amx[i] - C_sequential[i]);
      float err_mkl = fabs(C_mkl[i] - C_sequential[i]);
      if (err_amx > max_err_amx) max_err_amx = err_amx;
      if (err_mkl > max_err_mkl) max_err_mkl = err_mkl;
    }
    
    printf("AMX vs Sequential Reference - Max error: %e ", max_err_amx);
    printf(max_err_amx < 1e-2 ? "[PASS]\n" : "[FAIL]\n");
    
    printf("MKL vs Sequential Reference - Max error: %e ", max_err_mkl);
    printf(max_err_mkl < 1e-2 ? "[PASS]\n" : "[FAIL]\n");
  } else if (run_naive) {
    // Verify against naive implementation
    float max_err_amx = 0.0f;
    float max_err_mkl = 0.0f;
    
    for (int i = 0; i < M * N; i++) {
      float err_amx = fabs(C_amx[i] - C_ref[i]);
      float err_mkl = fabs(C_mkl[i] - C_ref[i]);
      if (err_amx > max_err_amx) max_err_amx = err_amx;
      if (err_mkl > max_err_mkl) max_err_mkl = err_mkl;
    }
    
    printf("AMX vs Reference - Max error: %e ", max_err_amx);
    printf(max_err_amx < 1e-2 ? "[PASS]\n" : "[FAIL]\n");
    
    printf("MKL vs Reference - Max error: %e ", max_err_mkl);
    printf(max_err_mkl < 1e-2 ? "[PASS]\n" : "[FAIL]\n");
  } else {
    // For large matrices, compare AMX vs MKL
    float max_err = 0.0f;
    float avg_err = 0.0f;
    int num_samples = M * N;
    
    for (int i = 0; i < M * N; i++) {
      float err = fabs(C_amx[i] - C_mkl[i]);
      avg_err += err;
      if (err > max_err) max_err = err;
    }
    avg_err /= num_samples;
    
    printf("AMX vs MKL comparison:\n");
    printf("  Max error: %e\n", max_err);
    printf("  Avg error: %e\n", avg_err);
    printf("  Status: ");
    
    // For BF16, expect some numerical differences
    if (max_err < 1e-1) {
      printf("[PASS] Results are numerically consistent\n");
    } else {
      printf("[WARNING] Large differences detected\n");
    }
  }

  // Optional: Print sample results
  printf("\n========================================\n");
  printf("SAMPLE RESULTS (First 4x4 of output)\n");
  printf("========================================\n");
  
  if (run_sequential) {
    printf("\nSequential Results:\n");
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        printf("%8.3f ", C_sequential[i * N + j]);
      }
      printf("\n");
    }
  }
  
  printf("\nAMX Results:\n");
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      printf("%8.3f ", C_amx[i * N + j]);
    }
    printf("\n");
  }

  printf("\nMKL Results:\n");
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      printf("%8.3f ", C_mkl[i * N + j]);
    }
    printf("\n");
  }

  if (run_naive && !run_sequential) {
    printf("\nReference Results:\n");
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        printf("%8.3f ", C_ref[i * N + j]);
      }
      printf("\n");
    }
  }

  // Cleanup
  mkl_free(A);
  mkl_free(B);
  mkl_free(C_sequential);
  mkl_free(C_amx);
  mkl_free(C_mkl);
  mkl_free(C_ref);

  return 0;
}