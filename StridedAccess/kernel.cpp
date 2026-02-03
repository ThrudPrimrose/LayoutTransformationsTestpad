#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include "papi_wrapper.h"

// Configuration
#ifndef N
#define N 2048
#endif

#ifndef M
#define M 2048
#endif

#ifndef KERNEL
#define KERNEL 0
#endif

// Data layout flags (0 = row-major, 1 = col-major)
#ifndef A_LAYOUT
#define A_LAYOUT 0
#endif

#ifndef B_LAYOUT
#define B_LAYOUT 0
#endif

#ifndef PAPI_METRIC
#define PAPI_METRIC ""
#endif

// Tile size selection (0=4, 1=8, 2=16, 3=32, 4=64, 5=128)
#ifndef TILE_SEL
#define TILE_SEL 3  // default 32x32
#endif

// Scale factor
static constexpr double SCALE = 1.5;

// Access macros for different layouts
#if A_LAYOUT == 0
    #define A_IDX(i, j) ((i) * M + (j))
#else
    #define A_IDX(i, j) ((j) * N + (i))
#endif

#if B_LAYOUT == 0
    #define B_IDX(i, j) ((i) * M + (j))
#else
    #define B_IDX(i, j) ((j) * N + (i))
#endif

// Fast pseudo-random number generator (xorshift64)
static inline uint64_t xorshift64(uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// Initialize arrays with deterministic pseudo-random values
static void init_arrays(double* __restrict__ A, double* __restrict__ B) {
    uint64_t seed_a = 0x123456789ABCDEF0ULL;
    uint64_t seed_b = 0xFEDCBA9876543210ULL;
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[A_IDX(i, j)] = (double)(xorshift64(seed_a) % 10000) / 100.0;
            B[B_IDX(i, j)] = (double)(xorshift64(seed_b) % 10000) / 100.0;
        }
    }
}

// Kernel 0: i-outer, j-inner loop order
static void kernel_0(double* __restrict__ A, double* __restrict__ B) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        __asm__ volatile("" ::: "memory");
        for (int j = 0; j < M; j++) {
            A[A_IDX(i, j)] = (A[A_IDX(i, j)] + B[B_IDX(i, j)]) * SCALE;
        }
    }
}

// Kernel 1: j-outer, i-inner loop order
static void kernel_1(double* __restrict__ A, double* __restrict__ B) {
    #pragma omp parallel for
    for (int j = 0; j < M; j++) {
        __asm__ volatile("" ::: "memory");
        for (int i = 0; i < N; i++) {
            A[A_IDX(i, j)] = (A[A_IDX(i, j)] + B[B_IDX(i, j)]) * SCALE;
        }
    }
}

// Kernel 2: Tiled/blocked - template version
template<int TILE_SIZE>
static void kernel_2_impl(double* __restrict__ A, double* __restrict__ B) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < M; jj += TILE_SIZE) {
            __asm__ volatile("" ::: "memory");
            
            int i_end = (ii + TILE_SIZE <= N) ? ii + TILE_SIZE : N;
            int j_end = (jj + TILE_SIZE <= M) ? jj + TILE_SIZE : M;
            
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    A[A_IDX(i, j)] = (A[A_IDX(i, j)] + B[B_IDX(i, j)]) * SCALE;
                }
            }
        }
    }
}

// Kernel 3: Tiled with explicit copy - template version
template<int TILE_SIZE>
static void kernel_3_impl(double* __restrict__ A, double* __restrict__ B) {
    const int num_tiles_i = (N + TILE_SIZE - 1) / TILE_SIZE;
    const int num_tiles_j = (M + TILE_SIZE - 1) / TILE_SIZE;
    const int total_tiles = num_tiles_i * num_tiles_j;
    
    #pragma omp parallel for schedule(static)
    for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
        // Thread-local tile buffers
        alignas(64) double A_tile[TILE_SIZE * TILE_SIZE];
        alignas(64) double B_tile[TILE_SIZE * TILE_SIZE];
        alignas(64) double C_tile[TILE_SIZE * TILE_SIZE];
        
        int tile_i = tile_idx / num_tiles_j;
        int tile_j = tile_idx % num_tiles_j;
        int ii = tile_i * TILE_SIZE;
        int jj = tile_j * TILE_SIZE;

        constexpr int tile_h = TILE_SIZE;
        constexpr int tile_w = TILE_SIZE;
        
        // Copy A tile to local buffer
        for (int ti = 0; ti < tile_h; ti++) {
            for (int tj = 0; tj < tile_w; tj++) {
                A_tile[ti * TILE_SIZE + tj] = A[A_IDX(ii + ti, jj + tj)];
            }
        }
        
        // Copy B tile to local buffer
        for (int ti = 0; ti < tile_h; ti++) {
            for (int tj = 0; tj < tile_w; tj++) {
                B_tile[ti * TILE_SIZE + tj] = B[B_IDX(ii + ti, jj + tj)];
            }
        }
        
        // Compute on local buffers
        for (int ti = 0; ti < tile_h; ti++) {
            for (int tj = 0; tj < tile_w; tj++) {
                C_tile[ti * TILE_SIZE + tj] = 
                    (A_tile[ti * TILE_SIZE + tj] + B_tile[ti * TILE_SIZE + tj]) * SCALE;
            }
        }
        
        // Write C tile back
        for (int ti = 0; ti < tile_h; ti++) {
            for (int tj = 0; tj < tile_w; tj++) {
                A[A_IDX(ii + ti, jj + tj)] = C_tile[ti * TILE_SIZE + tj];
            }
        }
    }
}

// Dispatch wrappers that select tile size at compile time
static void kernel_2(double* __restrict__ A, double* __restrict__ B) {
#if TILE_SEL == 0
    kernel_2_impl<4>(A, B);
#elif TILE_SEL == 1
    kernel_2_impl<8>(A, B);
#elif TILE_SEL == 2
    kernel_2_impl<16>(A, B);
#elif TILE_SEL == 3
    kernel_2_impl<32>(A, B);
#elif TILE_SEL == 4
    kernel_2_impl<64>(A, B);
#elif TILE_SEL == 5
    kernel_2_impl<128>(A, B);
#else
    kernel_2_impl<32>(A, B);
#endif
}

static void kernel_3(double* __restrict__ A, double* __restrict__ B) {
#if TILE_SEL == 0
    kernel_3_impl<4>(A, B);
#elif TILE_SEL == 1
    kernel_3_impl<8>(A, B);
#elif TILE_SEL == 2
    kernel_3_impl<16>(A, B);
#elif TILE_SEL == 3
    kernel_3_impl<32>(A, B);
#elif TILE_SEL == 4
    kernel_3_impl<64>(A, B);
#elif TILE_SEL == 5
    kernel_3_impl<128>(A, B);
#else
    kernel_3_impl<32>(A, B);
#endif
}

// Checksum for verification
static double checksum(double* A) {
    double sum = 0.0;
    for (int i = 0; i < N * M; i++) {
        sum += A[i];
    }
    return sum;
}

int main(int argc, char** argv) {
    // Allocate aligned arrays
    double* A = (double*)aligned_alloc(64, N * M * sizeof(double));
    double* B = (double*)aligned_alloc(64, N * M * sizeof(double));
    
    if (!A || !B) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize arrays
    init_arrays(A, B);
    
    // Get PAPI metric from compile-time define
    const char* papi_metric = PAPI_METRIC;
    init_papi_single_thread(papi_metric);
    
    // Tile sizes for reporting
    int tile_sizes[] = {4, 8, 16, 32, 64, 128};
    int tile_size = tile_sizes[TILE_SEL < 6 ? TILE_SEL : 3];
    
    // Warmup run
#if KERNEL == 0
    kernel_0(A, B);
#elif KERNEL == 1
    kernel_1(A, B);
#elif KERNEL == 2
    kernel_2(A, B);
#else
    kernel_3(A, B);
#endif
    
    // Re-initialize for timed run
    init_arrays(A, B);
    
    // Start timing and PAPI
    start_papi_single_thread();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute kernel
#if KERNEL == 0
    kernel_0(A, B);
#elif KERNEL == 1
    kernel_1(A, B);
#elif KERNEL == 2
    kernel_2(A, B);
#else
    kernel_3(A, B);
#endif
    
    auto end = std::chrono::high_resolution_clock::now();
    stop_papi_single_thread();
    
    // Calculate timing
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    long long papi_value = get_papi_value();
    
    // Compute checksum for verification
    double cs = checksum(A);
    
    // Output: kernel,a_layout,b_layout,tile_size,time_ms,papi_value,checksum
    printf("%d,%d,%d,%d,%.6f,%lld,%.6f\n", KERNEL, A_LAYOUT, B_LAYOUT, tile_size, elapsed_ms, papi_value, cs);
    
    free(A);
    free(B);
    
    return 0;
}