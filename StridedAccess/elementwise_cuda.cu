#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

// Configuration
#ifndef N
#define N 4192
#endif

#ifndef M
#define M 4192
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


// Thread tile selection - elements per thread (0=1x1, 1=2x2, 2=4x4, 3=8x8, 4=16x16)
#ifndef THREAD_TILE_SEL
#define THREAD_TILE_SEL 1  // default 2x2
#endif

// Scale factor
static constexpr double SCALE = 1.5;

// Thread block configuration
#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 1

// Access macros for different layouts (device)
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

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Fast pseudo-random number generator (xorshift64)
static inline uint64_t xorshift64(uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// Initialize arrays with deterministic pseudo-random values (host)
static void init_arrays(double* __restrict__ A, double* __restrict__ B) {
    uint64_t seed_a = 0x123456789ABCDEF0ULL;
    uint64_t seed_b = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[A_IDX(i, j)] = (double)(xorshift64(seed_a) % 10000) / 100.0;
            B[B_IDX(i, j)] = (double)(xorshift64(seed_b) % 10000) / 100.0;
        }
    }
}

// Kernel 0: i-outer, j-inner loop order (row-major traversal pattern)
__global__ void __launch_bounds__(128) kernel_0(double* __restrict__ A, double* __restrict__ B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * M;
    
    if (idx < total) {
        int i = idx / M;
        int j = idx % M;
        A[A_IDX(i, j)] = (A[A_IDX(i, j)] + B[B_IDX(i, j)]) * SCALE;
    }
}

// Kernel 1: j-outer, i-inner loop order (column-major traversal pattern)
__global__ void __launch_bounds__(128) kernel_1(double* __restrict__ A, double* __restrict__ B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * M;
    
    if (idx < total) {
        int j = idx / N;
        int i = idx % N;
        A[A_IDX(i, j)] = (A[A_IDX(i, j)] + B[B_IDX(i, j)]) * SCALE;
    }
}

// Kernel 2: Tiled - each thread computes THREAD_TILE x THREAD_TILE elements
template<int TILE_SIZE, int THREAD_TILE>
__global__ void __launch_bounds__(128) kernel_2_impl(double* __restrict__ A, double* __restrict__ B) {
    // Position of this thread-tile within the tile

    int ii = threadIdx.x * THREAD_TILE + blockIdx.x * THREAD_TILE * BLOCK_SIZE_X;
    int jj = threadIdx.y * THREAD_TILE + blockIdx.y * BLOCK_SIZE_Y;

    // Each thread computes THREAD_TILE x THREAD_TILE elements
    #pragma unroll
    for (int di = 0; di < THREAD_TILE; di++) {
        #pragma unroll
        for (int dj = 0; dj < THREAD_TILE; dj++) {
            int i = ii + di;
            int j = jj + dj;
            
            if (i < N && j < M) {
                A[A_IDX(i, j)] = (A[A_IDX(i, j)] + B[B_IDX(i, j)]) * SCALE;
            }
        }
    }
}

// Kernel 3: Tiled with shared memory - each thread computes THREAD_TILE x THREAD_TILE elements
template<int TILE_SIZE, int THREAD_TILE>
__global__ void kernel_3_impl(double* __restrict__ A, double* __restrict__ B) {
    // Shared memory tile dimensions = what this entire block processes
    constexpr int TILE_ROWS = BLOCK_SIZE_X * THREAD_TILE;  // 128 * THREAD_TILE
    constexpr int TILE_COLS = BLOCK_SIZE_Y * THREAD_TILE;  // 1 * THREAD_TILE
    constexpr int TILE_ELEMENTS = TILE_ROWS * TILE_COLS;
    
    extern __shared__ double smem[];
    double* A_tile = smem;
    double* B_tile = smem + TILE_ELEMENTS;
    double* C_tile = smem + 2 * TILE_ELEMENTS;
    
    // Block's starting position in global memory
    int block_ii = blockIdx.x * TILE_ROWS;
    int block_jj = blockIdx.y * TILE_COLS;
    
    // Cooperative load of A tile to shared memory
    #pragma unroll
    for (int idx = threadIdx.x; idx < TILE_ELEMENTS; idx += BLOCK_SIZE_X) {
        int ti = idx / TILE_COLS;
        int tj = idx % TILE_COLS;
        int i = block_ii + ti;
        int j = block_jj + tj;
        
        if (i < N && j < M) {
            A_tile[idx] = A[A_IDX(i, j)];
        } else {
            A_tile[idx] = 0.0;
        }
    }
    
    // Cooperative load of B tile to shared memory
    #pragma unroll
    for (int idx = threadIdx.x; idx < TILE_ELEMENTS; idx += BLOCK_SIZE_X) {
        int ti = idx / TILE_COLS;
        int tj = idx % TILE_COLS;
        int i = block_ii + ti;
        int j = block_jj + tj;
        
        if (i < N && j < M) {
            B_tile[idx] = B[B_IDX(i, j)];
        } else {
            B_tile[idx] = 0.0;
        }
    }
    
    __syncthreads();
    
    // Each thread's local position within shared memory tile
    int local_ii = threadIdx.x * THREAD_TILE;
    int local_jj = threadIdx.y * THREAD_TILE;  // = 0 since BLOCK_SIZE_Y = 1
    
    // Compute on shared memory with unrolled loops
    #pragma unroll
    for (int di = 0; di < THREAD_TILE; di++) {
        #pragma unroll
        for (int dj = 0; dj < THREAD_TILE; dj++) {
            int ti = local_ii + di;
            int tj = local_jj + dj;
            int smem_idx = ti * TILE_COLS + tj;
            
            C_tile[smem_idx] = (A_tile[smem_idx] + B_tile[smem_idx]) * SCALE;
        }
    }
    
    __syncthreads();
    
    // Cooperative store of C tile back to global memory
    #pragma unroll
    for (int idx = threadIdx.x; idx < TILE_ELEMENTS; idx += BLOCK_SIZE_X) {
        int ti = idx / TILE_COLS;
        int tj = idx % TILE_COLS;
        int i = block_ii + ti;
        int j = block_jj + tj;
        
        if (i < N && j < M) {
            A[A_IDX(i, j)] = C_tile[idx];
        }
    }
}

// Host wrapper functions
void launch_kernel_0(double* d_A, double* d_B) {
    int total = N * M;
    int num_blocks = (total + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    kernel_0<<<grid, block>>>(d_A, d_B);
}

void launch_kernel_1(double* d_A, double* d_B) {
    int total = N * M;
    int num_blocks = (total + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    kernel_1<<<grid, block>>>(d_A, d_B);
}

template<int TILE_SIZE, int THREAD_TILE>
void launch_kernel_2(double* d_A, double* d_B) {
    int totalX = N;
    int totalY = M;
    int num_blocksX = (totalX + BLOCK_SIZE_X * THREAD_TILE - 1) / (BLOCK_SIZE_X * THREAD_TILE);
    int num_blocksY = (totalY + BLOCK_SIZE_Y * THREAD_TILE - 1) / (BLOCK_SIZE_Y * THREAD_TILE);
    dim3 grid(num_blocksX, num_blocksY);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    kernel_2_impl<TILE_SIZE, THREAD_TILE><<<grid, block>>>(d_A, d_B);
}

template<int TILE_SIZE, int THREAD_TILE>
void launch_kernel_3(double* d_A, double* d_B) {
    int totalX = N;
    int totalY = M;
    int num_blocksX = (totalX + BLOCK_SIZE_X * THREAD_TILE - 1) / (BLOCK_SIZE_X * THREAD_TILE);
    int num_blocksY = (totalY + BLOCK_SIZE_Y * THREAD_TILE - 1) / (BLOCK_SIZE_Y * THREAD_TILE);
    dim3 grid(num_blocksX, num_blocksY);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Shared memory size: 3 tiles (A, B, C)
    size_t smem_size = 3 * TILE_SIZE * TILE_SIZE * sizeof(double);
    
    kernel_3_impl<TILE_SIZE, THREAD_TILE><<<grid, block, smem_size>>>(d_A, d_B);
}

// Thread tile size mapping
constexpr int get_thread_tile_size() {
#if THREAD_TILE_SEL == 0
    return 1;
#elif THREAD_TILE_SEL == 1
    return 2;
#elif THREAD_TILE_SEL == 2
    return 4;
#elif THREAD_TILE_SEL == 3
    return 8;
#elif THREAD_TILE_SEL == 4
    return 16;
#else
    return 2;
#endif
}

// Dispatch based on THREAD_TILE_SEL (TILE_SIZE fixed at 128)
#define DISPATCH_KERNEL_2(TILE_SIZE) \
    do { \
        constexpr int TT = get_thread_tile_size(); \
        static_assert(TILE_SIZE >= TT, "TILE_SIZE must be >= THREAD_TILE"); \
        static_assert(TILE_SIZE % TT == 0, "TILE_SIZE must be divisible by THREAD_TILE"); \
        launch_kernel_2<TILE_SIZE, TT>(d_A, d_B); \
    } while(0)

#define DISPATCH_KERNEL_3(TILE_SIZE) \
    do { \
        constexpr int TT = get_thread_tile_size(); \
        static_assert(TILE_SIZE >= TT, "TILE_SIZE must be >= THREAD_TILE"); \
        static_assert(TILE_SIZE % TT == 0, "TILE_SIZE must be divisible by THREAD_TILE"); \
        launch_kernel_3<TILE_SIZE, TT>(d_A, d_B); \
    } while(0)

void dispatch_kernel_2(double* d_A, double* d_B) {
    DISPATCH_KERNEL_2(128);
}

void dispatch_kernel_3(double* d_A, double* d_B) {
    DISPATCH_KERNEL_3(128);
}

// Checksum for verification (host)
static double checksum(double* A) {
    double sum = 0.0;
    for (int i = 0; i < N * M; i++) {
        sum += A[i];
    }
    return sum;
}

int main(int argc, char** argv) {
    // Host arrays
    double* h_A = (double*)aligned_alloc(64, N * M * sizeof(double));
    double* h_B = (double*)aligned_alloc(64, N * M * sizeof(double));
    
    if (!h_A || !h_B) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }
    
    // Device arrays
    double *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, N * M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * M * sizeof(double)));
    
    // Initialize host arrays
    init_arrays(h_A, h_B);
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * M * sizeof(double), cudaMemcpyHostToDevice));
    
    int thread_tile_sizes[] = {1, 2, 4, 8, 16};
    int thread_tile_size = thread_tile_sizes[THREAD_TILE_SEL < 5 ? THREAD_TILE_SEL : 1];
    
    // Warmup run
#if KERNEL == 0
    launch_kernel_0(d_A, d_B);
#elif KERNEL == 1
    launch_kernel_1(d_A, d_B);
#elif KERNEL == 2
    dispatch_kernel_2(d_A, d_B);
#else
    dispatch_kernel_3(d_A, d_B);
#endif
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Re-initialize for timed run
    init_arrays(h_A, h_B);
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * M * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * M * sizeof(double), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    // Execute kernel
#if KERNEL == 0
    launch_kernel_0(d_A, d_B);
#elif KERNEL == 1
    launch_kernel_1(d_A, d_B);
#elif KERNEL == 2
    dispatch_kernel_2(d_A, d_B);
#else
    dispatch_kernel_3(d_A, d_B);
#endif
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_A, d_A, N * M * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Compute checksum for verification
    double cs = checksum(h_A);
    
    // Output: kernel,a_layout,b_layout,block_tile,thread_tile,time_ms,gpu_metric,checksum
    printf("%d,%d,%d,%d,%d,%.6f,%lld,%.6f\n", KERNEL, A_LAYOUT, B_LAYOUT, 128, 
           thread_tile_size, (double)elapsed_ms, 0LL, cs);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);
    
    return 0;
}
