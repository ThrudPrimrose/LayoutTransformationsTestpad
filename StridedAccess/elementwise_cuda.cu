#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

// Configuration
// Matrix dimensions: M rows x N columns
#ifndef M
#define M 4192
#endif

#ifndef N
#define N 4192
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
// Matrix is M rows x N columns
// i = row index [0, M), j = column index [0, N)
#if A_LAYOUT == 0
    #define A_IDX(i, j) ((i) * N + (j))  // row-major: row * num_cols + col
#else
    #define A_IDX(i, j) ((j) * M + (i))  // col-major: col * num_rows + row
#endif

#if B_LAYOUT == 0
    #define B_IDX(i, j) ((i) * N + (j))
#else
    #define B_IDX(i, j) ((j) * M + (i))
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
// Matrix is M rows x N columns
static void init_arrays(double* __restrict__ A, double* __restrict__ B) {
    uint64_t seed_a = 0x123456789ABCDEF0ULL;
    uint64_t seed_b = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < M; i++) {       // i = row [0, M)
        for (int j = 0; j < N; j++) {   // j = col [0, N)
            A[A_IDX(i, j)] = (double)(xorshift64(seed_a) % 10000) / 100.0;
            B[B_IDX(i, j)] = (double)(xorshift64(seed_b) % 10000) / 100.0;
        }
    }
}

// Kernel 0: i-outer, j-inner loop order (row-major traversal pattern)
__global__ void __launch_bounds__(128) kernel_0(double* __restrict__ A, double* __restrict__ B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        int i = idx / N;  // row
        int j = idx % N;  // column
        A[A_IDX(i, j)] = (A[A_IDX(i, j)] + B[B_IDX(i, j)]) * SCALE;
    }
}

// Kernel 1: j-outer, i-inner loop order (column-major traversal pattern)
__global__ void __launch_bounds__(128) kernel_1(double* __restrict__ A, double* __restrict__ B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        int j = idx / M;  // column
        int i = idx % M;  // row
        A[A_IDX(i, j)] = (A[A_IDX(i, j)] + B[B_IDX(i, j)]) * SCALE;
    }
}

// Kernel 2: Tiled - each thread computes THREAD_TILE x THREAD_TILE elements
// Consecutive threads (threadIdx.x) access consecutive columns (j) for row-major coalescing
template<int TILE_SIZE, int THREAD_TILE>
__global__ void __launch_bounds__(128) kernel_2_impl(double* __restrict__ A, double* __restrict__ B) {
    // Block tile dimensions
    constexpr int TILE_COLS = BLOCK_SIZE_X * THREAD_TILE;  // columns covered by block
    constexpr int TILE_ROWS = BLOCK_SIZE_Y * THREAD_TILE;  // rows covered by block
    
    // Block origin in global coordinates
    // blockIdx.x -> columns (j), blockIdx.y -> rows (i)
    int block_col = blockIdx.x * TILE_COLS;
    int block_row = blockIdx.y * TILE_ROWS;

    // Each thread computes THREAD_TILE x THREAD_TILE elements
    // Strided pattern - consecutive threads access consecutive j values
    #pragma unroll
    for (int di = 0; di < THREAD_TILE; di++) {
        int i = block_row + threadIdx.y + di * BLOCK_SIZE_Y;
        
        #pragma unroll
        for (int dj = 0; dj < THREAD_TILE; dj++) {
            // Consecutive threadIdx.x values give consecutive j values
            int j = block_col + threadIdx.x + dj * BLOCK_SIZE_X;

            if (i < M && j < N) {
                A[A_IDX(i, j)] = (A[A_IDX(i, j)] + B[B_IDX(i, j)]) * SCALE;
            }
        }
    }
}

template<int TILE_SIZE, int THREAD_TILE>
__global__ void __launch_bounds__(128)
kernel_3_impl(double* __restrict__ A, double* __restrict__ B)
{
    // Tile dimensions
    constexpr int TILE_COLS = BLOCK_SIZE_X * THREAD_TILE; // columns (j)
    constexpr int TILE_ROWS = BLOCK_SIZE_Y * THREAD_TILE; // rows (i)
    constexpr int TILE_ELEMENTS = TILE_ROWS * TILE_COLS;

    extern __shared__ double smem[];
    double* __restrict__ A_tile = smem;
    double* __restrict__ B_tile = smem + TILE_ELEMENTS;
    double* __restrict__ C_tile = smem + 2 * TILE_ELEMENTS;

    // Block origin - blockIdx.x -> columns, blockIdx.y -> rows
    int block_col = blockIdx.x * TILE_COLS;  // j dimension
    int block_row = blockIdx.y * TILE_ROWS;  // i dimension

    // Cooperative load A: global → shared
    // Coalescing strategy depends on A's layout
    #pragma unroll
    for (int idx = threadIdx.x; idx < TILE_ELEMENTS; idx += BLOCK_SIZE_X) {
#if A_LAYOUT == 0
        // Row-major: consecutive threads access consecutive columns (j)
        // Memory: A[i * N + j], so consecutive j = consecutive addresses
        int ti = idx / TILE_COLS;
        int tj = idx % TILE_COLS;
#else
        // Col-major: consecutive threads access consecutive rows (i)
        // Memory: A[j * M + i], so consecutive i = consecutive addresses
        int ti = idx % TILE_ROWS;
        int tj = idx / TILE_ROWS;
#endif
        int i = block_row + ti;
        int j = block_col + tj;
        int smem_idx = ti * TILE_COLS + tj;  // shared mem always row-major

        if (i < M && j < N) {
            A_tile[smem_idx] = A[A_IDX(i, j)];
        } else {
            A_tile[smem_idx] = 0.0;
        }
    }

    // Cooperative load B: global → shared
    // Coalescing strategy depends on B's layout
    #pragma unroll
    for (int idx = threadIdx.x; idx < TILE_ELEMENTS; idx += BLOCK_SIZE_X) {
#if B_LAYOUT == 0
        // Row-major: consecutive threads access consecutive columns (j)
        int ti = idx / TILE_COLS;
        int tj = idx % TILE_COLS;
#else
        // Col-major: consecutive threads access consecutive rows (i)
        int ti = idx % TILE_ROWS;
        int tj = idx / TILE_ROWS;
#endif
        int i = block_row + ti;
        int j = block_col + tj;
        int smem_idx = ti * TILE_COLS + tj;

        if (i < M && j < N) {
            B_tile[smem_idx] = B[B_IDX(i, j)];
        } else {
            B_tile[smem_idx] = 0.0;
        }
    }

    __syncthreads();

    // Per-thread computation using strided access
    // Shared memory is row-major, so this is always efficient
    #pragma unroll
    for (int di = 0; di < THREAD_TILE; di++) {
        int ti = threadIdx.y + di * BLOCK_SIZE_Y;  // row in tile
        
        #pragma unroll
        for (int dj = 0; dj < THREAD_TILE; dj++) {
            int tj = threadIdx.x + dj * BLOCK_SIZE_X;  // col in tile
            int smem_idx = ti * TILE_COLS + tj;

            C_tile[smem_idx] = (A_tile[smem_idx] + B_tile[smem_idx]) * SCALE;
        }
    }

    __syncthreads();

    // Cooperative store: shared → global
    // Coalescing strategy depends on A's layout (output goes to A)
    #pragma unroll
    for (int idx = threadIdx.x; idx < TILE_ELEMENTS; idx += BLOCK_SIZE_X) {
#if A_LAYOUT == 0
        // Row-major: consecutive threads access consecutive columns
        int ti = idx / TILE_COLS;
        int tj = idx % TILE_COLS;
#else
        // Col-major: consecutive threads access consecutive rows
        int ti = idx % TILE_ROWS;
        int tj = idx / TILE_ROWS;
#endif
        int i = block_row + ti;
        int j = block_col + tj;
        int smem_idx = ti * TILE_COLS + tj;

        if (i < M && j < N) {
            A[A_IDX(i, j)] = C_tile[smem_idx];
        }
    }
}

// Host wrapper functions
void launch_kernel_0(double* d_A, double* d_B) {
    int total = M * N;
    int num_blocks = (total + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    kernel_0<<<grid, block>>>(d_A, d_B);
}

void launch_kernel_1(double* d_A, double* d_B) {
    int total = M * N;
    int num_blocks = (total + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    kernel_1<<<grid, block>>>(d_A, d_B);
}

// grid.x covers columns (N), grid.y covers rows (M)
template<int TILE_SIZE, int THREAD_TILE>
void launch_kernel_2(double* d_A, double* d_B) {
    constexpr int TILE_COLS = BLOCK_SIZE_X * THREAD_TILE;
    constexpr int TILE_ROWS = BLOCK_SIZE_Y * THREAD_TILE;
    
    // x dimension covers columns (N), y dimension covers rows (M)
    int num_blocks_x = (N + TILE_COLS - 1) / TILE_COLS;  // columns
    int num_blocks_y = (M + TILE_ROWS - 1) / TILE_ROWS;  // rows
    
    dim3 grid(num_blocks_x, num_blocks_y);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    kernel_2_impl<TILE_SIZE, THREAD_TILE><<<grid, block>>>(d_A, d_B);
}

template<int TILE_SIZE, int THREAD_TILE>
void launch_kernel_3(double* d_A, double* d_B) {
    constexpr int TILE_COLS = BLOCK_SIZE_X * THREAD_TILE;
    constexpr int TILE_ROWS = BLOCK_SIZE_Y * THREAD_TILE;
    
    // x dimension covers columns (N), y dimension covers rows (M)
    int num_blocks_x = (N + TILE_COLS - 1) / TILE_COLS;  // columns
    int num_blocks_y = (M + TILE_ROWS - 1) / TILE_ROWS;  // rows
    
    dim3 grid(num_blocks_x, num_blocks_y);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Shared memory size: 3 tiles (A, B, C)
    constexpr size_t smem_size = 3 * TILE_ROWS * TILE_COLS * sizeof(double);
    
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
    for (int i = 0; i < M * N; i++) {
        sum += A[i];
    }
    return sum;
}

int main(int argc, char** argv) {
    // Host arrays
    double* h_A = (double*)aligned_alloc(64, M * N * sizeof(double));
    double* h_B = (double*)aligned_alloc(64, M * N * sizeof(double));
    
    if (!h_A || !h_B) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }
    
    // Device arrays
    double *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, M * N * sizeof(double)));
    
    // Initialize host arrays
    init_arrays(h_A, h_B);
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, M * N * sizeof(double), cudaMemcpyHostToDevice));
    
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
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, M * N * sizeof(double), cudaMemcpyHostToDevice));
    
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
    CUDA_CHECK(cudaMemcpy(h_A, d_A, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    
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
