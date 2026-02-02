#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>


#ifdef PAPI_ENABLED
#include "papi_metrics.h"
#endif

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

// Alignment in bytes (cache line size)
#define ALIGN_BYTES 64
#define ALIGN_DOUBLES (ALIGN_BYTES / sizeof(double))  // 8 doubles

// Round up to multiple of ALIGN_DOUBLES
#define ALIGN_UP(x) (((x) + ALIGN_DOUBLES - 1) / ALIGN_DOUBLES * ALIGN_DOUBLES)

/*
 * SoA Block Layout with 64-byte alignment:
 * -----------------------------------------
 * Each local position (lj, li) has an array of BM*BN block values.
 * We pad num_blocks to a multiple of 8 (64 bytes / 8 bytes per double)
 * so each position's block array starts at a 64-byte boundary.
 * 
 * offset(bi, bj, lj, li) = (lj * BW_halo + li) * num_blocks_padded + bi * BN + bj
 */

typedef struct {
    int N;                    // Interior grid size (N x N)
    int BM, BN;               // Number of blocks in y and x
    int BH, BW;               // Interior size of each block (without halo)
    int BH_halo, BW_halo;     // Block size with halo (BH+2, BW+2)
    int num_blocks;           // BM * BN (actual)
    int num_blocks_padded;    // Padded to multiple of ALIGN_DOUBLES for 64-byte alignment
    int positions_per_block;  // BH_halo * BW_halo
    int total_size;           // Total array size (with padding)
    double *A;
    double *B;
} Jacobi2DSoA;

// SoA offset with alignment padding
static inline int soa_offset(const Jacobi2DSoA *s, int bi, int bj, int lj, int li) {
    return (lj * s->BW_halo + li) * s->num_blocks_padded + bi * s->BN + bj;
}

// Convert global (j, i) to block indices and local coordinates
static inline void global_to_block_local(const Jacobi2DSoA *s, int j, int i,
                                         int *bi, int *bj, int *lj, int *li) {
    if (j <= 0) {
        *bi = 0;
        *lj = 0;
    } else if (j > s->N) {
        *bi = s->BM - 1;
        *lj = s->BH + 1;
    } else {
        *bi = (j - 1) / s->BH;
        if (*bi >= s->BM) *bi = s->BM - 1;
        *lj = (j - 1) - (*bi) * s->BH + 1;
    }
    
    if (i <= 0) {
        *bj = 0;
        *li = 0;
    } else if (i > s->N) {
        *bj = s->BN - 1;
        *li = s->BW + 1;
    } else {
        *bj = (i - 1) / s->BW;
        if (*bj >= s->BN) *bj = s->BN - 1;
        *li = (i - 1) - (*bj) * s->BW + 1;
    }
}

static inline double soa_get(const Jacobi2DSoA *s, const double *arr, int j, int i) {
    int bi, bj, lj, li;
    global_to_block_local(s, j, i, &bi, &bj, &lj, &li);
    return arr[soa_offset(s, bi, bj, lj, li)];
}

static inline void soa_set(const Jacobi2DSoA *s, double *arr, int j, int i, double val) {
    int bi, bj, lj, li;
    global_to_block_local(s, j, i, &bi, &bj, &lj, &li);
    arr[soa_offset(s, bi, bj, lj, li)] = val;
}

void jacobi_init(Jacobi2DSoA *s, int N, int BM, int BN) {
    s->N = N;
    s->BM = BM;
    s->BN = BN;
    
    s->BH = (N + BM - 1) / BM;
    s->BW = (N + BN - 1) / BN;
    s->BH_halo = s->BH + 2;
    s->BW_halo = s->BW + 2;
    s->num_blocks = BM * BN;
    
    // Pad num_blocks to multiple of 8 for 64-byte alignment
    s->num_blocks_padded = ALIGN_UP(s->num_blocks);
    
    s->positions_per_block = s->BH_halo * s->BW_halo;
    s->total_size = s->positions_per_block * s->num_blocks_padded;
    
    // Allocate with 64-byte alignment
    s->A = (double*)aligned_alloc(ALIGN_BYTES, s->total_size * sizeof(double));
    s->B = (double*)aligned_alloc(ALIGN_BYTES, s->total_size * sizeof(double));
    
    if (!s->A || !s->B) {
        fprintf(stderr, "Failed to allocate aligned memory\n");
        exit(1);
    }
    
    // Zero out (including padding)
    memset(s->A, 0, s->total_size * sizeof(double));
    memset(s->B, 0, s->total_size * sizeof(double));
    
    printf("Alignment info:\n");
    printf("  num_blocks = %d, num_blocks_padded = %d\n", s->num_blocks, s->num_blocks_padded);
    printf("  Block size with halo: %d x %d = %d positions\n", 
           s->BH_halo, s->BW_halo, s->positions_per_block);
    printf("  Total array size: %d doubles (%.2f MB)\n", 
           s->total_size, s->total_size * sizeof(double) / (1024.0 * 1024.0));
    printf("  Each position's block array: %d doubles = %lu bytes (64-byte aligned: %s)\n",
           s->num_blocks_padded, s->num_blocks_padded * sizeof(double),
           (s->num_blocks_padded * sizeof(double)) % 64 == 0 ? "yes" : "no");
}

void jacobi_free(Jacobi2DSoA *s) {
    free(s->A);
    free(s->B);
}

void jacobi_initialize_data(Jacobi2DSoA *s) {
    int N = s->N;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j <= N + 1; j++) {
        for (int i = 0; i <= N + 1; i++) {
            soa_set(s, s->A, j, i, (double)(j * (i + 2)) / N);
            soa_set(s, s->B, j, i, (double)(j * (i + 3)) / N);
        }
    }
}

// Synchronize halos between adjacent blocks
void sync_halos(Jacobi2DSoA *s, double *arr) {
    int BM = s->BM;
    int BN = s->BN;
    int BH = s->BH;
    int BW = s->BW;
    int num_blocks_padded = s->num_blocks_padded;
    int BW_halo = s->BW_halo;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < BM; bi++) {
        for (int bj = 0; bj < BN; bj++) {
            // North halo (lj = 0)
            if (bi > 0) {
                #pragma omp simd
                for (int li = 1; li <= BW; li++) {
                    int src_off = (BH * BW_halo + li) * num_blocks_padded + (bi - 1) * BN + bj;
                    int dst_off = (0 * BW_halo + li) * num_blocks_padded + bi * BN + bj;
                    arr[dst_off] = arr[src_off];
                }
            }
            
            // South halo (lj = BH+1)
            if (bi < BM - 1) {
                #pragma omp simd
                for (int li = 1; li <= BW; li++) {
                    int src_off = (1 * BW_halo + li) * num_blocks_padded + (bi + 1) * BN + bj;
                    int dst_off = ((BH + 1) * BW_halo + li) * num_blocks_padded + bi * BN + bj;
                    arr[dst_off] = arr[src_off];
                }
            }
            
            // West halo (li = 0)
            if (bj > 0) {
                #pragma omp simd
                for (int lj = 1; lj <= BH; lj++) {
                    int src_off = (lj * BW_halo + BW) * num_blocks_padded + bi * BN + (bj - 1);
                    int dst_off = (lj * BW_halo + 0) * num_blocks_padded + bi * BN + bj;
                    arr[dst_off] = arr[src_off];
                }
            }
            
            // East halo (li = BW+1)
            if (bj < BN - 1) {
                #pragma omp simd
                for (int lj = 1; lj <= BH; lj++) {
                    int src_off = (lj * BW_halo + 1) * num_blocks_padded + bi * BN + (bj + 1);
                    int dst_off = (lj * BW_halo + BW + 1) * num_blocks_padded + bi * BN + bj;
                    arr[dst_off] = arr[src_off];
                }
            }
        }
    }
}

// Jacobi iteration kernel with alignment hints for vectorization
void jacobi_iteration(double * __restrict__ in, double * __restrict__ out, Jacobi2DSoA *s) {
    const int BM = s->BM;
    const int BN = s->BN;
    const int BH = s->BH;
    const int BW = s->BW;
    const int N = s->N;
    const int num_blocks_padded = s->num_blocks_padded;
    const int BW_halo = s->BW_halo;
    const int num_blocks = s->num_blocks;
    
    // Parallel loop over local positions within blocks
    // This allows vectorization across blocks at the same position
    #pragma omp parallel for collapse(2) schedule(static)
    for (int lj = 1; lj <= BH; lj++) {
        for (int li = 1; li <= BW; li++) {
            // Base offset for this local position across all blocks
            int base_c = (lj * BW_halo + li) * num_blocks_padded;
            int base_l = (lj * BW_halo + (li - 1)) * num_blocks_padded;
            int base_r = (lj * BW_halo + (li + 1)) * num_blocks_padded;
            int base_u = ((lj - 1) * BW_halo + li) * num_blocks_padded;
            int base_d = ((lj + 1) * BW_halo + li) * num_blocks_padded;
            
            // Pointers to aligned arrays for this position
            double * __restrict__ p_out = out + base_c;
            const double * __restrict__ p_c = in + base_c;
            const double * __restrict__ p_l = in + base_l;
            const double * __restrict__ p_r = in + base_r;
            const double * __restrict__ p_u = in + base_u;
            const double * __restrict__ p_d = in + base_d;

            // Vectorized loop across all blocks
            // Only process valid blocks (not padding)
            #pragma omp simd
            for (int b = 0; b < num_blocks; b++) {
                p_out[b] = 0.2 * (p_c[b] + p_l[b] + p_r[b] + p_u[b] + p_d[b]);
            }
        }
    }
    
    // Handle edge cases: blocks that extend beyond N
    // For blocks at the boundary, some local positions may be outside the domain
    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < BM; bi++) {
        for (int bj = 0; bj < BN; bj++) {
            int global_j_end = (bi + 1) * BH;
            int global_i_end = (bj + 1) * BW;
            
            // If this block extends beyond N, zero out the invalid interior points
            if (global_j_end > N || global_i_end > N) {
                for (int lj = 1; lj <= BH; lj++) {
                    int global_j = bi * BH + lj;
                    if (global_j > N) {
                        #pragma omp simd
                        for (int li = 1; li <= BW; li++) {
                            out[soa_offset(s, bi, bj, lj, li)] = 0.0;
                        }
                    } else {
                        #pragma omp simd
                        for (int li = 1; li <= BW; li++) {
                            int global_i = bj * BW + li;
                            if (global_i > N) {
                                out[soa_offset(s, bi, bj, lj, li)] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

double jacobi_run(Jacobi2DSoA *s, int tsteps) {
    double start = omp_get_wtime();
    
    double *curr = s->A;
    double *next = s->B;
    
    for (int t = 0; t < tsteps; t++) {
        sync_halos(s, curr);
        jacobi_iteration(curr, next, s);
        
        double *tmp = curr;
        curr = next;
        next = tmp;
    }
    
    if (curr != s->A) {
        double *tmp = s->A;
        s->A = s->B;
        s->B = tmp;
    }
    
    double end = omp_get_wtime();
    return end - start;
}

double jacobi_checksum(Jacobi2DSoA *s) {
    int N = s->N;
    int BM = s->BM;
    int BN = s->BN;
    int BH = s->BH;
    int BW = s->BW;
    double sum = 0.0;
    
    #pragma omp parallel for collapse(2) reduction(+:sum) schedule(static)
    for (int bi = 0; bi < BM; bi++) {
        for (int bj = 0; bj < BN; bj++) {
            for (int lj = 1; lj <= BH; lj++) {
                int global_j = bi * BH + lj;
                if (global_j > N) continue;
                
                for (int li = 1; li <= BW; li++) {
                    int global_i = bj * BW + li;
                    if (global_i > N) continue;
                    
                    sum += s->A[soa_offset(s, bi, bj, lj, li)];
                }
            }
        }
    }
    
    return sum;
}

int main(int argc, char **argv) {
    int N = 2048;
    int tsteps = 100;
    int num_threads = omp_get_max_threads();
    int BM = 4;
    int BN = 4;
    
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) tsteps = atoi(argv[2]);
    if (argc > 4) BM = atoi(argv[4]);
    if (argc > 5) BN = atoi(argv[5]);

    printf("Jacobi2D OpenMP with SoA Block-Contiguous Storage (64-byte aligned)\n");
    printf("N=%d, tsteps=%d, threads=%d, blocks=%dx%d\n", N, tsteps, num_threads, BM, BN);

    Jacobi2DSoA solver;
    jacobi_init(&solver, N, BM, BN);
    jacobi_initialize_data(&solver);


    #ifdef PAPI_ENABLED
    const char* papi_metric = getenv("PAPI_METRIC");
    init_papi(papi_metric);
    #pragma omp parallel 
    {
        #pragma omp critical 
        {
            start_papi_thread();
        }
    }
    #endif

    double time = jacobi_run(&solver, tsteps);
    double checksum = jacobi_checksum(&solver);

    #ifdef PAPI_ENABLED
    #pragma omp parallel 
    {
        #pragma omp critical 
        {
            stop_papi_thread();
        }
    }
    // Print PAPI results
    long long total_count = 0;
    printf("\nPAPI Results (%s):\n", papi_metric);
    for (int i = 0; i < num_threads; i++) {
        printf("  Thread %d: %lld\n", i, g_papi_values[i]);
        total_count += g_papi_values[i];
    }
    printf("  Total: %lld\n", total_count);
    printf("  Per iteration: %.2f\n", (double)total_count / tsteps);
    #endif

    printf("Time: %.6f seconds\n", time);
    printf("Checksum: %.10e\n", checksum);
    printf("GFLOPS: %.3f\n", (double)N * N * tsteps * 6 / time / 1e9);
    
    jacobi_free(&solver);
    
    return 0;
}