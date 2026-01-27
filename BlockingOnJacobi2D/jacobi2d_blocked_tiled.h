#ifndef JACOBI2D_BLOCKED_H
#define JACOBI2D_BLOCKED_H

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Jacobi2D with 4D blocked layout: shape (N/BM, N/BN, BM, BN)
// Stored as single 1D array in row-major order
template<int BM, int BN>
class Jacobi2DBlocked {
public:
    int N;        // Interior size (assumed multiple of BM and BN)
    int num_blocks_i;  // N / BM
    int num_blocks_j;  // N / BN
    
    // 4 separate 1D arrays with 4D blocked layout
    // Layout: [block_i][block_j][local_i][local_j]
    // Flattened index: ((block_i * num_blocks_j + block_j) * BM + local_i) * BN + local_j
    double *A;
    double *B;


    // Boundary arrays (contiguous storage) - boundaries are at index 0 and N+1
    double *A_top;      // Row 0
    double *A_bottom;   // Row N+1
    double *A_left;     // Column 0
    double *A_right;    // Column N+1
    double *B_top;
    double *B_bottom;
    double *B_left;
    double *B_right;

    Jacobi2DBlocked(int size) : N(size)  {
        // Verify N is multiple of BM and BN
        if (N % BM != 0 || N % BN != 0) {
            printf("Error: N=%d must be multiple of BM=%d and BN=%d\n", N, BM, BN);
            exit(1);
        }
        
        num_blocks_i = N / BM;
        num_blocks_j = N / BN;
        
        // Allocate 4D blocked arrays: (num_blocks_i, num_blocks_j, BM, BN)
        A = new double[N * N];
        B = new double[N * N];

        // Allocate boundary arrays
        A_top = new double[(N)];
        A_bottom = new double[(N)];
        A_left = new double[(N)];
        A_right = new double[(N)];
        B_top = new double[(N)];
        B_bottom = new double[(N)];
        B_left = new double[(N)];
        B_right = new double[(N)];

    }
    
    ~Jacobi2DBlocked() {
        delete[] A;
        delete[] B;
        delete[] A_top;
        delete[] A_bottom;
        delete[] A_left;
        delete[] A_right;
        delete[] B_top;
        delete[] B_bottom;
        delete[] B_left;
        delete[] B_right;
    }

    void initialize() {
        // Initialize interior in blocked layout
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                at(A, i, j) = (double)(i * (j + 2)) / N;
                at(B, i, j) = (double)(i * (j + 3)) / N;
            }
        }
        
        // Initialize boundaries
        for (int j = 1; j <= N; j++) {
            A_top[j-1]    = (double)(0 * (j + 2)) / N;
            A_bottom[j-1] = (double)((N + 1) * (j + 2)) / N;
            B_top[j-1] = (double)(0 * (j + 3)) / N;
            B_bottom[j-1] = (double)((N + 1) * (j + 3)) / N;
        }
        
        for (int i = 1; i <= N; i++) {
            A_left[i-1]   = (double)(i * (0 + 2)) / N;
            A_right[i-1]  = (double)(i * (N + 1 + 2)) / N;
            B_left[i-1] = (double)(i * (0 + 3)) / N;
            B_right[i-1] = (double)(i * (N + 1 + 3)) / N;
        }
    }

    // Convert 2D coordinates (i, j) in [1..N] to 4D blocked index
    inline int to_blocked_index(int i, int j) const {
        // Adjust to 0-based interior coordinates
        i -= 1;  // Now i in [0..N-1]
        j -= 1;  // Now j in [0..N-1]
        
        const int block_i = i / BM;
        const int block_j = j / BN;
        const int local_i = i % BM;
        const int local_j = j % BN;
        
        // Flattened 4D index: [block_i][block_j][local_i][local_j]
        return (block_i * (num_blocks_j * BM * BN)) + (block_j * (BM * BN)) + (local_i * BN) + local_j;
    }
    
    // Get reference to element at (i, j) where i, j in [1..N]
    inline double& at(double *arr, int i, int j) const {
        return arr[to_blocked_index(i, j)];
    }

    inline double get_value(double *interior, double *top, double *bottom, 
                        double *left, double *right, int i, int j) {
        if (i == 0) return top[j-1];
        if (i == N + 1) return bottom[j-1];
        if (j == 0) return left[i-1];
        if (j == N + 1) return right[i-1];
        
        // Interior: i,j in [1..N]
        return at(interior, i, j);
    }
    // Process a single block with boundary checks
    inline void process_block(double *in, double *out, double *in_top, double *in_bottom,
                      double *in_left, double *in_right, int block_i, int block_j) {
        int i_start = 1 + block_i * BM;
        int i_end = i_start + BM;
        int j_start = 1 + block_j * BN;
        int j_end = j_start + BN;
        
        #pragma omp unroll
        for (int i = i_start; i < i_end; i++) {
            #pragma omp simd
            for (int j = j_start; j < j_end; j++) {
                double center = get_value(in, in_top, in_bottom, in_left, in_right, i, j);
                double left_val = get_value(in, in_top, in_bottom, in_left, in_right, i, j - 1);
                double right_val = get_value(in, in_top, in_bottom, in_left, in_right, i, j + 1);
                double top_val = get_value(in, in_top, in_bottom, in_left, in_right, i - 1, j);
                double bottom_val = get_value(in, in_top, in_bottom, in_left, in_right, i + 1, j);

                at(out, i, j) = 0.2 * (center + left_val + right_val + top_val + bottom_val);
            }
        }
    }
    
    // Optimized version for interior blocks (no boundary checks)
    inline void process_interior_block(double *in, double *out, int block_i, int block_j) {
        int i_start = 1 + block_i * BM;
        int j_start = 1 + block_j * BN;
        
        #pragma omp unroll
        for (int li = 0; li < BM; li++) {
            int i = i_start + li;
            #pragma omp simd
            for (int lj = 0; lj < BN; lj++) {
                int j = j_start + lj;
                at(out, i, j) = 0.2 * (at(in, i, j) + 
                                       at(in, i, j - 1) + 
                                       at(in, i, j + 1) + 
                                       at(in, i - 1, j) + 
                                       at(in, i + 1, j));
            }
        }
    }
    
    void run_iteration(double *in, double *out,
                      double *in_top, double *in_bottom,
                      double *in_left, double *in_right) {
        #pragma omp parallel for collapse(2)
        for (int bi = 0; bi < num_blocks_i; bi++) {
            for (int bj = 0; bj < num_blocks_j; bj++) {
                // Check if this is a boundary block
                bool is_boundary = (bi == 0) || (bi == num_blocks_i - 1) || 
                                  (bj == 0) || (bj == num_blocks_j - 1);
                
                if (is_boundary) {
                    process_block(in, out, in_top, in_bottom, in_left, in_right, bi, bj);
                } else {
                    process_interior_block(in, out, bi, bj);
                }
            }
        }
    }
    

    
    
    double run(int tsteps) {
        double start = omp_get_wtime();
            
        double *curr = A;
        double *next = B;

        for (int t = 0; t < tsteps; t++) {
            // Use curr's boundaries (but they should be identical anyway)
            if (curr == A) {
                run_iteration(curr, next, A_top, A_bottom, A_left, A_right);
            } else {
                run_iteration(curr, next, B_top, B_bottom, B_left, B_right);
            }
            
            // Swap ONLY interior
            double *tmp = curr;
            curr = next;
            next = tmp;
        }


        double end = omp_get_wtime();
        return end - start;
    }
    
    double checksum() {
        double sum = 0.0;
        
        // Sum all interior points
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                sum += A[i * N + j];
            }
        }

        return sum;
    }
};

#endif // JACOBI2D_BLOCKED_H