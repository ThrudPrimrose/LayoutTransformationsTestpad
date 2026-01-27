#ifndef JACOBI2D_TEMPLATED_H
#define JACOBI2D_TEMPLATED_H

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Templated Jacobi2D with compile-time tile sizes
template<int BM, int BN>
class Jacobi2DTiled{
public:
    int N;        // Interior size
    double *A;
    double *B;
    int num_tiles_i;
    int num_tiles_j;
    
    Jacobi2DTiled(int size) : N(size) {
        A = new double[(N + 2) * (N + 2)];
        B = new double[(N + 2) * (N + 2)];
        num_tiles_i = (N + BM - 1) / BM;
        num_tiles_j = (N + BN - 1) / BN;
    }
    
    ~Jacobi2DTiled() {
        delete[] A;
        delete[] B;
    }
    
    void initialize() {
        #pragma omp parallel for
        for (int i = 0; i < (N + 2); i++) {
            for (int j = 0; j < (N + 2); j++) {
                A[i * (N + 2) + j] = (double)(i * (j + 2)) / N;
                B[i * (N + 2) + j] = (double)(i * (j + 3)) / N;
            }
        }
    }
    
    inline void process_tile(double *in, double *out, int i_start, int j_start) {
        int i_end = std::min(i_start + BM - 1, N);
        int j_end = std::min(j_start + BN - 1, N);
        
        #pragma omp unroll
        for (int i = i_start; i <= i_end; i++) {
            #pragma omp simd
            for (int j = j_start; j <= j_end; j++) {
                out[i * (N + 2) + j] = 0.2 * (in[i * (N + 2) + j] + 
                                               in[i * (N + 2) + (j - 1)] + 
                                               in[i * (N + 2) + (j + 1)] + 
                                               in[(i - 1) * (N + 2) + j] + 
                                               in[(i + 1) * (N + 2) + j]);
            }
        }
    }
    
    void run_iteration(double *in, double *out) {
        #pragma omp parallel for collapse(2)
        for (int ti = 0; ti < num_tiles_i; ti++) {
            for (int tj = 0; tj < num_tiles_j; tj++) {
                int i_start = 1 + ti * BM;
                int j_start = 1 + tj * BN;
                process_tile(in, out, i_start, j_start);
            }
        }
    }
    
    double run(int tsteps) {
        double start = omp_get_wtime();
        
        double *curr = A;
        double *next = B;
        
        for (int t = 0; t < tsteps; t++) {
            run_iteration(curr, next);
            // Swap pointers
            double *tmp = curr;
            curr = next;
            next = tmp;
        }

        
        double end = omp_get_wtime();
        return end - start;
    }
    
    double checksum() {
        double sum = 0.0;
        for (int i = 1; i < (N + 1); i++) {
            for (int j = 1; j < (N + 1); j++) {
                sum += A[i * (N + 2) + j];
            }
        }
        return sum;
    }

};

#endif // JACOBI2D_TEMPLATED_H