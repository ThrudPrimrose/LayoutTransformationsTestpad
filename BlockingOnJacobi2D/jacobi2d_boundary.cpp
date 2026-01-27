#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Baseline Jacobi2D with row-major data and OpenMP parallelization
class Jacobi2DBoundary {
public:
    int N;        // Interior size
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

    Jacobi2DBoundary(int size) : N(size) {
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
    
    ~Jacobi2DBoundary() {
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
        #pragma omp parallel for
        for (int i = 1; i < (N+1); i++) {
            for (int j = 1; j < (N+1); j++) {
                A[(i-1) * (N) + (j-1)] = (double)(i * (j + 2)) / N;
                B[(i-1) * (N) + (j-1)] = (double)(i * (j + 3)) / N;
            }
        }
        
        // Initialize boundaries
        for (int j = 1; j <= N; j++) {
            // Note: j is in logical coordinates [1..N]
            A_top[j-1]    = (double)(0 * (j + 2)) / N;
            A_bottom[j-1] = (double)((N + 1) * (j + 2)) / N;
            B_top[j-1] = (double)(0 * (j + 3)) / N;
            B_bottom[j-1] = (double)((N + 1) * (j + 3)) / N;
        }
        
        for (int i = 1; i <= N; i++) {
            // Note: i is in logical coordinates [1..N]
            A_left[i-1]   = (double)(i * (0 + 2)) / N;
            A_right[i-1]  = (double)(i * (N + 1 + 2)) / N;
            B_left[i-1] = (double)(i * (0 + 3)) / N;
            B_right[i-1] = (double)(i * (N + 1 + 3)) / N;
        }
    }

    inline double get_value(double *interior, double *top, double *bottom, 
                        double *left, double *right, int i, int j) {
        // i, j are in logical coordinates [1..N] (inclusive)
        if (i == 0) return top[j-1];        // Convert j from [1..N] to [0..N-1]
        if (i == N+1) return bottom[j-1];   // Same conversion
        if (j == 0) return left[i-1];       // Convert i from [1..N] to [0..N-1]
        if (j == N+1) return right[i-1];    // Same conversion
        
        // Interior: convert to [0..N-1] indexing
        return interior[(i - 1) * N + (j - 1)];
    }

    void run_iteration(double *in, double *out,
                      double *in_top, double *in_bottom,
                      double *in_left, double *in_right) {
        // Compute interior points [1..N] in logical coordinates
        #pragma omp parallel for
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                double center = get_value(in, in_top, in_bottom, in_left, in_right, i, j);
                double left_val = get_value(in, in_top, in_bottom, in_left, in_right, i, j - 1);
                double right_val = get_value(in, in_top, in_bottom, in_left, in_right, i, j + 1);
                double top_val = get_value(in, in_top, in_bottom, in_left, in_right, i - 1, j);
                double bottom_val = get_value(in, in_top, in_bottom, in_left, in_right, i + 1, j);
                
                out[(i - 1) * N + (j - 1)] = 0.2 * (center + left_val + right_val + top_val + bottom_val);
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
        for (int i = 0; i < (N); i++) {
            for (int j = 0; j < (N); j++) {
                sum += A[i * (N) + j];
            }
        }
        return sum;
    }
    

};

int main(int argc, char **argv) {
    int N = 2048;
    int tsteps = 100;
    int num_threads = omp_get_max_threads();
    
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) tsteps = atoi(argv[2]);
    if (argc > 3) {
        num_threads = atoi(argv[3]);
        omp_set_num_threads(num_threads);
    }
    
    printf("Jacobi2D Boundary\n");
    printf("N=%d, tsteps=%d, threads=%d\n", N, tsteps, num_threads);
    
    Jacobi2DBoundary solver(N);
    solver.initialize();
    
    double time = solver.run(tsteps);
    double checksum = solver.checksum();
    
    printf("Time: %.6f seconds\n", time);
    printf("Checksum: %.10e\n", checksum);
    printf("GFLOPS: %.3f\n", (double)N * N * tsteps * 6 / time / 1e9);
    
    return 0;
}