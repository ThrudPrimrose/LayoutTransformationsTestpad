#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Baseline Jacobi2D with row-major data and OpenMP parallelization
class Jacobi2DBaseline {
public:
    int N;        // Interior size
    double *A;
    double *B;
    
    Jacobi2DBaseline(int size) : N(size) {
        A = new double[(N + 2) * (N + 2)];
        B = new double[(N + 2) * (N + 2)];
    }
    
    ~Jacobi2DBaseline() {
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
    
    void run_iteration(double *in, double *out) {
        // Compute interior points [1..N] (0 and N+1 are boundaries)
        #pragma omp parallel for
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                out[i * (N + 2) + j] = 0.2 * (in[i * (N + 2) + j] + 
                                               in[i * (N + 2) + (j - 1)] + 
                                               in[i * (N + 2) + (j + 1)] + 
                                               in[(i - 1) * (N + 2) + j] + 
                                               in[(i + 1) * (N + 2) + j]);
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
    
    printf("Jacobi2D Baseline\n");
    printf("N=%d, tsteps=%d, threads=%d\n", N, tsteps, num_threads);
    
    Jacobi2DBaseline solver(N);
    solver.initialize();
    
    double time = solver.run(tsteps);
    double checksum = solver.checksum();
    
    printf("Time: %.6f seconds\n", time);
    printf("Checksum: %.10e\n", checksum);
    printf("GFLOPS: %.3f\n", (double)N * N * tsteps * 6 / time / 1e9);
    
    
    return 0;
}