#include "jacobi2d_blocked_tiled_v2.h"

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#ifdef PAPI_ENABLED
#include "papi_metrics.h"
#endif


int main(int argc, char **argv) {
    int N = 2048;
    int tsteps = 100;
    int block_config = 0; // 0: 16x16, 1: 32x32, 2: 64x64, 3: 32x16
    int num_threads = omp_get_max_threads();
    
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) tsteps = atoi(argv[2]);
    if (argc > 3) block_config = atoi(argv[3]);
    if (argc > 4) {
        num_threads = atoi(argv[4]);
        omp_set_num_threads(num_threads);
    }
    
    printf("Jacobi2D Blocked Layout V2\n");
    printf("N=%d, tsteps=%d, threads=%d\n", N, tsteps, num_threads);
    
    double time = 0.0;
    double checksum = 0.0;

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

    switch (block_config) {
        case 0: {
            printf("Block size: 16x16\n");
            Jacobi2DBlockedV2<16, 16> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 1: {
            printf("Block size: 32x32\n");
            Jacobi2DBlockedV2<32, 32> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 2: {
            printf("Block size: 64x64\n");
            Jacobi2DBlockedV2<64, 64> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 3: {
            printf("Block size: 32x16\n");
            Jacobi2DBlockedV2<32, 16> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 4: {
            printf("Block size: 16x32\n");
            Jacobi2DBlockedV2<16, 32> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 5: {
            printf("Block size: 128x128\n");
            Jacobi2DBlockedV2<128, 128> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 6: {
            printf("Block size: 256x256\n");
            Jacobi2DBlockedV2<256, 256> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 7: {
            printf("Block size: 8x8\n");
            Jacobi2DBlockedV2<8, 8> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 8: {
            printf("Block size: 64x32\n");
            Jacobi2DBlockedV2<64, 32> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 9: {
            printf("Block size: 32x64\n");
            Jacobi2DBlockedV2<32, 64> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 10: {
            printf("Block size: 128x64\n");
            Jacobi2DBlockedV2<128, 64> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 11: {
            printf("Block size: 64x128\n");
            Jacobi2DBlockedV2<64, 128> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 12: {
            printf("Block size: 16x8\n");
            Jacobi2DBlockedV2<16, 8> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 13: {
            printf("Block size: 8x16\n");
            Jacobi2DBlockedV2<8, 16> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 14: {
            printf("Block size: 512x512\n");
            Jacobi2DBlockedV2<512, 512> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        case 15: {
            printf("Block size: 4x4\n");
            Jacobi2DBlockedV2<4, 4> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
        default: {
            printf("Invalid block config. Using 32x32\n");
            Jacobi2DBlockedV2<32, 32> solver(N);
            solver.initialize();
            time = solver.run(tsteps);
            checksum = solver.checksum();
            break;
        }
    }

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
    
    return 0;
}