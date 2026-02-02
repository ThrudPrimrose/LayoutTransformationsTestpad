g++ -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict -DPAPI_ENABLED -I. jacobi2d_blocked.cpp -o jacobi2d_blocked_w_papi -lpapi
g++ -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict -DPAPI_ENABLED -I. jacobi2d_baseline.cpp -o jacobi2d_baseline_w_papi -lpapi
mpicxx -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict -DPAPI_ENABLED -I. jacobi2d_mpi.cpp -o jacobi2d_mpi_w_papi -lpapi
mpicxx -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict -DPAPI_ENABLED -I. jacobi2d_boundary.cpp -o jacobi2d_boundary_w_papi -lpapi
