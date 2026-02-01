g++ -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict jacobi2d_blocked.cpp -o jacobi2d_blocked
g++ -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict  jacobi2d_baseline.cpp -o jacobi2d_baseline
mpicxx -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict  jacobi2d_mpi.cpp -o jacobi2d_mpi
mpicxx -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict  jacobi2d_boundary.cpp -o jacobi2d_boundary
