g++ -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict jacobi2d_blocked.cpp -o blocked
g++ -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict  jacobi2d_baseline.cpp -o baseline
mpicxx -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict  jacobi2d_mpi.cpp -o dist
mpicxx -fopenmp -O3 -march=native -mtune=native -ffast-math -Wrestrict  jacobi2d_boundary.cpp -o boundary

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# N tsteps
./baseline 2048 100
# N px py tsteps tsteps
mpirun -n 8 --use-hwthread-cpus ./dist 2048 1 8 100
# N tsteps BX BY
./blocked 2048 100 8 8
# N tsteps BX BY
./boundary 2048 100

rm baseline dist blocked boundary