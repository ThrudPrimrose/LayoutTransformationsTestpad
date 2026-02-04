#!/bin/bash
#SBATCH --job-name=ijstr
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs results

# -------------------------------
# Load GCC 14.2 from Spack
# -------------------------------
spack load gcc@14.2
spack load python@3.12.9%gcc@14.2
spack load sqlite
spack load openmpi@5.0.6%gcc@14.2

echo "Compiler:"
gcc --version

# -------------------------------
# OpenMP configuration
# -------------------------------
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=false


unset PAPI_METRICS
unset PAPI_ENABLED

python3.12 benchmark.py

