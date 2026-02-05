#!/bin/bash
#SBATCH --job-name=istrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --output=istrid_%j.out
#SBATCH --error=istrid_%j.err

# Load required modules
spack load cuda@12.9 gcc@14.2

# Set CUDA library path
export LD_LIBRARY_PATH=$(spack location -i cuda)/lib64:$LD_LIBRARY_PATH

# Print environment info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA version:"
nvcc --version
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
echo ""

# Set OpenMP threads for CPU parallelism during data conversion
export OMP_NUM_THREADS=288

# Launch 4 independent Python processes, one per GPU
srun --ntasks=1 --gpus-per-task=1 --exact bash -c "export CUDA_VISIBLE_DEVICES=0; python3 benchmark_gpu.py" &

# Wait for all background jobs to complete
wait

# Print completion info
echo ""
echo "Job completed at: $(date)"