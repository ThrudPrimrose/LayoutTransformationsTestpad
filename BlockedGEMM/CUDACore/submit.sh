#!/bin/bash
#SBATCH --job-name=gemm_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --output=gemm_benchmark_%j.out
#SBATCH --error=gemm_benchmark_%j.err

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
export OMP_NUM_THREADS=72

# Launch 4 independent Python processes, one per GPU
srun --ntasks=1 --gpus-per-task=1 --exact bash -c "export CUDA_VISIBLE_DEVICES=0; python3 sweep_gemm.py 0 4" &
srun --ntasks=1 --gpus-per-task=1 --exact bash -c "export CUDA_VISIBLE_DEVICES=1; python3 sweep_gemm.py 1 4" &
srun --ntasks=1 --gpus-per-task=1 --exact bash -c "export CUDA_VISIBLE_DEVICES=2; python3 sweep_gemm.py 2 4" &
srun --ntasks=1 --gpus-per-task=1 --exact bash -c "export CUDA_VISIBLE_DEVICES=3; python3 sweep_gemm.py 3 4" &

# Wait for all background jobs to complete
wait

# Print completion info
echo ""
echo "Job completed at: $(date)"