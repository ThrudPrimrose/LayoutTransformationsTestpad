#!/usr/bin/env python3
import subprocess
import re
import csv
import sys
from pathlib import Path
from datetime import datetime
import glob
import itertools

if len(sys.argv) > 1:
    rank = int(sys.argv[1])
    total_ranks = int(sys.argv[2]) if len(sys.argv) > 2 else 4
else:
    rank = 0
    total_ranks = 1

print(f"Running as rank {rank} of {total_ranks}")

# Extensive parameter sweep configurations
# Format: (BM, BN, BK, TM, TN)
PARAM_CONFIGS = [
    # Original baseline
    (64, 64, 16, 8, 8),

    # Varying BM (Block M dimension)
    (32, 64, 16, 8, 8),
    (128, 64, 16, 8, 8),
    (256, 64, 16, 8, 8),
    
    # Varying BN (Block N dimension)
    (64, 32, 16, 8, 8),
    (64, 128, 16, 8, 8),
    (64, 256, 16, 8, 8),
    
    # Varying BK (Block K dimension)
    (64, 64, 8, 8, 8),
    (64, 64, 32, 8, 8),
    (64, 64, 64, 8, 8),
    
    # Square block sizes
    (32, 32, 16, 8, 8),
    (32, 32, 32, 8, 8),
    (128, 128, 16, 8, 8),
    (128, 128, 32, 8, 8),
    (256, 256, 16, 8, 8),
    
    # Rectangular blocks - tall
    (128, 64, 16, 8, 8),
    (256, 128, 16, 8, 8),
    (128, 64, 32, 8, 8),
    
    # Rectangular blocks - wide
    (64, 128, 16, 8, 8),
    (128, 256, 16, 8, 8),
    (64, 128, 32, 8, 8),
    
    # Varying TM (Thread tile M)
    (64, 64, 16, 4, 8),
    (64, 64, 16, 16, 8),
    (128, 128, 16, 4, 8),
    (128, 128, 16, 16, 8),
    
    # Varying TN (Thread tile N)
    (64, 64, 16, 8, 4),
    (64, 64, 16, 8, 16),
    (128, 128, 16, 8, 4),
    (128, 128, 16, 8, 16),
    
    # Varying both TM and TN
    (64, 64, 16, 4, 4),
    (64, 64, 16, 16, 16),
    (128, 128, 16, 4, 4),
    (128, 128, 16, 16, 16),
    
    # Larger BK values
    (64, 64, 16, 8, 8),
    (64, 64, 32, 8, 8),
    (128, 128, 8, 8, 8),
    (128, 128, 16, 8, 8),
    (128, 128, 32, 8, 8),
    
    # Mixed configurations optimized for different scenarios
    (64, 128, 32, 8, 8),
    (128, 64, 32, 8, 8),
    (64, 256, 16, 8, 8),
    (256, 64, 16, 8, 8),
    
    # Small thread tiles with large blocks
    (128, 128, 16, 4, 4),
    (256, 256, 16, 4, 4),
    (128, 128, 32, 4, 4),
    
    # Large thread tiles with medium blocks
    (64, 64, 16, 16, 16),
    (64, 64, 32, 16, 16),
    (128, 128, 16, 16, 16),
    
    # Exploring BK = 8
    (32, 32, 8, 4, 4),
    (64, 64, 8, 8, 8),
    (128, 128, 8, 8, 8),
    (64, 128, 8, 8, 8),
    (128, 64, 8, 8, 8),
    
    # Higher BK values
    (32, 32, 64, 8, 8),
    (64, 64, 64, 8, 8),
    (128, 128, 64, 8, 8),
    
    # Extreme configurations - very large blocks
    (256, 256, 32, 8, 8),
    (256, 256, 16, 16, 16),
    (256, 128, 32, 8, 8),
    (128, 256, 32, 8, 8),
    
    # Asymmetric thread tiles
    (64, 64, 16, 4, 16),
    (64, 64, 16, 16, 4),
    (128, 128, 16, 4, 16),
    (128, 128, 16, 16, 4),
    
    # More BK variations
    (64, 64, 24, 8, 8),
    (128, 128, 24, 8, 8),
    (64, 128, 24, 8, 8),
    
    # Small blocks, various configurations
    (32, 32, 16, 4, 4),
    (32, 32, 32, 4, 4),
    (32, 64, 16, 4, 8),
    (64, 32, 16, 8, 4),
    (32, 64, 32, 4, 8),
    (64, 32, 32, 8, 4),
    
    # Medium-large blocks
    (96, 96, 16, 8, 8),
    (96, 96, 32, 8, 8),
    (160, 160, 16, 8, 8),
    (192, 192, 16, 8, 8),
    
    # Non-power-of-2 dimensions
    (80, 80, 16, 8, 8),
    (96, 64, 16, 8, 8),
    (64, 96, 16, 8, 8),
    (80, 96, 16, 8, 8),
    
    # Testing thread tile variations with 128x128 blocks
    (128, 128, 16, 8, 4),
    (128, 128, 16, 4, 8),
    (128, 128, 16, 8, 16),
    (128, 128, 16, 16, 8),
    
    # Testing thread tile variations with 64x64 blocks
    (64, 64, 16, 8, 4),
    (64, 64, 16, 4, 8),
    (64, 64, 32, 8, 4),
    (64, 64, 32, 4, 8),
    
    # Large rectangular blocks
    (256, 64, 16, 16, 8),
    (64, 256, 16, 8, 16),
    (256, 64, 32, 16, 8),
    (64, 256, 32, 8, 16),
    
    # Various BK with square blocks
    (128, 128, 8, 4, 4),
    (128, 128, 16, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 128, 64, 4, 4),
    
    # Testing with TM=TN=2
    (64, 64, 16, 2, 2),
    (128, 128, 16, 2, 2),
    (64, 64, 32, 2, 2),
    
    # More extreme thread tiles
    (64, 64, 16, 32, 32),
    (128, 128, 16, 32, 32),
    
    # Balanced medium configurations
    (96, 96, 16, 4, 4),
    (96, 96, 16, 8, 8),
    (96, 96, 24, 8, 8),
    
    # More non-square combinations
    (48, 96, 16, 4, 8),
    (96, 48, 16, 8, 4),
    (80, 160, 16, 8, 8),
    (160, 80, 16, 8, 8),
]

DIMENSION_CONFIGS = [
    (8192, 8192, 8192),
    (8192, 8192, 256)
]


ALL_CONFIGS = list(itertools.product(PARAM_CONFIGS, DIMENSION_CONFIGS))

# Distribute configurations across ranks
ALL_CONFIGS = [config for i, config in enumerate(ALL_CONFIGS) if i % total_ranks == rank]

def compile_cuda(source_file, output_binary, bm, bn, bk, tm, tn, m, n, k):
    """Compile CUDA source with specified parameters."""
    compile_cmd = [
        'nvcc',
        '-O3',
        '-std=c++17',
        '-arch=sm_90',  # H100
        f'-D_BM={bm}',
        f'-D_BN={bn}',
        f'-D_BK={bk}',
        f'-D_TM={tm}',
        f'-D_TN={tn}',
        f'-D_M={m}',
        f'-D_N={n}',
        f'-D_K={k}',
        '-lcublas',
        '-Xcompiler', '-fopenmp',
        source_file,
        '-o', output_binary
    ]
    
    print(f"Compiling {source_file} with BM={bm}, BN={bn}, BK={bk}, TM={tm}, TN={tn}...")
    
    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print(f"✓ Compilation successful: {output_binary}")
        if result.stderr:
            stderr_lines = result.stderr.strip().split('\n')
            warnings = [line for line in stderr_lines if 'warning' in line.lower()]
            if warnings:
                print(f"  Warnings: {len(warnings)} warning(s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Compilation failed!")
        print(f"  stderr: {e.stderr}")
        return False

def parse_2d_output(output, m, n, k):
    """Parse 2D tiled GEMM output - only iterations, no summary."""
    results = []
    
    # Extract parameters
    param_match = re.search(r'BM=(\d+) BN=(\d+) BK=(\d+) TM=(\d+) TN=(\d+)', output)
    if not param_match:
        return results
    
    bm, bn, bk, tm, tn = map(int, param_match.groups())
    
    # Extract kernel iterations
    kernel_pattern = r'2D Block-Tiled GEMM.*?: ([\d.]+) ms \((\d+)/\d+\)'
    for match in re.finditer(kernel_pattern, output):
        time_ms, iteration = match.groups()
        results.append({
            'type': '2D_kernel',
            'bm': bm, 'bn': bn, 'bk': bk, 'tm': tm, 'tn': tn,
            'm': m, 'n': n, 'k': k,
            'iteration': int(iteration),
            'time_ms': float(time_ms)
        })
    
    # Extract cuBLAS iterations
    cublas_pattern = r'cuBLAS DGEMM.*?: ([\d.]+) ms \((\d+)/\d+\)'
    for match in re.finditer(cublas_pattern, output):
        time_ms, iteration = match.groups()
        results.append({
            'type': '2D_cublas',
            'bm': bm, 'bn': bn, 'bk': bk, 'tm': tm, 'tn': tn,
            'm': m, 'n': n, 'k': k,
            'iteration': int(iteration),
            'time_ms': float(time_ms)
        })
    
    return results

def parse_4d_output(output, m, n, k):
    """Parse 4D tensor GEMM output - only iterations, no summary."""
    results = []
    
    # Extract parameters
    param_match = re.search(r'BM=(\d+) BN=(\d+) BK=(\d+) TM=(\d+) TN=(\d+)', output)
    if not param_match:
        return results
    
    bm, bn, bk, tm, tn = map(int, param_match.groups())
    
    # Extract 4D kernel iterations
    kernel_pattern = r'4D Tensor GEMM.*?: ([\d.]+) ms \((\d+)/\d+\)'
    for match in re.finditer(kernel_pattern, output):
        time_ms, iteration = match.groups()
        results.append({
            'type': '4D_kernel',
            'bm': bm, 'bn': bn, 'bk': bk, 'tm': tm, 'tn': tn,
            'm': m, 'n': n, 'k': k,
            'iteration': int(iteration),
            'time_ms': float(time_ms)
        })
    
    # Extract cuBLAS iterations
    cublas_pattern = r'cuBLAS DGEMM.*?: ([\d.]+) ms \((\d+)/\d+\)'
    for match in re.finditer(cublas_pattern, output):
        time_ms, iteration = match.groups()
        results.append({
            'type': '4D_cublas',
            'bm': bm, 'bn': bn, 'bk': bk, 'tm': tm, 'tn': tn,
            'm': m, 'n': n, 'k': k,
            'iteration': int(iteration),
            'time_ms': float(time_ms)
        })
    
    return results

def run_binary(binary_path):
    """Run compiled binary and capture output."""
    print(f"Running {binary_path}...")
    try:
        result = subprocess.run(
            [f'./{binary_path}'],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        print(f"✓ Execution successful")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ Execution failed!")
        print(f"  Return code: {e.returncode}")
        if e.stderr:
            print(f"  stderr: {e.stderr[:200]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"✗ Execution timed out!")
        return None

def save_to_csv(results, filename, mode='a'):
    """Save results to CSV file."""
    if not results:
        return
    
    # Define explicit field order
    fieldnames = ['bk', 'bm', 'bn', 'iteration', 'time_ms', 'tm', 'tn', 'type', 'm', 'n', 'k']
    
    file_exists = Path(filename).exists()
    
    with open(filename, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        
        if not file_exists or mode == 'w':
            writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    print(f"✓ Saved {len(results)} results to {filename}")

def validate_config(bm, bn, bk, tm, tn):
    """Validate that configuration is likely to compile and run."""
    # Check thread count
    threads_per_block = (bm * bn) // (tm * tn)
    if threads_per_block > 1024 or threads_per_block <= 0:
        print(f"  ⚠ Skipping: Invalid thread count {threads_per_block} (must be 1-1024)")
        return False
    
    # Check shared memory (approximate)
    smem_size = (bm * bk + bk * bn) * 8  # 8 bytes per double
    if smem_size > 48 * 1024:
        print(f"  ⚠ Skipping: Shared memory {smem_size} bytes exceeds 48KB limit")
        return False
    
    # Check divisibility
    if bm % tm != 0 or bn % tn != 0:
        print(f"  ⚠ Skipping: BM must be divisible by TM, BN by TN")
        return False
    
    return True

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_2d = f'results_2d_rank{rank}_{timestamp}.csv'
    csv_4d = f'results_4d_rank{rank}_{timestamp}.csv'

    print("="*80)
    print("CUDA GEMM Benchmark Suite - Extended Parameter Sweep")
    print(f"Total configurations to test: {len(ALL_CONFIGS)}")
    print("="*80)
    
    # Initialize CSV files with headers
    save_to_csv([], csv_2d, mode='w')
    save_to_csv([], csv_4d, mode='w')
    
    successful_configs = 0
    failed_configs = 0
    skipped_configs = 0
    
    for idx, ((bm, bn, bk, tm, tn), (m, n, k)) in enumerate(ALL_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"Configuration {idx}/{len(ALL_CONFIGS)}")
        print(f"Dimensions: M={m}, N={n}, K={k}")
        print(f"Parameters: BM={bm}, BN={bn}, BK={bk}, TM={tm}, TN={tn}")
        print(f"{'='*80}")
        
        # Validate configuration before attempting compilation
        if not validate_config(bm, bn, bk, tm, tn):
            skipped_configs += 1
            continue
        
        config_success = True
        
        # Compile and run 2D version
        binary_2d = f'gemm_2d_m{m}_n{n}_k{k}_bm{bm}_bn{bn}_bk{bk}_tm{tm}_tn{tn}_m{m}_n{n}_k{k}'
        if compile_cuda('gemm_fp64_2d_tiled.cu', binary_2d, bm, bn, bk, tm, tn, m, n, k):
            output_2d = run_binary(binary_2d)
            if output_2d:
                results_2d = parse_2d_output(output_2d, m, n, k)
                if results_2d:
                    save_to_csv(results_2d, csv_2d, mode='a')
                    print(f"✓ Parsed {len(results_2d)} data points from 2D output")
                else:
                    print("✗ Failed to parse 2D output")
                    config_success = False
            else:
                config_success = False
        else:
            config_success = False
        
        # Compile and run 4D version
        binary_4d = f'gemm_4d_m{m}_n{n}_k{k}_bm{bm}_bn{bn}_bk{bk}_tm{tm}_tn{tn}_m{m}_n{n}_k{k}'
        if compile_cuda('gemm_fp64_4d_storage.cu', binary_4d, bm, bn, bk, tm, tn, m, n, k):
            output_4d = run_binary(binary_4d)
            if output_4d:
                results_4d = parse_4d_output(output_4d, m, n, k)
                if results_4d:
                    save_to_csv(results_4d, csv_4d, mode='a')
                    print(f"✓ Parsed {len(results_4d)} data points from 4D output")
                else:
                    print("✗ Failed to parse 4D output")
                    config_success = False
            else:
                config_success = False
        else:
            config_success = False
        
        if config_success:
            successful_configs += 1
        else:
            failed_configs += 1
        
        print(f"\nProgress: {idx}/{len(ALL_CONFIGS)} | Success: {successful_configs} | Failed: {failed_configs} | Skipped: {skipped_configs}")
    
    # Clean up generated binaries
    cleanup_binaries()
    
    print(f"\n{'='*80}")
    print("Benchmark Complete!")
    print(f"{'='*80}")
    print(f"Total configurations: {len(ALL_CONFIGS)}")
    print(f"Successful: {successful_configs}")
    print(f"Failed: {failed_configs}")
    print(f"Skipped: {skipped_configs}")
    print(f"\nResults saved to:")
    print(f"  - {csv_2d}")
    print(f"  - {csv_4d}")
    print(f"{'='*80}")

def cleanup_binaries():
    """Remove all generated binary files."""
    print("\nCleaning up generated binaries...")
    
    # Find all generated binaries
    patterns = ['gemm_2d_*', 'gemm_4d_*']
    removed_count = 0
    
    for pattern in patterns:
        binaries = glob.glob(pattern)
        for binary in binaries:
            try:
                Path(binary).unlink()
                removed_count += 1
            except Exception as e:
                print(f"  Warning: Could not remove {binary}: {e}")
    
    print(f"✓ Removed {removed_count} binary files")

if __name__ == '__main__':
    main()