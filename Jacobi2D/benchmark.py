#!/usr/bin/env python3
"""
Benchmark script for Jacobi2D implementations.
Runs baseline, blocked (SoA), and MPI versions with various configurations.
"""

import subprocess
import csv
import os
import sys
import time
from datetime import datetime
from itertools import product
import re

# Configuration
N = 16384
TSTEPS = 100
NUM_RUNS = 20
NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])

# Output CSV file
OUTPUT_FILE = "jacobi2d_benchmark_results.csv"

# Executables (adjust paths as needed)
BASELINE_EXE = "./jacobi2d_baseline"
BLOCKED_EXE = "./jacobi2d_blocked"
BOUNDARY_EXE = "./jacobi2d_boundary"
MPI_EXE = "./jacobi2d_mpi"

# Configurations
# Block configurations for OpenMP blocked version: (BM, BN)

# Num cores = 128
NUM_CORES = int(os.environ.get("OMP_NUM_THREADS", "128"))

BLOCK_CONFIGS = [(u, v) for (u, v) in {
    (1, 1),
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (2, 4),
    (4, 2),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (32, 32),
    (1, 128),
    (128, 1),
    (64, 2),
    (2, 64),
    (4, 32),
    (32, 4),
    (32, 8),
    (8, 32),
    (64, 64),
    (128, 128)
} if u * v >= NUM_CORES
]

# MPI processor grid configurations: (px, py) where px * py = total procs
MPI_CONFIGS = [(u, v) for (u, v) in {
    (1, 1),
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (2, 4),
    (4, 2),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (32, 32),
    (1, 128),
    (128, 1),
    (64, 2),
    (2, 64),
    (4, 32),
    (32, 4),
    (6, 6),
    (4, 9), (9, 4),
    (3, 12), (12, 3),
    (2, 18), (18, 2),
    (1, 36), (36, 1),
    (4, 9), (9, 4)
    (3, 12), (12, 3)
    (6, 8), (8, 6)
} if u * v == NUM_CORES
]

def parse_output(output):
    """Parse the output from a Jacobi2D run to extract time, checksum, and GFLOPS."""
    time_val = None
    checksum = None
    gflops = None
    
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('Time:'):
            match = re.search(r'Time:\s*([\d.]+)', line)
            if match:
                time_val = float(match.group(1))
        elif line.startswith('Checksum:'):
            match = re.search(r'Checksum:\s*(\S+)', line)
            if match:
                checksum = match.group(1)
        elif line.startswith('GFLOPS:'):
            match = re.search(r'GFLOPS:\s*([\d.]+)', line)
            if match:
                gflops = float(match.group(1))
    
    return time_val, checksum, gflops


def run_command(cmd, timeout=600):
    """Run a command and return parsed output."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            env={**os.environ, 'OMP_NUM_THREADS': str(NUM_THREADS), 'OMP_PROC_BIND': 'TRUE', 'OMP_PLACES': 'CORES'}
        )
        # Combine stdout and stderr in case output goes to stderr
        combined_output = result.stdout + result.stderr
        return parse_output(combined_output)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s")
        return None, None, None
    except FileNotFoundError as e:
        print(f"    ERROR: {e}")
        return None, None, None


def run_baseline(n, tsteps, num_threads):
    """Run baseline implementation."""
    cmd = [BASELINE_EXE, str(n), str(tsteps), str(num_threads)]
    return run_command(cmd)


def run_boundary(n, tsteps, num_threads):
    """Run boundary implementation."""
    cmd = [BOUNDARY_EXE, str(n), str(tsteps), str(num_threads)]
    return run_command(cmd)


def run_blocked(n, tsteps, num_threads, bm, bn):
    """Run blocked (SoA) implementation."""
    cmd = [BLOCKED_EXE, str(n), str(tsteps), str(num_threads), str(bm), str(bn)]
    return run_command(cmd)


def run_mpi(n, tsteps, px, py):
    """Run MPI implementation."""
    nprocs = px * py
    cmd = [
        "mpirun", "--use-hwthread-cpus", "-np", str(nprocs),
        MPI_EXE, str(n), str(px), str(py), str(tsteps)
    ]
    return run_command(cmd)


def main():
    # Check which executables exist
    available = {}
    for name, exe in [('baseline', BASELINE_EXE), ('blocked', BLOCKED_EXE), 
                       ('mpi', MPI_EXE), ('boundary', BOUNDARY_EXE)]:
        available[name] = os.path.exists(exe)
        status = "FOUND" if available[name] else "NOT FOUND"
        print(f"  {name}: {exe} - {status}")
    
    print()
    
    # Open CSV file and write header
    fieldnames = [
        'timestamp', 'kernel', 'N', 'tsteps', 'num_threads',
        'config_param1', 'config_param2', 'run_id',
        'time_seconds', 'checksum', 'gflops'
    ]
    
    file_exists = os.path.exists(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            csvfile.flush()
        
        print(f"Starting benchmark: N={N}, tsteps={TSTEPS}, runs={NUM_RUNS}")
        print(f"Results will be written to: {OUTPUT_FILE}")
        print("=" * 70)
        
        # Run baseline
        if available['baseline']:
            print(f"\n[Baseline] threads={NUM_THREADS}")
            for run_id in range(NUM_RUNS):
                time_val, checksum, gflops = run_baseline(N, TSTEPS, NUM_THREADS)
                
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'kernel': 'baseline',
                    'N': N,
                    'tsteps': TSTEPS,
                    'num_threads': NUM_THREADS,
                    'config_param1': '-',
                    'config_param2': '-',
                    'run_id': run_id,
                    'time_seconds': time_val,
                    'checksum': checksum,
                    'gflops': gflops
                }
                
                writer.writerow(row)
                csvfile.flush()
                
                if time_val:
                    print(f"  Run {run_id+1:2d}/{NUM_RUNS}: time={time_val:.4f}s, GFLOPS={gflops:.2f}, checksum={checksum}")
                else:
                    print(f"  Run {run_id+1:2d}/{NUM_RUNS}: FAILED")
        
        # Run boundary
        if available['boundary']:
            print(f"\n[Boundary] threads={NUM_THREADS}")
            for run_id in range(NUM_RUNS):
                time_val, checksum, gflops = run_boundary(N, TSTEPS, NUM_THREADS)
                
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'kernel': 'boundary',
                    'N': N,
                    'tsteps': TSTEPS,
                    'num_threads': NUM_THREADS,
                    'config_param1': '-',
                    'config_param2': '-',
                    'run_id': run_id,
                    'time_seconds': time_val,
                    'checksum': checksum,
                    'gflops': gflops
                }
                
                writer.writerow(row)
                csvfile.flush()
                
                if time_val:
                    print(f"  Run {run_id+1:2d}/{NUM_RUNS}: time={time_val:.4f}s, GFLOPS={gflops:.2f}, checksum={checksum}")
                else:
                    print(f"  Run {run_id+1:2d}/{NUM_RUNS}: FAILED")
        
        # Run blocked (SoA) configurations

        if available['blocked']:
            for bm, bn in BLOCK_CONFIGS:
                # Skip if block size doesn't divide N evenly
                if N % bm != 0 or N % bn != 0:
                    print(f"\n[Blocked] BM={bm}, BN={bn} - SKIPPED (N={N} not divisible)")
                    continue
                
                print(f"\n[Blocked] BM={bm}, BN={bn}, threads={NUM_THREADS}")
                for run_id in range(NUM_RUNS):
                    time_val, checksum, gflops = run_blocked(N, TSTEPS, NUM_THREADS, bm, bn)
                    
                    row = {
                        'timestamp': datetime.now().isoformat(),
                        'kernel': 'blocked_soa',
                        'N': N,
                        'tsteps': TSTEPS,
                        'num_threads': NUM_THREADS,
                        'config_param1': f'BM={bm}',
                        'config_param2': f'BN={bn}',
                        'run_id': run_id,
                        'time_seconds': time_val,
                        'checksum': checksum,
                        'gflops': gflops
                    }
                    
                    writer.writerow(row)
                    csvfile.flush()
                    
                    if time_val:
                        print(f"  Run {run_id+1:2d}/{NUM_RUNS}: time={time_val:.4f}s, GFLOPS={gflops:.2f}, checksum={checksum}")
                    else:
                        print(f"  Run {run_id+1:2d}/{NUM_RUNS}: FAILED")
        
        # Run MPI configurations
        if available['mpi']:
            for px, py in MPI_CONFIGS:
                print(f"\n[MPI] px={px}, py={py}, nprocs={px*py}")
                for run_id in range(NUM_RUNS):
                    time_val, checksum, gflops = run_mpi(N, TSTEPS, px, py)
                    
                    row = {
                        'timestamp': datetime.now().isoformat(),
                        'kernel': 'mpi',
                        'N': N,
                        'tsteps': TSTEPS,
                        'num_threads': px * py,
                        'config_param1': f'px={px}',
                        'config_param2': f'py={py}',
                        'run_id': run_id,
                        'time_seconds': time_val,
                        'checksum': checksum,
                        'gflops': gflops
                    }

                    writer.writerow(row)
                    csvfile.flush()

                    if time_val:
                        print(f"  Run {run_id+1:2d}/{NUM_RUNS}: time={time_val:.4f}s, GFLOPS={gflops:.2f}, checksum={checksum}")
                    else:
                        print(f"  Run {run_id+1:2d}/{NUM_RUNS}: FAILED")
        
        print("\n" + "=" * 70)
        print(f"Benchmark complete. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()