#!/usr/bin/env python3
"""
Benchmark script for Jacobi2D implementations with PAPI metrics.
Runs baseline, blocked (SoA), boundary, and MPI versions with various configurations.
Collects performance counters: L3 cache hits/misses, load-store instructions, total cache misses/accesses.
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
N = 1024
TSTEPS = 200
NUM_RUNS = 1  # Changed from 20 to 10 for PAPI runs
NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", "128"))

# Output CSV file
OUTPUT_FILE = "jacobi2d_papi_benchmark_results.csv"

# Executables (PAPI-enabled versions)
BASELINE_EXE = "./jacobi2d_baseline_w_papi"
BLOCKED_EXE = "./jacobi2d_blocked_w_papi"
BOUNDARY_EXE = "./jacobi2d_boundary_w_papi"
MPI_EXE = "./jacobi2d_mpi_w_papi"

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
    (6, 8), (8, 6)
} if u * v == NUM_CORES
]

# PAPI metrics to collect
PAPI_METRICS = {
    'L2_DCR': 'PAPI_L2_DCR',
    'LST_INS': 'PAPI_LST_INS',
    'L3_TCM': 'PAPI_L3_TCM',
    'L3_TCA': 'PAPI_L3_TCA',
    'L3_DCR': 'PAPI_L3_DCR',
    'L3_DCW': 'PAPI_L3_DCW',
    'L3_DCM': 'PAPI_L3_DCM',
    'L3_DCA': 'PAPI_L3_DCA',
    'L2_TCM': 'PAPI_L2_TCM',
    'L2_TCA': 'PAPI_L2_TCA',
    'L2_DCA': 'PAPI_L2_DCA',
    'L2_DCW': 'PAPI_L2_DCW',
    'L2_DCM': 'PAPI_L2_DCM',
}


def parse_output(output):
    """Parse the output from a Jacobi2D run to extract time, checksum, GFLOPS, and PAPI metrics."""
    time_val = None
    checksum = None
    gflops = None
    papi_total = None
    papi_per_iter = None
    
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
        elif 'Total:' in line:
            match = re.search(r'Total:\s*(\d+)', line)
            if match:
                papi_total = int(match.group(1))
        elif 'Per iteration:' in line:
            match = re.search(r'Per iteration:\s*([\d.]+)', line)
            if match:
                papi_per_iter = float(match.group(1))
    
    return time_val, checksum, gflops, papi_total, papi_per_iter


def run_command(cmd, env_vars, timeout=600):
    """Run a command with specified environment variables and return parsed output."""
    try:
        env = {**os.environ, **env_vars}
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            env=env
        )
        # Combine stdout and stderr in case output goes to stderr
        combined_output = result.stdout + result.stderr
        print(combined_output)
        return parse_output(combined_output)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s")
        return None, None, None, None, None
    except FileNotFoundError as e:
        print(f"    ERROR: {e}")
        return None, None, None, None, None


def run_baseline(n, tsteps, num_threads, papi_metric):
    """Run baseline implementation with PAPI metric."""
    cmd = [BASELINE_EXE, str(n), str(tsteps), str(num_threads)]
    env_vars = {
        'OMP_NUM_THREADS': str(num_threads),
        'OMP_PROC_BIND': 'TRUE',
        'OMP_PLACES': 'CORES',
        'PAPI_METRIC': papi_metric
    }
    return run_command(cmd, env_vars)


def run_boundary(n, tsteps, num_threads, papi_metric):
    """Run boundary implementation with PAPI metric."""
    cmd = [BOUNDARY_EXE, str(n), str(tsteps), str(num_threads)]
    env_vars = {
        'OMP_NUM_THREADS': str(num_threads),
        'OMP_PROC_BIND': 'TRUE',
        'OMP_PLACES': 'CORES',
        'PAPI_METRIC': papi_metric
    }
    return run_command(cmd, env_vars)


def run_blocked(n, tsteps, num_threads, bm, bn, papi_metric):
    """Run blocked (SoA) implementation with PAPI metric."""
    cmd = [BLOCKED_EXE, str(n), str(tsteps), str(num_threads), str(bm), str(bn)]
    env_vars = {
        'OMP_NUM_THREADS': str(num_threads),
        'OMP_PROC_BIND': 'TRUE',
        'OMP_PLACES': 'CORES',
        'PAPI_METRIC': papi_metric
    }
    return run_command(cmd, env_vars)


def run_mpi(n, tsteps, px, py, papi_metric):
    """Run MPI implementation with PAPI metric."""
    nprocs = px * py
    cmd = [
        "mpirun", "--use-hwthread-cpus", "-np", str(nprocs),
        MPI_EXE, str(n), str(px), str(py), str(tsteps)
    ]
    env_vars = {
        'PAPI_METRIC': papi_metric
    }
    return run_command(cmd, env_vars)


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
        'config_param1', 'config_param2', 'papi_metric', 'run_id',
        'time_seconds', 'checksum', 'gflops', 'papi_total', 'papi_per_iter'
    ]
    
    file_exists = os.path.exists(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            csvfile.flush()
        
        print(f"Starting PAPI benchmark: N={N}, tsteps={TSTEPS}, runs={NUM_RUNS}")
        print(f"Results will be written to: {OUTPUT_FILE}")
        print("=" * 70)
        
        # Iterate over each PAPI metric
        for metric_name, metric_code in PAPI_METRICS.items():
            print(f"\n{'='*70}")
            print(f"COLLECTING METRIC: {metric_name} ({metric_code})")
            print(f"{'='*70}")
            
            # Run baseline
            if available['baseline']:
                print(f"\n[Baseline] threads={NUM_THREADS}, metric={metric_name}")
                for run_id in range(NUM_RUNS):
                    time_val, checksum, gflops, papi_total, papi_per_iter = run_baseline(
                        N, TSTEPS, NUM_THREADS, metric_code
                    )
                    
                    row = {
                        'timestamp': datetime.now().isoformat(),
                        'kernel': 'baseline',
                        'N': N,
                        'tsteps': TSTEPS,
                        'num_threads': NUM_THREADS,
                        'config_param1': '-',
                        'config_param2': '-',
                        'papi_metric': metric_name,
                        'run_id': run_id,
                        'time_seconds': time_val,
                        'checksum': checksum,
                        'gflops': gflops,
                        'papi_total': papi_total,
                        'papi_per_iter': papi_per_iter
                    }
                    
                    writer.writerow(row)
                    csvfile.flush()
                    if papi_total is None:
                        papi_total = 0
                    if papi_per_iter is None:
                        papi_per_iter = 0
                    if time_val:
                        print(f"  Run {run_id+1:2d}/{NUM_RUNS}: time={time_val:.4f}s, GFLOPS={gflops:.2f}, "
                              f"{metric_name}={papi_total}, per_iter={papi_per_iter:.2f}")
                    else:
                        print(f"  Run {run_id+1:2d}/{NUM_RUNS}: FAILED")
            # Run boundary
            if available['boundary']:
                print(f"\n[Boundary] threads={NUM_THREADS}, metric={metric_name}")
                for run_id in range(NUM_RUNS):
                    time_val, checksum, gflops, papi_total, papi_per_iter = run_boundary(
                        N, TSTEPS, NUM_THREADS, metric_code
                    )
                    
                    row = {
                        'timestamp': datetime.now().isoformat(),
                        'kernel': 'boundary',
                        'N': N,
                        'tsteps': TSTEPS,
                        'num_threads': NUM_THREADS,
                        'config_param1': '-',
                        'config_param2': '-',
                        'papi_metric': metric_name,
                        'run_id': run_id,
                        'time_seconds': time_val,
                        'checksum': checksum,
                        'gflops': gflops,
                        'papi_total': papi_total,
                        'papi_per_iter': papi_per_iter
                    }
                    
                    writer.writerow(row)
                    csvfile.flush()
                    if papi_total is None:
                        papi_total = 0
                    if papi_per_iter is None:
                        papi_per_iter = 0
                    if time_val:
                        print(f"  Run {run_id+1:2d}/{NUM_RUNS}: time={time_val:.4f}s, GFLOPS={gflops:.2f}, "
                              f"{metric_name}={papi_total}, per_iter={papi_per_iter:.2f}")
                    else:
                        print(f"  Run {run_id+1:2d}/{NUM_RUNS}: FAILED")
            
            # Run blocked (SoA) configurations
            if available['blocked']:
                for bm, bn in BLOCK_CONFIGS:
                    # Skip if block size doesn't divide N evenly
                    if N % bm != 0 or N % bn != 0:
                        continue
                    
                    print(f"\n[Blocked] BM={bm}, BN={bn}, threads={NUM_THREADS}, metric={metric_name}")
                    for run_id in range(NUM_RUNS):
                        time_val, checksum, gflops, papi_total, papi_per_iter = run_blocked(
                            N, TSTEPS, NUM_THREADS, bm, bn, metric_code
                        )
                        
                        row = {
                            'timestamp': datetime.now().isoformat(),
                            'kernel': 'blocked_soa',
                            'N': N,
                            'tsteps': TSTEPS,
                            'num_threads': NUM_THREADS,
                            'config_param1': f'BM={bm}',
                            'config_param2': f'BN={bn}',
                            'papi_metric': metric_name,
                            'run_id': run_id,
                            'time_seconds': time_val,
                            'checksum': checksum,
                            'gflops': gflops,
                            'papi_total': papi_total,
                            'papi_per_iter': papi_per_iter
                        }
                        
                        writer.writerow(row)
                        csvfile.flush()
                        if papi_total is None:
                            papi_total = 0
                        if papi_per_iter is None:
                            papi_per_iter = 0
                        if time_val:
                            print(f"  Run {run_id+1:2d}/{NUM_RUNS}: time={time_val:.4f}s, GFLOPS={gflops:.2f}, "
                                  f"{metric_name}={papi_total}, per_iter={papi_per_iter:.2f}")
                        else:
                            print(f"  Run {run_id+1:2d}/{NUM_RUNS}: FAILED")
            
            # Run MPI configurations
            if available['mpi']:
                for px, py in MPI_CONFIGS:
                    print(f"\n[MPI] px={px}, py={py}, nprocs={px*py}, metric={metric_name}")
                    for run_id in range(NUM_RUNS):
                        time_val, checksum, gflops, papi_total, papi_per_iter = run_mpi(
                            N, TSTEPS, px, py, metric_code
                        )
                        
                        row = {
                            'timestamp': datetime.now().isoformat(),
                            'kernel': 'mpi',
                            'N': N,
                            'tsteps': TSTEPS,
                            'num_threads': px * py,
                            'config_param1': f'px={px}',
                            'config_param2': f'py={py}',
                            'papi_metric': metric_name,
                            'run_id': run_id,
                            'time_seconds': time_val,
                            'checksum': checksum,
                            'gflops': gflops,
                            'papi_total': papi_total,
                            'papi_per_iter': papi_per_iter
                        }

                        writer.writerow(row)
                        csvfile.flush()
                        if papi_total is None:
                            papi_total = 0
                        if papi_per_iter is None:
                            papi_per_iter = 0
                        if time_val:
                            print(f"  Run {run_id+1:2d}/{NUM_RUNS}: time={time_val:.4f}s, GFLOPS={gflops:.2f}, "
                                  f"{metric_name}={papi_total}, per_iter={papi_per_iter:.2f}")
                        else:
                            print(f"  Run {run_id+1:2d}/{NUM_RUNS}: FAILED")
        
        print("\n" + "=" * 70)
        print(f"PAPI Benchmark complete. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()