#!/usr/bin/env python3
"""
PAPI benchmark script for Jacobi2D implementations.
Runs all variants with different configurations and saves PAPI results to CSV.
"""

import subprocess
import csv
import time
import os
from pathlib import Path
import statistics

# Configuration
N = 2048
TSTEPS = 100
NUM_THREADS = 16
NUM_RUNS = 5  # Reduced for PAPI (can be more expensive)
OUTPUT_CSV = "jacobi2d_papi_results.csv"

# PAPI metrics to measure
#PAPI_METRICS = ["PAPI_L1_DCM", "PAPI_L1_TCM", "PAPI_L2_DCM", "PAPI_L2_TCM", "PAPI_L3_TCM", "PAPI_L3_DCM"]
PAPI_METRICS = ["PAPI_L3_DCM"]

# Benchmark configurations
BENCHMARKS = [
    ("jacobi2d_baseline.out", "Baseline", []),
    ("jacobi2d_boundary.out", "Boundary", []),
]

# Add tiled configurations (0-15)
for i in range(16):
    BENCHMARKS.append(("jacobi2d_tiled.out", f"Tiled-Config{i}", [str(i)]))

# Add blocked configurations (0-15)
for i in range(16):
    BENCHMARKS.append(("jacobi2d_blocked_tiled.out", f"Blocked-Config{i}", [str(i)]))

def run_command(cmd, env=None):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, 
                               check=True, env=env)
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"    Timeout: {' '.join(cmd)}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"    Command failed (exit {e.returncode}): {' '.join(cmd)}")
        if e.stderr:
            print(f"    Stderr: {e.stderr[:200]}...")
        return None

def parse_time(output):
    """Extract time from output."""
    if not output:
        return None
    for line in output.split('\n'):
        if 'Time:' in line:
            try:
                return float(line.split(':')[1].split()[0])
            except:
                return None
    return None

def parse_checksum(output):
    """Extract checksum from output."""
    if not output:
        return None
    for line in output.split('\n'):
        if 'Checksum:' in line:
            try:
                return float(line.split(':')[1].strip())
            except:
                return None
    return None

def parse_papi_results(output, metric):
    """Extract PAPI results from output."""
    if not output:
        return None
    
    results = {
        'thread_values': [],
        'total': None,
        'per_iteration': None
    }
    
    lines = output.split('\n')
    metric_found = False
    
    for i, line in enumerate(lines):
        # Find the metric header
        if f'PAPI Results ({metric}):' in line:
            metric_found = True
            continue
        
        # Parse thread values
        if metric_found and line.strip().startswith('Thread'):
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    value = int(parts[1].strip())
                    results['thread_values'].append(value)
            except:
                continue
        
        # Parse total
        if metric_found and 'Total:' in line:
            try:
                results['total'] = int(line.split(':')[1].strip())
            except:
                pass
        
        # Parse per iteration
        if metric_found and 'Per iteration:' in line:
            try:
                results['per_iteration'] = float(line.split(':')[1].strip())
            except:
                pass
        
        # Stop parsing if we hit the next section
        if metric_found and ('Time:' in line or 'Checksum:' in line):
            break
    
    # If we found any thread values, return the results
    if results['thread_values']:
        return results
    return None

def warmup(executable, args, env):
    """Run warmup iteration (without PAPI)."""
    # Create environment without PAPI for warmup
    warmup_env = env.copy()
    if 'PAPI_METRIC' in warmup_env:
        del warmup_env['PAPI_METRIC']
    
    cmd = [f"./{executable}", str(N), str(TSTEPS), str(NUM_THREADS)] + args
    print(f"  Warmup: {' '.join(cmd[:3])} {' '.join(args)}")
    run_command(cmd, env=warmup_env)

def run_papi_benchmark(executable, config_name, args, metric):
    """Run benchmark with specific PAPI metric and return results."""
    # Set up environment with PAPI
    env = os.environ.copy()
    env['PAPI_ENABLED'] = 'TRUE'
    env['PAPI_METRIC'] = metric
    
    cmd = [f"./{executable}", str(N), str(TSTEPS), str(NUM_THREADS)] + args
    output = run_command(cmd, env=env)
    
    if not output:
        return None, None, None
    
    time_val = parse_time(output)
    checksum_val = parse_checksum(output)
    papi_results = parse_papi_results(output, metric)
    
    return time_val, checksum_val, papi_results

def write_papi_result(writer, executable, config_name, metric, run_data):
    """Write PAPI results to CSV."""
    if not run_data or len(run_data['times']) == 0:
        return
    
    # Calculate statistics
    avg_time = statistics.mean(run_data['times']) if run_data['times'] else 0
    median_time = statistics.median(run_data['times']) if run_data['times'] else 0
    
    # Collect all thread values across runs
    all_thread_values = []
    all_totals = []
    all_per_iterations = []
    
    for papi_results in run_data['papi_results']:
        if papi_results:
            all_thread_values.extend(papi_results['thread_values'])
            if papi_results['total'] is not None:
                all_totals.append(papi_results['total'])
            if papi_results['per_iteration'] is not None:
                all_per_iterations.append(papi_results['per_iteration'])
    
    # Calculate PAPI statistics
    thread_mean = statistics.mean(all_thread_values) if all_thread_values else 0
    thread_median = statistics.median(all_thread_values) if all_thread_values else 0
    thread_std = statistics.stdev(all_thread_values) if len(all_thread_values) > 1 else 0
    
    total_mean = statistics.mean(all_totals) if all_totals else 0
    total_median = statistics.median(all_totals) if all_totals else 0
    total_std = statistics.stdev(all_totals) if len(all_totals) > 1 else 0
    
    per_iter_mean = statistics.mean(all_per_iterations) if all_per_iterations else 0
    per_iter_median = statistics.median(all_per_iterations) if all_per_iterations else 0
    
    # Calculate GFLOPS
    gflops = (N * N * TSTEPS * 6) / (avg_time * 1e9) if avg_time > 0 else 0
    
    # Prepare row
    row = {
        'executable': executable,
        'config': config_name,
        'metric': metric,
        'N': N,
        'tsteps': TSTEPS,
        'threads': NUM_THREADS,
        'num_runs': len(run_data['times']),
        
        # Time statistics
        'avg_time': avg_time,
        'median_time': median_time,
        'min_time': min(run_data['times']) if run_data['times'] else 0,
        'max_time': max(run_data['times']) if run_data['times'] else 0,
        
        # Performance
        'gflops': gflops,
        'checksum': run_data['checksums'][0] if run_data['checksums'] else 0,
        
        # PAPI total statistics
        'papi_total_mean': total_mean,
        'papi_total_median': total_median,
        'papi_total_std': total_std,
        'papi_total_min': min(all_totals) if all_totals else 0,
        'papi_total_max': max(all_totals) if all_totals else 0,
    }
    
    writer.writerow(row)

def main():
    print("=" * 70)
    print("Jacobi2D PAPI Metrics")
    print("=" * 70)
    print(f"N={N}, tsteps={TSTEPS}, threads={NUM_THREADS}, runs={NUM_RUNS}")
    print(f"PAPI Metrics: {', '.join(PAPI_METRICS)}")
    print()
    
    # Clean and build with PAPI
    print("Building executables with PAPI support...")
    subprocess.run(['make', 'clean'], capture_output=True)
    
    # Set PAPI_ENABLED for the build
    build_env = os.environ.copy()
    build_env['PAPI_ENABLED'] = 'TRUE'
    
    result = subprocess.run(['make', '-j'], capture_output=True, text=True, env=build_env)
    if result.returncode != 0:
        print("Build failed!")
        print(result.stderr)
        return
    print("Build successful!\n")
    
    # Open CSV file
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    fieldnames = [
        'executable', 'config', 'metric', 'N', 'tsteps', 'threads', 'num_runs',
        'avg_time', 'median_time', 'min_time', 'max_time', 'gflops', 'checksum',
        'papi_total_mean', 'papi_total_median', 'papi_total_std', 'papi_total_min', 'papi_total_max',
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    
    # Run benchmarks for each metric
    total_benchmarks = len(BENCHMARKS) * len(PAPI_METRICS)
    completed = 0
    
    # Warmup environment (no PAPI)
    warmup_env = os.environ.copy()
    if 'PAPI_ENABLED' in warmup_env:
        del warmup_env['PAPI_ENABLED']
    if 'PAPI_METRIC' in warmup_env:
        del warmup_env['PAPI_METRIC']
    
    for metric in PAPI_METRICS:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking with metric: {metric}")
        print(f"{'=' * 70}")
        
        for idx, (executable, config_name, args) in enumerate(BENCHMARKS, 1):
            completed += 1
            print(f"[{completed}/{total_benchmarks}] {config_name} ({metric})...")
            
            # Warmup (without PAPI)
            warmup_cmd = [f"./{executable}", str(N), str(TSTEPS), str(NUM_THREADS)] + args
            print(f"  Warmup: {' '.join(warmup_cmd[:3])} {' '.join(args)}")
            run_command(warmup_cmd, env=warmup_env)
            
            # Run benchmark with PAPI
            run_data = {
                'times': [],
                'checksums': [],
                'papi_results': []
            }
            
            for run in range(NUM_RUNS):
                print(f"    Run {run+1}/{NUM_RUNS}...", end=' ', flush=True)
                time_val, checksum_val, papi_results = run_papi_benchmark(
                    executable, config_name, args, metric
                )
                
                if time_val and checksum_val and papi_results:
                    run_data['times'].append(time_val)
                    run_data['checksums'].append(checksum_val)
                    run_data['papi_results'].append(papi_results)
                    print(f"✓ time={time_val:.3f}s, total={papi_results['total']}")
                else:
                    print("✗ failed")
            
            # Write result for this metric/config
            if run_data['times']:
                avg_time = statistics.mean(run_data['times'])
                gflops = (N * N * TSTEPS * 6) / (avg_time * 1e9)
                
                # Calculate some PAPI stats
                totals = [r['total'] for r in run_data['papi_results'] if r and r['total']]
                total_mean = statistics.mean(totals) if totals else 0
                
                print(f"  ✓ Completed: avg={avg_time:.3f}s, GFLOPS={gflops:.1f}, "
                      f"mean_{metric}={total_mean:.0f}")
                
                write_papi_result(writer, executable, config_name, metric, run_data)
                csv_file.flush()
            else:
                print(f"  ✗ All runs failed for {config_name} with {metric}")
            
            print()
    
    csv_file.close()
    
    print("=" * 70)
    print(f"PAPI benchmarking complete! Results saved to {OUTPUT_CSV}")
    print("=" * 70)
    
    # Print summary
    print("\nSummary of collected PAPI metrics:")
    print(f"  - L2 Data Cache Misses (PAPI_L2_DCM)")
    print(f"  - L2 Total Cache Misses (PAPI_L2_TCM)")
    print(f"  - L3 Total Cache Misses (PAPI_L3_TCM)")
    print(f"  - L3 Data Cache Misses (PAPI_L3_DCM)")
    print(f"\nFor each metric, collected:")
    print(f"  - Per-thread values (mean, median, std)")
    print(f"  - Total across threads (mean, median, std)")
    print(f"  - Per iteration values (mean, median)")
    print(f"  - Performance metrics (time, GFLOPS)")

if __name__ == "__main__":
    main()