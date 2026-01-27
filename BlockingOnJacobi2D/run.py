#!/usr/bin/env python3
"""
Simple benchmark script for Jacobi2D implementations.
Runs all variants with different configurations and saves results to CSV.
"""

import subprocess
import csv
import time
from pathlib import Path

# Configuration
N = 2048
TSTEPS = 100
NUM_THREADS = 16
NUM_RUNS = 10
OUTPUT_CSV = "jacobi2d_benchmark_results.csv"

# Benchmark configurations
BENCHMARKS = [
    ("jacobi2d_baseline.out", "Baseline", []),
    ("jacobi2d_boundary.out", "Boundary", []),
]

# Add tiled configurations (0-15)
for i in range(16):
    BENCHMARKS.append(("jacobi2d_tiled.out", f"Tiled-Config-{i}", [str(i)]))

# Add blocked configurations (0-15)
for i in range(16):
    BENCHMARKS.append(("jacobi2d_blocked_tiled.out", f"Blocked-Config-{i}", [str(i)]))

for i in range(16):
    BENCHMARKS.append(("jacobi2d_blocked_tiled_v2.out", f"Blocked-ConfigV2-{i}", [str(i)]))
    
def run_command(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError:
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

def warmup(executable, args):
    """Run warmup iteration."""
    cmd = [f"./{executable}", str(N), str(TSTEPS), str(NUM_THREADS)] + args
    print(f"  Warmup: {' '.join(cmd)}")
    run_command(cmd)

def run_benchmark(executable, config_name, args):
    """Run benchmark and return results."""
    times = []
    checksum = None
    
    # Warmup
    warmup(executable, args)
    
    # Actual runs
    for run in range(NUM_RUNS):
        cmd = [f"./{executable}", str(N), str(TSTEPS), str(NUM_THREADS)] + args
        output = run_command(cmd)
        
        if output:
            t = parse_time(output)
            if t is not None:
                times.append(t)
                if checksum is None:
                    checksum = parse_checksum(output)
        
        print(f"    Run {run+1}/{NUM_RUNS}: {times[-1]:.6f}s" if times else f"    Run {run+1}/{NUM_RUNS}: FAILED")
    
    return times, checksum

def write_result(writer, executable, config_name, times, checksum):
    """Write a single result to CSV."""
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        gflops = (N * N * TSTEPS * 6) / (avg_time * 1e9)
        
        row = {
            'executable': executable,
            'config': config_name,
            'N': N,
            'tsteps': TSTEPS,
            'threads': NUM_THREADS,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'gflops': gflops,
            'checksum': checksum if checksum else 0.0,
            'num_runs': len(times),
            'all_times': ','.join(f'{t:.6f}' for t in times)
        }
        writer.writerow(row)

def main():
    print("=" * 70)
    print("Jacobi2D Benchmark Suite")
    print("=" * 70)
    print(f"N={N}, tsteps={TSTEPS}, threads={NUM_THREADS}, runs={NUM_RUNS}")
    print()
    
    # Clean and build
    print("Building executables...")
    subprocess.run(['make', 'clean'], capture_output=True)
    result = subprocess.run(['make', '-j'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Build failed!")
        print(result.stderr)
        return
    print("Build successful!\n")
    
    # Open CSV file
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    fieldnames = ['executable', 'config', 'N', 'tsteps', 'threads', 
                  'avg_time', 'min_time', 'max_time', 'gflops', 
                  'checksum', 'num_runs', 'all_times']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    
    # Run benchmarks
    for idx, (executable, config_name, args) in enumerate(BENCHMARKS, 1):
        print(f"[{idx}/{len(BENCHMARKS)}] Running {config_name}...")
        
        times, checksum = run_benchmark(executable, config_name, args)
        
        if times:
            avg = sum(times) / len(times)
            gflops = (N * N * TSTEPS * 6) / (avg * 1e9)
            checksum_str = f"{checksum:.10e}" if checksum else "N/A"
            print(f"  ✓ Completed: avg={avg:.6f}s, GFLOPS={gflops:.2f}, checksum={checksum_str}")
        else:
            print(f"  ✗ Failed")
        
        # Write result immediately
        write_result(writer, executable, config_name, times, checksum)
        csv_file.flush()
        print()
    
    csv_file.close()
    
    print("=" * 70)
    print(f"Benchmarking complete! Results saved to {OUTPUT_CSV}")
    print("=" * 70)

if __name__ == "__main__":
    main()