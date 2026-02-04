#!/usr/bin/env python3
"""
CUDA Array Kernel Benchmark Script
Tests 4 kernels × 4 data layouts × block tile sizes × thread tile sizes
Uses Nsight Compute (ncu) for GPU metrics collection
"""

import subprocess
import os
import statistics
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configuration
N = 4096
M = 4096
NUM_RUNS = 10
NUM_NCU_RUNS = 1  # ncu is slow, typically 1 run suffices

# Kernels: loop order / tiling strategy
KERNELS = {
    0: "i-outer, j-inner",
    1: "j-outer, i-inner",
    2: "tiled",
    3: "tiled+smem",
}

# Data layouts: (A_layout, B_layout) where 0=row-major, 1=col-major
LAYOUTS = [
    (0, 0),  # A:row, B:row
    (0, 1),  # A:row, B:col
    (1, 0),  # A:col, B:row
    (1, 1),  # A:col, B:col
]

# Per-thread tile sizes (THREAD_TILE_SEL) - elements computed per thread
THREAD_TILE_CONFIGS = {
    0: "1x1",
    1: "2x2",
    2: "4x4",
}

def layout_str(a_layout: int, b_layout: int) -> str:
    a = "row" if a_layout == 0 else "col"
    b = "row" if b_layout == 0 else "col"
    return f"A:{a}, B:{b}"

# Directories
SCRIPT_DIR = Path(__file__).parent.resolve()
BUILD_DIR = SCRIPT_DIR / "build"
NCU_DIR = SCRIPT_DIR / "ncu_reports"
RESULTS_DIR = SCRIPT_DIR / "results"

# Compiler and flags
NVCC = "nvcc"
BASE_FLAGS = ["-std=c++17"]
OPT_FLAGS = ["-O3", "-lineinfo"]
ARCH_FLAG = "-arch=sm_80"  # Adjust for your GPU (sm_70, sm_80, sm_86, sm_89, sm_90)

def setup_dirs():
    BUILD_DIR.mkdir(exist_ok=True)
    NCU_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

def variant_name(kernel: int, a_layout: int, b_layout: int,
                  thread_tile_sel: int) -> str:
    return f"k{kernel}_A{a_layout}_B{b_layout}_tt{thread_tile_sel}"

def get_valid_combinations() -> List[tuple]:
    """Generate valid (kernel, a_layout, b_layout, tile_sel, thread_tile_sel) combinations."""
    combinations = []
    
    for kernel in KERNELS:
        for a_layout, b_layout in LAYOUTS:
            if kernel in [0, 1]:
                # Non-tiled kernels: only one config needed
                combinations.append((kernel, a_layout, b_layout, 0))
            else:
                # Tiled kernels: iterate tile sizes and thread tile sizes
                for thread_tile_sel in THREAD_TILE_CONFIGS:
                    # Validate: tile_size must be >= thread_tile_size and divisible
                    combinations.append((kernel, a_layout, b_layout, thread_tile_sel))
    
    return combinations

def compile_variant(kernel: int, a_layout: int, b_layout: int,
                    thread_tile_sel: int) -> Optional[Path]:
    """Compile a specific variant."""
    src = SCRIPT_DIR / "elementwise_cuda.cu"
    name = variant_name(kernel, a_layout, b_layout, thread_tile_sel)
    exe = BUILD_DIR / name
    
    defines = [
        f"-DN={N}", f"-DM={M}",
        f"-DKERNEL={kernel}",
        f"-DA_LAYOUT={a_layout}",
        f"-DB_LAYOUT={b_layout}",
        f"-DTHREAD_TILE_SEL={thread_tile_sel}",
    ]
    
    cmd = [NVCC, ARCH_FLAG] + BASE_FLAGS + OPT_FLAGS + defines + [str(src), "-o", str(exe)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
        if result.returncode != 0:
            print(f"  Compile error ({name}):\n{result.stderr}")
            return None
        return exe
    except Exception as e:
        print(f"  Compile exception: {e}")
        return None

def run_executable(exe: Path, num_runs: int = 1) -> List[Dict[str, Any]]:
    """Run executable and parse output."""
    results = []
    for _ in range(num_runs):
        try:
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                line = result.stdout.strip()
                if line:
                    parts = line.split(',')
                    # Output: kernel,a_layout,b_layout,tile_size,thread_tile,time_ms,papi_value,checksum
                    if len(parts) >= 8:
                        results.append({
                            'kernel': int(parts[0]),
                            'a_layout': int(parts[1]),
                            'b_layout': int(parts[2]),
                            'block_tile': int(parts[3]),
                            'thread_tile': int(parts[4]),
                            'time_ms': float(parts[5]),
                            'gpu_metric': int(parts[6]),
                            'checksum': float(parts[7])
                        })
        except subprocess.TimeoutExpired:
            print("  Timeout!")
        except Exception as e:
            print(f"  Run error: {e}")
    return results

def run_ncu_profile(exe: Path, name: str) -> Optional[Path]:
    """Run Nsight Compute profiling with --set=full and save report."""
    ncu_report = f"{name}"

    
    # Run ncu with full metrics set
    cmd = [
        "ncu",
        "--set=full",
        "--force-overwrite",
        f"-o={ncu_report}",
        str(exe)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            # Save CSV output
            with open(ncu_csv, 'w') as f:
                f.write(result.stdout)
            return ncu_csv
        else:
            print(f"  NCU error: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("  NCU timeout!")
        return None
    except FileNotFoundError:
        print("  NCU not found! Install Nsight Compute or add to PATH.")
        return None
    except Exception as e:
        print(f"  NCU exception: {e}")
        return None

def parse_ncu_csv(csv_path: Path) -> Dict[str, Any]:
    """Parse NCU CSV output and extract metrics."""
    metrics = {}
    
    try:
        with open(csv_path, 'r') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        # Find header line (typically starts with "ID" or contains "Metric Name")
        header_idx = -1
        for i, line in enumerate(lines):
            if '"ID"' in line or 'ID,' in line or '"Metric Name"' in line:
                header_idx = i
                break
        
        if header_idx >= 0:
            # Parse as CSV
            reader = csv.DictReader(lines[header_idx:])
            for row in reader:
                metric_name = row.get('Metric Name', row.get('metric_name', ''))
                metric_value = row.get('Metric Value', row.get('metric_value', ''))
                
                if metric_name and metric_value:
                    try:
                        value = metric_value.replace(',', '').strip()
                        if value.endswith('K'):
                            metrics[metric_name] = float(value[:-1]) * 1e3
                        elif value.endswith('M'):
                            metrics[metric_name] = float(value[:-1]) * 1e6
                        elif value.endswith('G'):
                            metrics[metric_name] = float(value[:-1]) * 1e9
                        elif value.endswith('T'):
                            metrics[metric_name] = float(value[:-1]) * 1e12
                        elif value.endswith('%'):
                            metrics[metric_name] = float(value[:-1])
                        else:
                            metrics[metric_name] = float(value)
                    except ValueError:
                        metrics[metric_name] = metric_value
                        
    except Exception as e:
        print(f"  Parse error: {e}")
    
    return metrics

def tile_str(kernel: int, thread_tile_sel: int) -> str:
    """Generate tile description string for tiled kernels."""
    if kernel in [2, 3]:
        return f" thr:{THREAD_TILE_CONFIGS[thread_tile_sel]}"
    return ""

def benchmark_performance() -> Dict[str, Dict[str, Any]]:
    """Run performance benchmarks for all variants."""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARKING")
    print("="*70)
    
    perf_results = {}
    combinations = get_valid_combinations()
    
    for kernel, a_layout, b_layout, thread_tile_sel in combinations:
        name = variant_name(kernel, a_layout, b_layout, thread_tile_sel)
        desc = f"K{kernel} ({KERNELS[kernel]}{tile_str(kernel, thread_tile_sel)}), {layout_str(a_layout, b_layout)}"
        print(f"\n{desc}")
        
        exe = compile_variant(kernel, a_layout, b_layout, thread_tile_sel)
        if not exe:
            continue
        
        results = run_executable(exe, NUM_RUNS)
        if results:
            times = [r['time_ms'] for r in results]
            perf_results[name] = {
                'kernel': kernel,
                'a_layout': a_layout,
                'b_layout': b_layout,
                'thread_tile_sel': thread_tile_sel,
                'mean': statistics.mean(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times),
                'checksum': results[0]['checksum']
            }
            print(f"  Time: {perf_results[name]['mean']:.3f} ± {perf_results[name]['stdev']:.3f} ms")
        else:
            print("  Failed running executable")
    
    return perf_results

def benchmark_ncu() -> Dict[str, Dict[str, Any]]:
    """Run Nsight Compute profiling for all variants."""
    print("\n" + "="*70)
    print("NSIGHT COMPUTE PROFILING (--set=full)")
    print("="*70)
    
    ncu_results = {}
    combinations = get_valid_combinations()
    
    for kernel, a_layout, b_layout, thread_tile_sel in combinations:
        name = variant_name(kernel, a_layout, b_layout, thread_tile_sel)
        desc = f"K{kernel} ({KERNELS[kernel]}{tile_str(kernel, thread_tile_sel)}), {layout_str(a_layout, b_layout)}"
        print(f"\n{desc}")
        
        exe = compile_variant(kernel, a_layout, b_layout, thread_tile_sel)
        if not exe:
            ncu_results[name] = {}
            continue
        
        csv_path = run_ncu_profile(exe, name)
        if csv_path:
            ncu_results[name] = parse_ncu_csv(csv_path)
            print(f"  Collected {len(ncu_results[name])} metrics")
        else:
            ncu_results[name] = {}
            print("  Failed profiling")
    
    return ncu_results

def print_results(perf_results: Dict, ncu_results: Dict):
    """Print summary of results."""
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    # Performance table
    print(f"\n{'Variant':<55} {'Time (ms)':<18} {'Speedup':<10}")
    print("-" * 83)
    
    if perf_results:
        baseline = min(r['mean'] for r in perf_results.values())
        for name in sorted(perf_results.keys()):
            r = perf_results[name]
            kernel_desc = KERNELS[r['kernel']][:12]
            tile_desc = tile_str(r['kernel'], r['thread_tile_sel'])
            desc = f"K{r['kernel']} ({kernel_desc}{tile_desc}), {layout_str(r['a_layout'], r['b_layout'])}"
            speedup = baseline / r['mean'] if r['mean'] > 0 else 0
            print(f"{desc:<55} {r['mean']:>7.3f} ± {r['stdev']:<6.3f}  {speedup:>6.2f}x")
    
    # NCU summary (show key metrics for select variants)
    if ncu_results and any(ncu_results.values()):
        print("\n" + "-"*83)
        print("Key NCU Metrics:")
        print("-"*83)
        
        print(f"{'Variant':<45} {'Occupancy':<12} {'DRAM Read':<15} {'L1 Hit%':<10}")
        print("-"*82)
        
        for name in sorted(ncu_results.keys()):
            metrics = ncu_results[name]
            if not metrics:
                continue
                
            occ = metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active', 'N/A')
            dram = metrics.get('dram__bytes_read.sum', 'N/A')
            l1 = metrics.get('l1tex__t_sector_hit_rate.pct', 'N/A')
            
            occ_str = f"{occ:.1f}%" if isinstance(occ, (int, float)) else str(occ)[:10]
            dram_str = f"{dram/1e6:.1f}MB" if isinstance(dram, (int, float)) else str(dram)[:12]
            l1_str = f"{l1:.1f}%" if isinstance(l1, (int, float)) else str(l1)[:8]
            
            print(f"{name:<45} {occ_str:<12} {dram_str:<15} {l1_str:<10}")

def save_results(perf_results: Dict, ncu_results: Dict):
    """Save results to CSV files."""
    # Performance CSV
    perf_file = RESULTS_DIR / "cuda_performance.csv"
    with open(perf_file, 'w') as f:
        f.write("variant,kernel,a_layout,b_layout,thread_tile_sel,mean_ms,stdev_ms,min_ms,max_ms,checksum\n")
        for name in sorted(perf_results.keys()):
            r = perf_results[name]
            f.write(f"{name},{r['kernel']},{r['a_layout']},{r['b_layout']},"
                    f"{r['thread_tile_sel']},{r['mean']:.6f},{r['stdev']:.6f},"
                    f"{r['min']:.6f},{r['max']:.6f},{r['checksum']:.6f}\n")
    
    # NCU metrics CSV
    ncu_file = RESULTS_DIR / "ncu_metrics.csv"

    # Collect all unique metric names
    all_metrics = set()
    for metrics in ncu_results.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)
    
    with open(ncu_file, 'w') as f:
        header = "variant,kernel,a_layout,b_layout,thread_tile_sel," + ",".join(f'"{m}"' for m in all_metrics)
        f.write(header + "\n")
        
        for name in sorted(ncu_results.keys()):
            # Parse variant name: k{kernel}_A{a}_B{b}_bt{tile}_tt{thread}
            parts = name.split('_')
            kernel = parts[0][1:]
            a_layout = parts[1][1:]
            b_layout = parts[2][1:]
            thread_tile_sel = parts[3][2:]
            
            row = [name, kernel, a_layout, b_layout, thread_tile_sel]
            for metric in all_metrics:
                val = ncu_results[name].get(metric, '')
                row.append(str(val) if val != '' else '')
            f.write(",".join(row) + "\n")
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"  - {perf_file}")
    print(f"  - {ncu_file}")

def check_ncu() -> bool:
    """Check if Nsight Compute is available."""
    try:
        result = subprocess.run(["ncu", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"NCU version: {result.stdout.strip().split(chr(10))[0]}")
            return True
        return False
    except:
        return False

def check_nvcc() -> bool:
    """Check if NVCC is available."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'release' in line.lower():
                    print(f"NVCC: {line.strip()}")
                    break
            return True
        return False
    except:
        return False

def check_gpu() -> bool:
    """Check for available GPU."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            gpu_name = result.stdout.strip().split('\n')[0]
            print(f"GPU: {gpu_name}")
            return True
        return False
    except:
        return False

def main():
    setup_dirs()
    
    print("="*70)
    print("CUDA Array Kernel Benchmark")
    print("="*70)
    print(f"Array size: {N} x {M}")
    print(f"Kernels: {len(KERNELS)}")
    for k, v in KERNELS.items():
        print(f"  {k}: {v}")
    print(f"Layouts: {len(LAYOUTS)} (A/B row/col combinations)")
    print(f"Block tile sizes: [128x1]")
    print(f"Thread tile sizes: {list(THREAD_TILE_CONFIGS.values())}")
    
    combinations = get_valid_combinations()
    print(f"Total valid variants: {len(combinations)}")
    print(f"Performance runs per variant: {NUM_RUNS}")
    print(f"NCU runs per variant: {NUM_NCU_RUNS}")
    
    print("\n" + "-"*70)
    print("Environment Check:")
    
    nvcc_ok = check_nvcc()
    if not nvcc_ok:
        print("ERROR: NVCC not found! Please install CUDA toolkit.")
        return
    
    gpu_ok = check_gpu()
    if not gpu_ok:
        print("ERROR: No GPU found!")
        return
    
    ncu_available = check_ncu()
    if not ncu_available:
        print("WARNING: Nsight Compute (ncu) not available, skipping GPU metrics")
    
    # Run performance benchmarks
    perf_results = benchmark_performance()
    
    # Run NCU profiling if available
    ncu_results = {}
    if ncu_available:
        ncu_results = benchmark_ncu()
    else:
        for combo in combinations:
            name = variant_name(*combo)
            ncu_results[name] = {}

    # Print and save results
    print_results(perf_results, ncu_results)
    save_results(perf_results, ncu_results)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)

if __name__ == "__main__":
    main()