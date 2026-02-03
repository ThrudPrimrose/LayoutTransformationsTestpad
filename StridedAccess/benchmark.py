#!/usr/bin/env python3
"""
Array Kernel Benchmark Script
Tests 2 kernels (loop orders) × 4 data layouts (A/B row/col major combinations)
"""

import subprocess
import os
import statistics
from pathlib import Path

# Configuration
N = 2048
M = 2048
NUM_RUNS = 10
NUM_PAPI_RUNS = 3

# Kernels: loop order
KERNELS = {
    0: "i-outer, j-inner",
    1: "j-outer, i-inner",
    2: "tiled",
    3: "tiled+copy",
}

# Data layouts: (A_layout, B_layout) where 0=row-major, 1=col-major
LAYOUTS = [
    (0, 0),  # A:row, B:row
    (0, 1),  # A:row, B:col
    (1, 0),  # A:col, B:row
    (1, 1),  # A:col, B:col
]

TILE_CONFIGS = {
    0: "4x4",
    1: "8x8",
    2: "16x16",
    3: "32x32",
    4: "64x64",
    5: "128x128",
}
TILES = TILE_CONFIGS

def layout_str(a_layout, b_layout):
    a = "row" if a_layout == 0 else "col"
    b = "row" if b_layout == 0 else "col"
    return f"A:{a}, B:{b}"

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

# Directories
SCRIPT_DIR = Path(__file__).parent.resolve()
BUILD_DIR = SCRIPT_DIR / "build"
REPORT_DIR = SCRIPT_DIR / "opt_reports"
RESULTS_DIR = SCRIPT_DIR / "results"

# Compiler and flags
CXX = "g++"
BASE_FLAGS = ["-std=c++17", "-march=native", "-mtune=native"]
OPT_FLAGS = ["-O3", "-ffast-math", "-funroll-loops", "-ftree-vectorize"]
REPORT_FLAGS = ["-fdump-tree-all-graph", "-fdump-ipa-all", "-fdump-rtl-all", "-save-temps", "-fverbose-asm"]

def setup_dirs():
    BUILD_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

def variant_name(kernel, a_layout, b_layout, tile_selection):
    return f"k{kernel}_A{a_layout}_B{b_layout}_tl{tile_selection}"

def compile_variant(kernel, a_layout, b_layout, tile_selection, papi_metric="", with_reports=False):
    src = SCRIPT_DIR / "kernel.cpp"
    suffix = f"_papi_{papi_metric.replace('PAPI_', '')}" if papi_metric else ""
    name = variant_name(kernel, a_layout, b_layout, tile_selection)
    exe = BUILD_DIR / f"{name}{suffix}"
    
    defines = [
        f"-DN={N}", f"-DM={M}",
        f"-DKERNEL={kernel}",
        f"-DA_LAYOUT={a_layout}",
        f"-DB_LAYOUT={b_layout}",
        f'-DPAPI_METRIC="{papi_metric}"',
        f'-DTILE_SEL={tile_selection}'
    ]
    
    libs = []
    if papi_metric:
        defines.append("-DUSE_PAPI")
        libs = ["-lpapi", "-fopenmp"]
    
    flags = BASE_FLAGS + OPT_FLAGS + defines
    
    if with_reports:
        report_file = REPORT_DIR / f"{name}_opt.txt"
        flags += [f"-fopt-info-all={report_file}"]
    
    cmd = [CXX] + flags + [str(src), "-o", str(exe)] + libs
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
        if result.returncode != 0:
            print(f"Compile error ({name}):\n{result.stderr}")
            return None
        return exe
    except Exception as e:
        print(f"Compile exception: {e}")
        return None

def compile_with_full_reports(kernel, a_layout, b_layout, tile_selection):
    src = SCRIPT_DIR / "kernel.cpp"
    name = variant_name(kernel, a_layout, b_layout, tile_selection)
    report_subdir = REPORT_DIR / name
    report_subdir.mkdir(exist_ok=True)
    
    defines = [
        f"-DN={N}", f"-DM={M}",
        f"-DKERNEL={kernel}",
        f"-DA_LAYOUT={a_layout}",
        f"-DB_LAYOUT={b_layout}",
        f"-DTILE_SEL={tile_selection}",
        '-DPAPI_METRIC=""',
    ]
    
    flags = BASE_FLAGS + OPT_FLAGS + defines + REPORT_FLAGS
    exe = report_subdir / name
    cmd = [CXX] + flags + [str(src), "-o", str(exe)]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, cwd=str(report_subdir))
        print(f"  Reports saved: {report_subdir}")
    except Exception as e:
        print(f"  Report error: {e}")

def run_executable(exe, num_runs=1):
    results = []
    for _ in range(num_runs):
        try:
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                line = result.stdout.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 6:
                        results.append({
                            'kernel': int(parts[0]),
                            'a_layout': int(parts[1]),
                            'b_layout': int(parts[2]),
                            'tile_sel': int(parts[3]),
                            'time_ms': float(parts[4]),
                            'papi': int(parts[5]),
                            'checksum': float(parts[6])
                        })
        except subprocess.TimeoutExpired:
            print("  Timeout!")
        except Exception as e:
            print(f"  Run error: {e}")
    return results


def tile_str(kernel_id: int, tile_sel_id: int):
    if (kernel_id == 2 or kernel_id == 3):
        return f" {TILES[tile_sel_id]}"
    else:
        return ""

def benchmark_performance():
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARKING")
    print("="*70)
    
    perf_results = {}
    
    for kernel in KERNELS:
        for a_layout, b_layout in LAYOUTS:
            for tile_selection in [0, 1, 2, 3, 4, 5]:
                name = variant_name(kernel, a_layout, b_layout, tile_selection)
                # Tile selection relevant only for kernel 2
                if (kernel != 2 and kernel != 3) and tile_selection != 0:
                    continue
                desc = f"K{kernel} ({KERNELS[kernel]}{tile_str(kernel, tile_selection)}), {layout_str(a_layout, b_layout)}"
                print(f"\n{desc}")
                
                exe = compile_variant(kernel, a_layout, b_layout, tile_selection, with_reports=True)
                if not exe:
                    print("Failed compiling")
                    continue
                
                results = run_executable(exe, NUM_RUNS)
                if results:
                    times = [r['time_ms'] for r in results]
                    perf_results[name] = {
                        'kernel': kernel,
                        'a_layout': a_layout,
                        'b_layout': b_layout,
                        'tile_sel': tile_selection,
                        'mean': statistics.mean(times),
                        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
                        'min': min(times),
                        'max': max(times),
                        'checksum': results[0]['checksum']
                    }
                    print(f"  Time: {perf_results[name]['mean']:.3f} ± {perf_results[name]['stdev']:.3f} ms")
                else:
                    print("Failed running executable")
        
    return perf_results

def benchmark_papi():
    print("\n" + "="*70)
    print("PAPI METRICS COLLECTION")
    print("="*70)
    
    papi_results = {}
    for kernel in KERNELS:
        for a_layout, b_layout in LAYOUTS:
            for tile_selection in [0, 1, 2, 3, 4, 5]:
                name = variant_name(kernel, a_layout, b_layout, tile_selection)
                # Tile selection relevant only for kernel 2
                if (kernel != 2 and kernel != 3) and tile_selection != 0:
                    continue
                name = variant_name(kernel, a_layout, b_layout, tile_selection)
                papi_results[name] = {}
    
    for metric_name, metric_code in PAPI_METRICS.items():
        print(f"\nCollecting {metric_name}...")
        
        for kernel in KERNELS:
            for a_layout, b_layout in LAYOUTS:
                for tile_selection in [0, 1, 2, 3, 4, 5]:
                    name = variant_name(kernel, a_layout, b_layout, tile_selection)
                    # Tile selection relevant only for kernel 2
                    if (kernel != 2 and kernel != 3) and tile_selection != 0:
                        continue
                    exe = compile_variant(kernel, a_layout, b_layout, tile_selection, papi_metric=metric_code)
                    if not exe:
                        papi_results[name][metric_name] = None
                        continue
                    
                    results = run_executable(exe, NUM_PAPI_RUNS)
                    if results:
                        values = [r['papi'] for r in results]
                        papi_results[name][metric_name] = int(statistics.median(values))
                    else:
                        papi_results[name][metric_name] = None
    
    return papi_results

def generate_reports():
    print("\n" + "="*70)
    print("GENERATING OPTIMIZATION REPORTS")
    print("="*70)
    
    for kernel in KERNELS:
        for a_layout, b_layout in LAYOUTS:
            for tile_selection in [0,1,2,3,4,5]:
                if kernel != 2 and kernel != 3 and tile_selection != 0:
                    continue
                name = variant_name(kernel, a_layout, b_layout, tile_selection)
                print(f"\n{name}...")
                compile_with_full_reports(kernel, a_layout, b_layout, tile_selection)

def print_results(perf_results, papi_results):
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    # Performance table
    print(f"\n{'Variant':<50} {'Time (ms)':<18} {'Speedup':<10}")
    print("-" * 78)
    
    if perf_results:
        baseline = min(r['mean'] for r in perf_results.values())
        for name in sorted(perf_results.keys()):
            r = perf_results[name]
            desc = f"K{r['kernel']} ({KERNELS[r['kernel']][:15]}), {layout_str(r['a_layout'], r['b_layout'])}"
            speedup = r['mean'] / baseline
            print(f"{desc:<50} {r['mean']:>7.3f} ± {r['stdev']:<6.3f}  {speedup:>6.2f}x")
    
    # PAPI table
    variants = sorted(papi_results.keys())
    print(f"\n{'Metric':<10}", end="")
    for v in variants:
        print(f"{v:>12}", end="")
    print()
    print("-" * (10 + 12 * len(variants)))
    
    for metric in PAPI_METRICS:
        print(f"{metric:<10}", end="")
        for v in variants:
            val = papi_results.get(v, {}).get(metric)
            if val is not None:
                if val > 1e9:
                    print(f"{val/1e9:>11.2f}G", end="")
                elif val > 1e6:
                    print(f"{val/1e6:>11.2f}M", end="")
                elif val > 1e3:
                    print(f"{val/1e3:>11.2f}K", end="")
                else:
                    print(f"{val:>12}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

def save_results(perf_results, papi_results):
    # Performance CSV
    perf_file = RESULTS_DIR / "performance.csv"
    with open(perf_file, 'w') as f:
        f.write("variant,kernel,a_layout,b_layout,tile_sel,mean_ms,stdev_ms,min_ms,max_ms,checksum\n")
        for name in sorted(perf_results.keys()):
            r = perf_results[name]
            f.write(f"{name},{r['kernel']},{r['a_layout']},{r['b_layout']},{r['tile_sel']},{r['mean']:.6f},{r['stdev']:.6f},{r['min']:.6f},{r['max']:.6f},{r['checksum']:.6f}\n")
            f.flush()

    # PAPI CSV
    papi_file = RESULTS_DIR / "papi_metrics.csv"
    with open(papi_file, 'w') as f:
        header = "variant,kernel,a_layout,b_layout," + ",".join(PAPI_METRICS.keys())
        f.write(header + "\n")
        for name in sorted(papi_results.keys()):
            # Parse name to get kernel and layouts
            parts = name.split('_')
            kernel = parts[0][1:]  # k0 -> 0
            a_layout = parts[1][1:]  # A0 -> 0
            b_layout = parts[2][1:]  # B0 -> 0
            row = [name, kernel, a_layout, b_layout]
            for metric in PAPI_METRICS:
                val = papi_results.get(name, {}).get(metric)
                row.append(str(val) if val is not None else "")
            f.write(",".join(row) + "\n")
    
    print(f"\nResults saved to {RESULTS_DIR}/")

def check_papi():
    try:
        result = subprocess.run(["papi_avail", "-a"], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def main():
    setup_dirs()
    
    print("Array Kernel Benchmark")
    print(f"Array size: {N} x {M}")
    print(f"Kernels: {len(KERNELS)} (loop orders)")
    print(f"Layouts: {len(LAYOUTS)} (A/B row/col combinations)")
    print(f"Total variants: {len(KERNELS) * len(LAYOUTS)}")
    print(f"Performance runs: {NUM_RUNS}")
    
    papi_available = check_papi()
    if not papi_available:
        print("\nWarning: PAPI not available, skipping PAPI metrics")
    
    perf_results = benchmark_performance()
    
    papi_results = {}
    if papi_available:
        papi_results = benchmark_papi()
    else:
        for kernel in KERNELS:
            for a_layout, b_layout in LAYOUTS:
                name = variant_name(kernel, a_layout, b_layout)
                papi_results[name] = {}
    
    generate_reports()
    print_results(perf_results, papi_results)
    save_results(perf_results, papi_results)
    
    print("\nDone!")

if __name__ == "__main__":
    main()