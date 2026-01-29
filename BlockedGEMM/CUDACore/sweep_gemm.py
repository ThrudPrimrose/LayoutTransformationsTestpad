#!/usr/bin/env python3
"""
GEMM Tiling Parameter Autotuner

This script compiles and tests different tiling parameter combinations
for the 2D block-tiled GEMM kernel to find optimal configurations.
"""

import subprocess
import os
import json
import csv
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import itertools

class GEMMAutotuner:
    def __init__(self, cuda_file: str = "gemm_fp64_2d_tiled.cu", 
                 output_dir: str = "tuning_results"):
        self.cuda_file = cuda_file
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # GPU constraints
        self.MAX_THREADS_PER_BLOCK = 1024
        self.MAX_SHARED_MEMORY = 48 * 1024  # 48 KB in bytes
        self.DOUBLE_SIZE = 8  # bytes
        
    def validate_params(self, BM: int, BN: int, BK: int, TM: int, TN: int) -> Tuple[bool, str]:
        """Validate tiling parameters against GPU constraints"""
        
        # Check divisibility
        if BM % TM != 0:
            return False, f"BM ({BM}) must be divisible by TM ({TM})"
        if BN % TN != 0:
            return False, f"BN ({BN}) must be divisible by TN ({TN})"
        
        # Check thread count
        threads_per_block = (BM * BN) // (TM * TN)
        if threads_per_block > self.MAX_THREADS_PER_BLOCK:
            return False, f"Threads per block ({threads_per_block}) exceeds max ({self.MAX_THREADS_PER_BLOCK})"
        if threads_per_block <= 0:
            return False, f"Invalid thread count ({threads_per_block})"
        
        # Check shared memory
        smem_size = (BM * BK + BK * BN) * self.DOUBLE_SIZE
        if smem_size > self.MAX_SHARED_MEMORY:
            return False, f"Shared memory ({smem_size} bytes) exceeds max ({self.MAX_SHARED_MEMORY} bytes)"
        
        # Check if parameters are powers of 2 or common values
        # This is a soft constraint for better performance
        valid_values = [4, 8, 16, 32, 64, 128, 256]
        if BM not in valid_values or BN not in valid_values:
            return False, f"BM and BN should be in {valid_values}"
        if BK not in [4, 8, 16, 32]:
            return False, f"BK should be in [4, 8, 16, 32]"
        if TM not in [4, 8, 16] or TN not in [4, 8, 16]:
            return False, f"TM and TN should be in [4, 8, 16]"
        
        return True, "OK"
    
    def add_dispatch_entry(self, BM: int, BN: int, BK: int, TM: int, TN: int) -> str:
        """Generate dispatch macro entry for a parameter combination"""
        return f"        else DISPATCH_GEMM({BM}, {BN}, {BK}, {TM}, {TN})"
    
    def update_dispatch_table(self, param_sets: List[Tuple[int, int, int, int, int]]):
        """Update the CUDA file with all parameter combinations in dispatch table"""
        
        print(f"Updating dispatch table with {len(param_sets)} configurations...")
        
        # Read the original file
        with open(self.cuda_file, 'r') as f:
            content = f.read()
        
        # Generate dispatch entries
        dispatch_entries = []
        for i, (BM, BN, BK, TM, TN) in enumerate(param_sets):
            if i == 0:
                # First entry without 'else'
                entry = f"        DISPATCH_GEMM({BM}, {BN}, {BK}, {TM}, {TN})"
            else:
                entry = f"        else DISPATCH_GEMM({BM}, {BN}, {BK}, {TM}, {TN})"
            dispatch_entries.append(entry)
        
        dispatch_block = "\n".join(dispatch_entries)
        
        # Find and replace the dispatch section
        start_marker = "        // Add common configurations here"
        end_marker = "        else {"
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx)
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Could not find dispatch table markers in CUDA file")
        
        # Replace the dispatch table
        new_content = (
            content[:start_idx] +
            start_marker + "\n" +
            dispatch_block + "\n" +
            content[end_idx:]
        )
        
        # Write back
        with open(self.cuda_file, 'w') as f:
            f.write(new_content)
        
        print("Dispatch table updated successfully")
    
    def compile_kernel(self, arch: str = "sm_80") -> bool:
        """Compile the CUDA kernel"""
        
        output_binary = "gemm_test"
        compile_cmd = [
            "nvcc",
            "-O3",
            "-Xcompiler", "-fopenmp",
            "-Xcompiler", "-fPIC",
            "-Xcompiler", "-O3",
            f"-arch={arch}",
            "-o", output_binary,
            self.cuda_file
        ]
        
        print(f"Compiling with: {' '.join(compile_cmd)}")
        
        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"Compilation failed:")
                print(result.stderr)
                return False
            
            print("Compilation successful")
            return True
            
        except subprocess.TimeoutExpired:
            print("Compilation timeout")
            return False
        except Exception as e:
            print(f"Compilation error: {e}")
            return False
    
    def run_benchmark(self, M: int, N: int, K: int,
                     BM: int, BN: int, BK: int, TM: int, TN: int,
                     verify: bool = False) -> Optional[float]:
        """Run benchmark for specific parameters and return GFLOPS"""
        
        binary = "./gemm_test"
        
        # Create a simple test program that calls the function
        test_prog = f"""
#include <stdio.h>
#include <cuda_runtime.h>

extern "C" double run_gemm_with_params(int M, int N, int K,
                                       int BM, int BN, int BK, int TM, int TN,
                                       bool verify_correctness);

int main() {{
    int M = {M}, N = {N}, K = {K};
    int BM = {BM}, BN = {BN}, BK = {BK}, TM = {TM}, TN = {TN};
    
    double time_ms = run_gemm_with_params(M, N, K, BM, BN, BK, TM, TN, {str(verify).lower()});
    
    if (time_ms > 0) {{
        double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
        printf("TIME_MS:%.6f\\n", time_ms);
        printf("GFLOPS:%.2f\\n", gflops);
        return 0;
    }} else {{
        printf("ERROR\\n");
        return 1;
    }}
}}
"""
        
        # Write test program
        test_file = "test_runner.cu"
        with open(test_file, 'w') as f:
            f.write(test_prog)
        
        # Compile and run (combining both kernels)
        compile_cmd = [
            "nvcc", "-O3", "-arch=sm_80",
            "-o", "test_runner",
            test_file, self.cuda_file
        ]
        
        try:
            # Compile
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"  Test compilation failed: {result.stderr[:200]}")
                return None
            
            # Run
            result = subprocess.run(
                ["./test_runner"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"  Execution failed")
                return None
            
            # Parse output
            output = result.stdout
            time_ms = None
            gflops = None
            
            for line in output.split('\n'):
                if line.startswith("TIME_MS:"):
                    time_ms = float(line.split(':')[1])
                elif line.startswith("GFLOPS:"):
                    gflops = float(line.split(':')[1])
            
            if time_ms is not None and gflops is not None:
                return gflops
            else:
                print(f"  Could not parse output")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"  Execution timeout")
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None
        finally:
            # Cleanup
            for f in ["test_runner", test_file]:
                if os.path.exists(f):
                    os.remove(f)
    
    def test_configuration(self, M: int, N: int, K: int,
                          BM: int, BN: int, BK: int, TM: int, TN: int,
                          verify: bool = False) -> Dict:
        """Test a single configuration"""
        
        result = {
            'M': M, 'N': N, 'K': K,
            'BM': BM, 'BN': BN, 'BK': BK,
            'TM': TM, 'TN': TN,
            'valid': False,
            'gflops': None,
            'threads_per_block': None,
            'smem_bytes': None,
            'error': None
        }
        
        # Validate parameters
        valid, msg = self.validate_params(BM, BN, BK, TM, TN)
        if not valid:
            result['error'] = msg
            return result
        
        result['valid'] = True
        result['threads_per_block'] = (BM * BN) // (TM * TN)
        result['smem_bytes'] = (BM * BK + BK * BN) * self.DOUBLE_SIZE
        
        # Run benchmark
        gflops = self.run_benchmark(M, N, K, BM, BN, BK, TM, TN, verify)
        
        if gflops is not None:
            result['gflops'] = gflops
        else:
            result['error'] = "Execution failed"
        
        return result
    
    def generate_parameter_space(self) -> List[Tuple[int, int, int, int, int]]:
        """Generate a reasonable parameter space to explore"""
        
        param_sets = []
        
        # Common block sizes
        block_sizes = [64, 128, 256]
        
        # K-dimension chunk sizes
        bk_values = [8, 16]
        
        # Thread tile sizes
        thread_tiles = [4, 8]
        
        # Generate combinations
        for BM in block_sizes:
            for BN in block_sizes:
                for BK in bk_values:
                    for TM in thread_tiles:
                        for TN in thread_tiles:
                            valid, _ = self.validate_params(BM, BN, BK, TM, TN)
                            if valid:
                                param_sets.append((BM, BN, BK, TM, TN))
        
        # Also add some rectangular blocks
        rectangular = [
            (128, 64, 8, 8, 8),
            (64, 128, 8, 8, 8),
            (256, 128, 8, 8, 8),
            (128, 256, 8, 8, 8),
        ]
        
        for params in rectangular:
            valid, _ = self.validate_params(*params)
            if valid and params not in param_sets:
                param_sets.append(params)
        
        return param_sets
    
    def run_tuning(self, M: int = 4096, N: int = 4096, K: int = 4096,
                   verify_first: bool = True):
        """Run autotuning across parameter space"""
        
        print(f"\n{'='*70}")
        print(f"GEMM Autotuner - Matrix size: {M}x{N}x{K}")
        print(f"{'='*70}\n")
        
        # Generate parameter space
        param_sets = self.generate_parameter_space()
        print(f"Testing {len(param_sets)} parameter combinations\n")
        
        # Update dispatch table with all combinations
        self.update_dispatch_table(param_sets)
        
        # Compile once with all configurations
        if not self.compile_kernel():
            print("Failed to compile kernel")
            return
        
        print(f"\n{'='*70}")
        print("Running benchmarks...")
        print(f"{'='*70}\n")
        
        # Test each configuration
        for i, (BM, BN, BK, TM, TN) in enumerate(param_sets):
            print(f"[{i+1}/{len(param_sets)}] Testing BM={BM:3d} BN={BN:3d} BK={BK:2d} TM={TM:2d} TN={TN:2d} ... ", end='', flush=True)
            
            # Verify first configuration
            verify = verify_first and i == 0
            
            result = self.test_configuration(M, N, K, BM, BN, BK, TM, TN, verify)
            
            if result['gflops'] is not None:
                print(f"{result['gflops']:7.2f} GFLOPS")
            else:
                print(f"FAILED - {result.get('error', 'Unknown error')}")
            
            self.results.append(result)
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save results to JSON and CSV files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = os.path.join(self.output_dir, f"results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {json_file}")
        
        # Save CSV
        csv_file = os.path.join(self.output_dir, f"results_{timestamp}.csv")
        if self.results:
            keys = self.results[0].keys()
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"Results saved to: {csv_file}")
    
    def print_summary(self):
        """Print summary of results"""
        
        successful = [r for r in self.results if r['gflops'] is not None]
        
        if not successful:
            print("\nNo successful runs!")
            return
        
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total configurations tested: {len(self.results)}")
        print(f"Successful runs: {len(successful)}")
        print(f"Failed runs: {len(self.results) - len(successful)}")
        
        # Sort by performance
        successful.sort(key=lambda x: x['gflops'], reverse=True)
        
        print(f"\n{'='*70}")
        print("TOP 10 CONFIGURATIONS")
        print(f"{'='*70}")
        print(f"{'Rank':<6} {'BM':<5} {'BN':<5} {'BK':<5} {'TM':<5} {'TN':<5} {'GFLOPS':<10} {'Threads':<8} {'SMEM(KB)'}")
        print(f"{'-'*70}")
        
        for i, result in enumerate(successful[:10]):
            print(f"{i+1:<6} "
                  f"{result['BM']:<5} "
                  f"{result['BN']:<5} "
                  f"{result['BK']:<5} "
                  f"{result['TM']:<5} "
                  f"{result['TN']:<5} "
                  f"{result['gflops']:<10.2f} "
                  f"{result['threads_per_block']:<8} "
                  f"{result['smem_bytes']/1024:.1f}")
        
        # Best configuration
        best = successful[0]
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION:")
        print(f"  BM={best['BM']}, BN={best['BN']}, BK={best['BK']}, TM={best['TM']}, TN={best['TN']}")
        print(f"  Performance: {best['gflops']:.2f} GFLOPS")
        print(f"  Threads per block: {best['threads_per_block']}")
        print(f"  Shared memory: {best['smem_bytes']/1024:.1f} KB")
        print(f"{'='*70}\n")


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='GEMM Tiling Parameter Autotuner')
    parser.add_argument('--M', type=int, default=4096, help='Matrix M dimension')
    parser.add_argument('--N', type=int, default=4096, help='Matrix N dimension')
    parser.add_argument('--K', type=int, default=4096, help='Matrix K dimension')
    parser.add_argument('--cuda-file', type=str, default='gemm_fp64_2d_tiled.cu',
                       help='CUDA source file')
    parser.add_argument('--output-dir', type=str, default='tuning_results',
                       help='Output directory for results')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip correctness verification')
    
    args = parser.parse_args()
    
    # Create autotuner
    autotuner = GEMMAutotuner(
        cuda_file=args.cuda_file,
        output_dir=args.output_dir
    )
    
    # Run tuning
    autotuner.run_tuning(
        M=args.M,
        N=args.N,
        K=args.K,
        verify_first=not args.no_verify
    )


if __name__ == "__main__":
    main()