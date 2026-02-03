import pandas as pd
import sys

# Mappings
KERNELS = {
    0: "i-outer, j-inner",
    1: "j-outer, i-inner",
    2: "tiled",
    3: "tiled+copy",
}

LAYOUTS = {
    (0, 0): "A:row, B:row",
    (0, 1): "A:row, B:col",
    (1, 0): "A:col, B:row",
    (1, 1): "A:col, B:col",
}

TILE_CONFIGS = {
    0: "4x4",
    1: "8x8",
    2: "16x16",
    3: "32x32",
    4: "64x64",
    5: "128x128",
}


def compute_reuse_distance(row, N=1024):
    """
    Compute average reuse distance for matrix multiply C = A * B
    Based on kernel type, layout, and tile size.
    
    For row-major: elements in same row are contiguous
    For col-major: elements in same column are contiguous
    
    Reuse distance = number of unique memory accesses between two accesses to the same element
    """
    kernel = row['kernel']
    a_layout = row['a_layout']  # 0=row-major, 1=col-major
    b_layout = row['b_layout']
    tile_sel = row['tile_sel']

    tile_size = int(TILE_CONFIGS.get(tile_sel, "4x4").split('x')[0])
    cache_line_size = 8
    stride = 2048

    # The kernel is A[:] = ( A[:] + B[:] ) * scale

    def tile_reuse():
        if tile_size < cache_line_size:
            non_unit = (tile_size - 1) * ((stride + cache_line_size) // cache_line_size) + 1
            unit = (tile_size * tile_size - tile_size + 1)
            return non_unit + unit
        else:
            non_unit = (tile_size - 1) * ((stride + cache_line_size) // cache_line_size)
            unit = (tile_size * tile_size - tile_size)
            return non_unit + unit


    if kernel == 0:  # i-outer, j-inner: for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
        # A access pattern: A[i,k] - iterates over k in inner-ish loop
        # B access pattern: B[k,j] - iterates over j in inner loop, k in middle
        
        # A reuse: same A[i,k] reused across j loop (N times), distance ~ N (size of j loop)
        # B reuse: B[k,j] - j changes every iteration, k changes every N iterations
        
        if a_layout == 0:  # A row-major: A[i,k] and A[i,k+1] are adjacent
            a_reuse = 1
        else:  # A col-major: A[i,k] and A[i+1,k] are adjacent, but we iterate over k
            a_reuse = stride
        
        if b_layout == 0:  # B row-major: B[k,j] and B[k,j+1] adjacent, j is innermost
            b_reuse = 1
        else:  # B col-major: B[k,j] and B[k+1,j] adjacent
            b_reuse = stride  # stride-N access

        avg_reuse = ((2*a_reuse) + b_reuse) / 3

    elif kernel == 1:  # j-outer, i-inner: for j: for i: for k: C[i,j] += A[i,k] * B[k,j]
        # A access pattern: A[i,k] - i changes in inner-ish loop
        # B access pattern: B[k,j] - j is outermost, reused across i loop
        
        if a_layout == 0:  # A row-major: stride-1 over k, stride-N over i
            a_reuse = stride  # i changes, causing stride-N jumps
        else:  # A col-major: A[i,k] and A[i+1,k] adjacent, i is inner
            a_reuse = 1  # good spatial locality
        
        if b_layout == 0:  # B row-major: B[k,j] and B[k,j+1] adjacent, but j is outer
            b_reuse = stride  # poor reuse
        else:  # B col-major: B[k,j] and B[k+1,j] adjacent
            b_reuse = 1  # reuse across i-loop
        
        avg_reuse = ((2*a_reuse) + b_reuse) / 3
        
    elif kernel == 2:  # tiled, follows schedule of 0, but should be irrelevant
        # Tiling improves reuse by keeping tile in cache
        # Reuse distance bounded by tile working set
        # Regardless of the order, 1 access is strided other access not strided
        reuse = tile_reuse()
        avg_reuse = reuse
    elif kernel == 3:  # tiled+copy, follows schedule of 0
        # Copy to contiguous buffer eliminates layout penalty
        # Reuse distance mainly depends on tile size
        reuse = tile_reuse()
        avg_reuse = reuse
    else:
        avg_reuse = N * N  # unknown, assume worst

    return avg_reuse

def main():
    # Parse data
    if len(sys.argv) < 2:
        print("Usage: python analyze_perf.py <csv_file>, default=results/performance.csv")
        csv_file = "results/performance.csv"
    else:
        csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    # Add human-readable columns
    df['kernel_name'] = df['kernel'].map(KERNELS)
    df['layout'] = df.apply(lambda r: LAYOUTS[(r['a_layout'], r['b_layout'])], axis=1)
    df['tile_size'] = df['tile_sel'].map(TILE_CONFIGS)

    # Create description column
    def make_description(row):
        desc = f"{row['kernel_name']} | {row['layout']}"
        if row['kernel'] in [2, 3]:  # tiled or tiled+copy kernel
            desc += f" | tile={row['tile_size']}"
        return desc

    df['description'] = df.apply(make_description, axis=1)
    
    # Compute reuse distance metric
    df['reuse_distance'] = df.apply(compute_reuse_distance, axis=1)

    # Sort by mean runtime
    df_sorted = df.sort_values('mean_ms').reset_index(drop=True)
    df_sorted['runtime_rank'] = df_sorted.index + 1
    
    # Compute reuse distance rank (lower reuse distance = better = lower rank)
    df_sorted['reuse_rank'] = df_sorted['reuse_distance'].rank(method='min').astype(int)

    # Print results
    print("=" * 140)
    print("RUNTIME ORDERING (fastest to slowest) with REUSE DISTANCE METRIC")
    print("=" * 140)
    print()
    print(f"{'Rank':<5} {'Runtime':<22} {'Description':<50} {'Reuse Dist (Lower Better)':<12} {'Reuse Rank':<10}")
    print("-" * 140)

    for _, row in df_sorted.iterrows():
        runtime_str = f"{row['mean_ms']:8.3f} ms (±{row['stdev_ms']:6.3f})"
        print(f"{row['runtime_rank']:<5} {runtime_str:<22} {row['description']:<50} {row['reuse_distance']:<12.1f} {row['reuse_rank']:<10}")

    print()
    print("=" * 140)
    print("SUMMARY BY KERNEL TYPE")
    print("=" * 140)

    for kernel_id in sorted(df['kernel'].unique()):
        kernel_data = df[df['kernel'] == kernel_id]
        best = kernel_data.loc[kernel_data['mean_ms'].idxmin()]
        worst = kernel_data.loc[kernel_data['mean_ms'].idxmax()]
        
        print(f"\nKernel {kernel_id}: {KERNELS[kernel_id]}")
        print(f"  Best:  {best['mean_ms']:8.3f} ms | {best['layout']}", end="")
        if kernel_id in [2, 3]:
            print(f" | tile={best['tile_size']}")
        else:
            print()
        print(f"  Worst: {worst['mean_ms']:8.3f} ms | {worst['layout']}", end="")
        if kernel_id in [2, 3]:
            print(f" | tile={worst['tile_size']}")
        else:
            print()

    # === TABLE 2: Non-tiled kernels only (kernel 0 and 1) ===
    print("\n")
    df2 = df[df['kernel'].isin([0, 1])].copy()
    
    # Sort by mean runtime
    df2_sorted = df2.sort_values('mean_ms').reset_index(drop=True)
    df2_sorted['runtime_rank'] = df2_sorted.index + 1
    
    # Compute reuse distance rank (lower reuse distance = better = lower rank)
    df2_sorted['reuse_rank'] = df2_sorted['reuse_distance'].rank(method='min').astype(int)

    # Print results
    print("=" * 140)
    print("RUNTIME ORDERING (fastest to slowest) with REUSE DISTANCE METRIC - NON-TILED KERNELS ONLY")
    print("=" * 140)
    print()
    print(f"{'Rank':<5} {'Runtime':<22} {'Description':<50} {'Reuse Dist':<12} {'Reuse Rank':<10}")
    print("-" * 140)

    for _, row in df2_sorted.iterrows():
        runtime_str = f"{row['mean_ms']:8.3f} ms (±{row['stdev_ms']:6.3f})"
        print(f"{row['runtime_rank']:<5} {runtime_str:<22} {row['description']:<50} {row['reuse_distance']:<12.1f} {row['reuse_rank']:<10}")

    print()
    print("=" * 140)
    print("SUMMARY BY KERNEL TYPE (NON-TILED)")
    print("=" * 140)

    for kernel_id in sorted(df2['kernel'].unique()):
        kernel_data = df2[df2['kernel'] == kernel_id]
        best = kernel_data.loc[kernel_data['mean_ms'].idxmin()]
        worst = kernel_data.loc[kernel_data['mean_ms'].idxmax()]
        
        print(f"\nKernel {kernel_id}: {KERNELS[kernel_id]}")
        print(f"  Best:  {best['mean_ms']:8.3f} ms | {best['layout']}")
        print(f"  Worst: {worst['mean_ms']:8.3f} ms | {worst['layout']}")

    print()
    
    # Correlation analysis for non-tiled
    correlation = df2_sorted['mean_ms'].corr(df2_sorted['reuse_distance'])
    print(f"• Correlation (runtime vs reuse distance): {correlation:.3f}")


if __name__ == "__main__":
    main()