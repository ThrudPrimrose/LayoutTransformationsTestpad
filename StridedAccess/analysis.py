from math import prod
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


def compute_reuse_distance(row, N=2048):
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

def compute_tile_closeness(row):
    kernel = row['kernel']
    a_layout = row['a_layout']  # 0=row-major, 1=col-major
    b_layout = row['b_layout']
    tile_sel = row['tile_sel']

    N = 2048

    tile_size = int(TILE_CONFIGS.get(tile_sel, "4x4").split('x')[0])
    stride = 2048

    def compute_closeness(strides, dims):
        total_elements = prod(dims)
        
        # Use analytical formula for large sizes
        if total_elements > 32*32:
            avg_dist = 0
            for dim_size, stride in zip(dims, strides):
                if dim_size > 1:
                    avg_coord_diff = (dim_size - 1) / 3.0
                    avg_dist += avg_coord_diff * stride
            return avg_dist
        
        # Exact computation for small sizes
        total_dist = 0
        for el_i in range(total_elements):
            for el_j in range(total_elements):
                abs_dist = abs(el_i - el_j)
                
                memory_dist = 0
                remaining = abs_dist
                for dim_idx in range(len(dims) - 1, -1, -1):
                    coord_diff = remaining % dims[dim_idx]
                    remaining = remaining // dims[dim_idx]
                    memory_dist += coord_diff * strides[dim_idx]
                
                total_dist += memory_dist

        num_pairs = total_elements * total_elements
        return total_dist / num_pairs

    def compute_address_span(strides, dims):
        """Compute address distance between first and last element of tile."""
        # First element is at address 0
        # Last element is at address sum((dim-1) * stride) for each dimension
        max_addr = sum((d - 1) * s for d, s in zip(dims, strides))
        return max_addr

    if kernel == 0:  # i-outer, j-inner
        a_stride = [stride, 1] if a_layout == 0 else [1, stride]
        b_stride = [stride, 1] if b_layout == 0 else [1, stride]
        a_dim = [1, 8]
        b_dim = [1, 8]
    elif kernel == 1:  # j-outer, i-inner
        a_stride = [1, stride] if a_layout == 0 else [stride, 1]
        b_stride = [1, stride] if b_layout == 0 else [stride, 1]
        a_dim = [1, 8]
        b_dim = [1, 8]
    elif kernel in [2, 3]:  # tiled or tiled+copy
        a_stride = [stride, 1] if a_layout == 0 else [1, stride]
        b_stride = [stride, 1] if b_layout == 0 else [1, stride]
        a_dim = [tile_size, tile_size]
        b_dim = [tile_size, tile_size]
    else:
        return N * N, N * N

    a_closeness = compute_closeness(a_stride, a_dim)
    b_closeness = compute_closeness(b_stride, b_dim)
    avg_closeness = (2 * a_closeness + b_closeness) / 3

    a_span = compute_address_span(a_stride, a_dim)
    b_span = compute_address_span(b_stride, b_dim)
    avg_span = (2 * a_span + b_span) / 3

    return avg_closeness, avg_span

def compute_average_stride(row):
    kernel = row['kernel']
    a_layout = row['a_layout']  # 0=row-major, 1=col-major
    b_layout = row['b_layout']
    tile_sel = row['tile_sel']

    tile_size = int(TILE_CONFIGS.get(tile_sel, "4x4").split('x')[0])
    """Compute average stride between consecutive accesses."""
    if kernel == 0:  # i-outer, j-inner
        av = 1 if a_layout == 0 else 2048
        bv = 1 if b_layout == 0 else 2048
    elif kernel == 1:  # j-outer, i-inner
        av = 1 if a_layout == 1 else 2048
        bv = 1 if b_layout == 1 else 2048
    elif kernel in [2, 3]:  # tiled or tiled+copy
        av = ((tile_size - 1) * 1 + (tile_size - 1) * 2048) / ((tile_size - 1) ** 2)
        bv = ((tile_size - 1) * 1 + (tile_size - 1) * 2048) / ((tile_size - 1) ** 2)
    else:
        return 2048 * 2048

    return (2*av + bv)/3

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
    
    # Compute metrics
    df['reuse_distance'] = df.apply(compute_reuse_distance, axis=1)
    closeness_span = df.apply(compute_tile_closeness, axis=1)
    df['tile_closeness'] = closeness_span.apply(lambda x: x[0])
    df['address_span'] = closeness_span.apply(lambda x: x[1])
    df['avg_stride'] = df.apply(compute_average_stride, axis=1)

    # Define datasets to analyze
    datasets = [
        ("ALL KERNELS", df.copy()),
        ("NON-TILED KERNELS ONLY", df[df['kernel'].isin([0, 1])].copy()),
    ]

    for title, data in datasets:
        # Sort by mean runtime
        df_sorted = data.sort_values('mean_ms').reset_index(drop=True)
        df_sorted['runtime_rank'] = df_sorted.index + 1
        
        # Compute ranks (lower = better)
        df_sorted['reuse_rank'] = df_sorted['reuse_distance'].rank(method='min').astype(int)
        df_sorted['closeness_rank'] = df_sorted['tile_closeness'].rank(method='min').astype(int)
        df_sorted['span_rank'] = df_sorted['address_span'].rank(method='min').astype(int)
        df_sorted['stride_rank'] = df_sorted['avg_stride'].rank(method='min').astype(int)

        # Print results
        print("=" * 200)
        print(f"RUNTIME ORDERING (fastest to slowest) - {title}")
        print("=" * 200)
        print()
        print(f"{'Rank':<5} {'Runtime':<22} {'Description':<45} {'ReuseDist':<10} {'ReuseRk':<8} {'Closeness':<12} {'CloseRk':<8} {'AddrSpan':<12} {'SpanRk':<8} {'AvgStride':<10} {'StrideRk':<8}")
        print("-" * 200)

        for _, row in df_sorted.iterrows():
            runtime_str = f"{row['mean_ms']:8.3f} ms (±{row['stdev_ms']:6.3f})"
            print(f"{row['runtime_rank']:<5} {runtime_str:<22} {row['description']:<45} {row['reuse_distance']:<10.1f} {row['reuse_rank']:<8} {row['tile_closeness']:<12.1f} {row['closeness_rank']:<8} {row['address_span']:<12.1f} {row['span_rank']:<8} {row['avg_stride']:<10.1f} {row['stride_rank']:<8}")

        print()
        print("=" * 200)
        print(f"SUMMARY BY KERNEL TYPE ({title})")
        print("=" * 200)

        for kernel_id in sorted(data['kernel'].unique()):
            kernel_data = data[data['kernel'] == kernel_id]
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

        # Correlation analysis
        print()
        print(f"• Correlation (runtime vs reuse_distance): {df_sorted['mean_ms'].corr(df_sorted['reuse_distance']):.3f}")
        print(f"• Correlation (runtime vs tile_closeness): {df_sorted['mean_ms'].corr(df_sorted['tile_closeness']):.3f}")
        print(f"• Correlation (runtime vs address_span):   {df_sorted['mean_ms'].corr(df_sorted['address_span']):.3f}")
        print(f"• Correlation (runtime vs avg_stride):     {df_sorted['mean_ms'].corr(df_sorted['avg_stride']):.3f}")
        print("\n")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()