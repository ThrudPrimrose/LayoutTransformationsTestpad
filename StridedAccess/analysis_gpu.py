from math import prod
import numpy as np
import pandas as pd
import sys
from scipy.stats import spearmanr, kendalltau

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
    0: "1x1",
    1: "2x2",
    2: "4x4",
    3: "8x8",
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
    thread_tile_sel = row['thread_tile_sel']

    tile_size_x = int(TILE_CONFIGS.get(thread_tile_sel, "1x1").split('x')[0]) * 32
    tile_size_y = int(TILE_CONFIGS.get(thread_tile_sel, "1x1").split('x')[0])
    cache_line_size = 8
    stride = 2048

    # The kernel is A[:] = ( A[:] + B[:] ) * scale

    def tile_reuse_a():
        if a_layout == 0:
            unit = tile_size_x * tile_size_y - tile_size_y
            non_unit = tile_size_y * stride
            return (unit + non_unit) / (tile_size_y * tile_size_x)
        else:
            unit = tile_size_x * tile_size_y - tile_size_x
            non_unit = tile_size_x * stride
            return (unit + non_unit) / (tile_size_y * tile_size_x)

    def tile_reuse_b():
        if b_layout == 0:
            unit = tile_size_x * tile_size_y - tile_size_y
            non_unit = tile_size_y * stride
            return (unit + non_unit) / (tile_size_y * tile_size_x)
        else:
            unit = tile_size_x * tile_size_y - tile_size_x
            non_unit = tile_size_x * stride
            return (unit + non_unit) / (tile_size_y * tile_size_x)

    if kernel == 0:  # i-outer, j-inner: for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
        # A access pattern: A[i,k] - iterates over k in inner-ish loop
        # B access pattern: B[k,j] - iterates over j in inner loop, k in middle
        
        # A reuse: same A[i,k] reused across j loop (N times), distance ~ N (size of j loop)
        # B reuse: B[k,j] - j changes every iteration, k changes every N iterations
        
        if a_layout == 0:  # A row-major: A[i,k] and A[i,k+1] are adjacent
            a_reuse = 1
        else:  # A col-major: A[i,k] and A[i+1,k] are adjacent, but we iterate over k
            a_reuse = (stride/16) - 1
        
        if b_layout == 0:  # B row-major: B[k,j] and B[k,j+1] adjacent, j is innermost
            b_reuse = 1
        else:  # B col-major: B[k,j] and B[k+1,j] adjacent
            b_reuse = (stride/16) - 1  # stride-N access

        avg_reuse = ((2*a_reuse) + b_reuse) / 3
    elif kernel == 1:  # j-outer, i-inner: for j: for i: for k: C[i,j] += A[i,k] * B[k,j]
        # A access pattern: A[i,k] - i changes in inner-ish loop
        # B access pattern: B[k,j] - j is outermost, reused across i loop
        
        if a_layout == 0:  # A row-major: stride-1 over k, stride-N over i
            a_reuse = (stride/16) - 1  # i changes, causing stride-N jumps
        else:  # A col-major: A[i,k] and A[i+1,k] adjacent, i is inner
            a_reuse = 1  # good spatial locality
        
        if b_layout == 0:  # B row-major: B[k,j] and B[k,j+1] adjacent, but j is outer
            b_reuse = (stride/16) - 1  # poor reuse
        else:  # B col-major: B[k,j] and B[k+1,j] adjacent
            b_reuse = 1  # reuse across i-loop
        
        avg_reuse = ((2*a_reuse) + b_reuse) / 3
        
    elif kernel == 2:  # tiled, follows schedule of 0, but should be irrelevant
        # Tiling improves reuse by keeping tile in cache
        # Reuse distance bounded by tile working set
        # Regardless of the order, 1 access is strided other access not strided
        reuse_a = tile_reuse_a()
        reuse_b = tile_reuse_b()
        avg_reuse = (2*reuse_a + reuse_b) / 3
    elif kernel == 3:  # tiled+copy, follows schedule of 0
        # Copy to contiguous buffer eliminates layout penalty
        # Reuse distance mainly depends on tile size
        reuse_a = tile_reuse_a()
        reuse_b = tile_reuse_b()
        avg_reuse = (2*reuse_a + reuse_b) / 3
    else:
        avg_reuse = N * N  # unknown, assume worst

    return avg_reuse

def compute_reuse_distance_w_penalty(row, N=2048):
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
    thread_tile_sel = row['thread_tile_sel']

    tile_size = int(TILE_CONFIGS.get(thread_tile_sel, "4x4").split('x')[0])
    cache_line_size = 16
    stride = 2048

    # The kernel is A[:] = ( A[:] + B[:] ) * scale

    def tile_reuse():
        if tile_size < cache_line_size:
            non_unit = (tile_size - 1) * ((stride + cache_line_size) // cache_line_size) + 1
            unit = (tile_size * tile_size - tile_size + 1)

            # penalty for tile_size
            # 3 tiles therefore divide L1 in 3
            penalty = max(1, ( tile_size * tile_size ) // (48 * 1024 / (16 * 3)))
            return penalty * (non_unit + unit) / (tile_size * tile_size)
        else:
            non_unit = (tile_size - 1) * ((stride + cache_line_size) // cache_line_size)
            unit = (tile_size * tile_size - tile_size)

            # penalty for tile_size
            # take half of L1?
            # 3 tiles therefore divide L1 in 3
            penalty = max(1, ( tile_size * tile_size ) // (16 * 1024 / (16 * 3)))
            return penalty * (non_unit + unit) / (tile_size * tile_size)


    if kernel == 0:  # i-outer, j-inner: for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
        # A access pattern: A[i,k] - iterates over k in inner-ish loop
        # B access pattern: B[k,j] - iterates over j in inner loop, k in middle
        
        # A reuse: same A[i,k] reused across j loop (N times), distance ~ N (size of j loop)
        # B reuse: B[k,j] - j changes every iteration, k changes every N iterations
        
        if a_layout == 0:  # A row-major: A[i,k] and A[i,k+1] are adjacent
            a_reuse = 1
        else:  # A col-major: A[i,k] and A[i+1,k] are adjacent, but we iterate over k
            a_reuse = (stride/16) - 1
        
        if b_layout == 0:  # B row-major: B[k,j] and B[k,j+1] adjacent, j is innermost
            b_reuse = 1
        else:  # B col-major: B[k,j] and B[k+1,j] adjacent
            b_reuse = (stride/16) - 1  # stride-N access

        avg_reuse = ((2*a_reuse) + b_reuse) / 3

    elif kernel == 1:  # j-outer, i-inner: for j: for i: for k: C[i,j] += A[i,k] * B[k,j]
        # A access pattern: A[i,k] - i changes in inner-ish loop
        # B access pattern: B[k,j] - j is outermost, reused across i loop
        
        if a_layout == 0:  # A row-major: stride-1 over k, stride-N over i
            a_reuse = (stride/16) - 1  # i changes, causing stride-N jumps
        else:  # A col-major: A[i,k] and A[i+1,k] adjacent, i is inner
            a_reuse = 1  # good spatial locality
        
        if b_layout == 0:  # B row-major: B[k,j] and B[k,j+1] adjacent, but j is outer
            b_reuse = (stride/16) - 1  # poor reuse
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
    thread_tile_sel = row['thread_tile_sel']

    N = 2048

    tile_size = int(TILE_CONFIGS.get(thread_tile_sel, "4x4").split('x')[0])
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
        a_dim = [1, 32]
        b_dim = [1, 32]
    elif kernel == 1:  # j-outer, i-inner
        a_stride = [1, stride] if a_layout == 0 else [stride, 1]
        b_stride = [1, stride] if b_layout == 0 else [stride, 1]
        a_dim = [1, 32]
        b_dim = [1, 32]
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
    thread_tile_sel = row['thread_tile_sel']

    tile_size = int(TILE_CONFIGS.get(thread_tile_sel, "4x4").split('x')[0])
    """Compute average stride between consecutive accesses."""
    if kernel == 0:  # i-outer, j-inner
        av = 1 if a_layout == 0 else 2048
        bv = 1 if b_layout == 0 else 2048
    elif kernel == 1:  # j-outer, i-inner
        av = 1 if a_layout == 1 else 2048
        bv = 1 if b_layout == 1 else 2048
    elif kernel in [2, 3]:  # tiled or tiled+copy
        if tile_size == 1:
            tile_size += 1
        av = ((tile_size * tile_size - tile_size) * 1 + (tile_size) * 2048) / ((tile_size - 1) ** 2)
        bv = ((tile_size * tile_size - tile_size) * 1 + (tile_size) * 2048) / ((tile_size - 1) ** 2)
    else:
        return 2048 * 2048

    return (2*av + bv)/3


def compute_cache_line_utilization(row):
    """
    Compute cache line utilization.
    
    Cache line = 8 elements (64 bytes for 8-byte doubles).
    If stride is 1 (unit stride), we use all 8 elements in a cache line -> utilization = 1.0 (100%)
    If stride > 1, we only use 1 element per cache line loaded -> utilization = 1/8 = 0.125 (12.5%)
    
    Returns average cache line utilization across A and B accesses.
    """
    kernel = row['kernel']
    a_layout = row['a_layout']  # 0=row-major, 1=col-major
    b_layout = row['b_layout']
    thread_tile_sel = row['thread_tile_sel']
    
    cache_line_elements = 8
    N = 2048
    tile_size = int(TILE_CONFIGS.get(thread_tile_sel, "4x4").split('x')[0])
    
    def get_utilization_for_stride(stride):
        """Given a stride, compute cache line utilization."""
        if stride == 1:
            return 1.0  # 100% - all elements in cache line are used
        elif stride < cache_line_elements:
            # Partial utilization: we use cache_line_elements / stride elements
            return 1.0 / stride
        else:
            # Stride >= cache line size: only 1 element used per cache line
            return 1.0 / cache_line_elements
    
    if kernel == 0:  # i-outer, j-inner: for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
        # A[i,k]: k is inner-ish dimension
        # Row-major A: A[i,k] to A[i,k+1] is stride 1
        # Col-major A: A[i,k] to A[i,k+1] is stride N
        a_stride = 1 if a_layout == 0 else N
        
        # B[k,j]: j is innermost
        # Row-major B: B[k,j] to B[k,j+1] is stride 1
        # Col-major B: B[k,j] to B[k,j+1] is stride N
        b_stride = 1 if b_layout == 0 else N
        
    elif kernel == 1:  # j-outer, i-inner: for j: for i: for k: C[i,j] += A[i,k] * B[k,j]
        # A[i,k]: i changes in middle loop, k in inner loop
        # Row-major A: accessing along k (stride 1) then jumping rows (stride N)
        # Col-major A: A[i,k] to A[i+1,k] is stride 1
        a_stride = 1 if a_layout == 1 else N
        
        # B[k,j]: j is outermost, k is inner
        # Row-major B: B[k,j] to B[k+1,j] is stride N
        # Col-major B: B[k,j] to B[k+1,j] is stride 1
        b_stride = 1 if b_layout == 1 else N
        
    elif kernel == 2:  # tiled (no copy)
        # Within a tile, we have mixed access patterns
        # For row-major: horizontal traversal is stride 1, vertical is stride N
        # For col-major: vertical traversal is stride 1, horizontal is stride N
        # Average utilization depends on tile traversal pattern
        
        # Assuming tile traverses row-by-row (i inner, j outer within tile):
        # Row-major: mostly stride-1 access within rows, stride-N between rows
        # Col-major: stride-N within rows, stride-1 between rows
        
        # Fraction of accesses that are stride-1 vs stride-N in a tile:
        # For a tile_size x tile_size tile traversed row-by-row:
        # - (tile_size - 1) stride-1 accesses per row, tile_size rows
        # - (tile_size - 1) stride-N accesses between rows
        # Total: tile_size * (tile_size - 1) stride-1 + (tile_size - 1) stride-N
        
        if tile_size > 1:
            stride1_accesses = tile_size * (tile_size - 1)  # within rows
            strideN_accesses = tile_size - 1  # between rows
            total_accesses = stride1_accesses + strideN_accesses
            
            # For row-major: stride-1 is good, stride-N is bad
            # For col-major: stride-1 is bad (it's stride-N), stride-N is good (it's stride-1)
            if a_layout == 0:  # row-major
                a_util = (stride1_accesses * 1.0 + strideN_accesses * (1.0/cache_line_elements)) / total_accesses
            else:  # col-major
                a_util = (stride1_accesses * (1.0/cache_line_elements) + strideN_accesses * 1.0) / total_accesses
            
            if b_layout == 0:
                b_util = (stride1_accesses * 1.0 + strideN_accesses * (1.0/cache_line_elements)) / total_accesses
            else:
                b_util = (stride1_accesses * (1.0/cache_line_elements) + strideN_accesses * 1.0) / total_accesses
        else:
            a_util = 1.0 / cache_line_elements
            b_util = 1.0 / cache_line_elements
        
        return (2 * a_util + b_util) / 3
        
    elif kernel == 3:  # tiled+copy
        # Copy to contiguous buffer means all accesses are stride-1
        # Perfect cache line utilization
        return 1.0
    else:
        return 1.0 / cache_line_elements  # assume worst
    
    a_util = get_utilization_for_stride(a_stride)
    b_util = get_utilization_for_stride(b_stride)
    
    return (2 * a_util + b_util) / 3


def compute_page_utilization(row):
    """
    Compute page utilization metrics.
    
    Page size = 4KB = 512 elements (for 8-byte doubles).
    
    Returns tuple of:
    - pages_loaded: Number of unique pages touched due to stride pattern
    - pages_needed: Minimum pages needed if data were contiguous
    - page_utilization: pages_needed / pages_loaded (higher is better)
    
    For strided access, we touch more pages than necessary.
    """
    kernel = row['kernel']
    a_layout = row['a_layout']
    b_layout = row['b_layout']
    thread_tile_sel = row['thread_tile_sel']
    
    page_size_bytes = 4096
    element_size = 8  # 8 bytes per double
    elements_per_page = page_size_bytes // element_size  # 512 elements
    N = 2048
    tile_size = int(TILE_CONFIGS.get(thread_tile_sel, "4x4").split('x')[0])
    
    def compute_pages_for_access(num_elements, stride, matrix_stride=N):
        """
        Compute pages loaded vs pages needed for an access pattern.
        
        num_elements: how many elements we access
        stride: stride between consecutive accesses
        matrix_stride: stride of the matrix (N for NxN matrix)
        """
        if num_elements == 0:
            return 0, 0, 1.0
        
        # Minimum pages needed (if data were contiguous)
        pages_needed = (num_elements + elements_per_page - 1) // elements_per_page
        
        if stride == 1:
            # Contiguous access - we load exactly the pages we need
            pages_loaded = pages_needed
        else:
            # Strided access - we may touch more pages
            # Address span = (num_elements - 1) * stride
            address_span = (num_elements - 1) * stride
            # Number of pages spanned
            pages_loaded = (address_span // elements_per_page) + 1
        
        utilization = pages_needed / pages_loaded if pages_loaded > 0 else 1.0
        return pages_loaded, pages_needed, utilization
    
    # Determine access patterns based on kernel and layout
    if kernel == 0:  # i-outer, j-inner
        # A[i,k]: access N elements along k for each i
        # B[k,j]: access N elements along j for each k
        a_stride = 1 if a_layout == 0 else N
        b_stride = 1 if b_layout == 0 else N
        # Per iteration of outer loops, we access N elements
        a_elements = N
        b_elements = N
        
    elif kernel == 1:  # j-outer, i-inner
        a_stride = 1 if a_layout == 1 else N
        b_stride = 1 if b_layout == 1 else N
        a_elements = N
        b_elements = N
        
    elif kernel == 2:  # tiled (no copy)
        # Within a tile: access tile_size x tile_size elements
        # For row-major: rows are contiguous (stride 1), columns are stride N
        # For col-major: columns are contiguous (stride 1), rows are stride N
        
        # For a tile, we need to consider 2D access pattern
        # Approximate: we access tile_size rows/cols, each of tile_size elements
        
        # Row-major A: tile spans tile_size rows, each row has tile_size contiguous elements
        # But rows are N elements apart
        if a_layout == 0:  # row-major
            # tile_size rows, each row is N elements apart
            # Within each row, tile_size contiguous elements
            a_pages_per_row = (tile_size + elements_per_page - 1) // elements_per_page
            # Address span across rows
            a_row_span = (tile_size - 1) * N + tile_size
            a_pages_loaded = (a_row_span + elements_per_page - 1) // elements_per_page
        else:  # col-major
            # tile_size columns, each column is N elements apart in logical indexing
            # but in memory, column elements are contiguous
            a_pages_per_col = (tile_size + elements_per_page - 1) // elements_per_page
            a_col_span = (tile_size - 1) * N + tile_size
            a_pages_loaded = (a_col_span + elements_per_page - 1) // elements_per_page
        
        if b_layout == 0:
            b_row_span = (tile_size - 1) * N + tile_size
            b_pages_loaded = (b_row_span + elements_per_page - 1) // elements_per_page
        else:
            b_col_span = (tile_size - 1) * N + tile_size
            b_pages_loaded = (b_col_span + elements_per_page - 1) // elements_per_page
        
        a_pages_needed = (tile_size * tile_size + elements_per_page - 1) // elements_per_page
        b_pages_needed = (tile_size * tile_size + elements_per_page - 1) // elements_per_page
        
        a_util = a_pages_needed / a_pages_loaded if a_pages_loaded > 0 else 1.0
        b_util = b_pages_needed / b_pages_loaded if b_pages_loaded > 0 else 1.0
        
        avg_pages_loaded = (2 * a_pages_loaded + b_pages_loaded) / 3
        avg_pages_needed = (2 * a_pages_needed + b_pages_needed) / 3
        avg_util = (2 * a_util + b_util) / 3
        
        return avg_pages_loaded, avg_pages_needed, avg_util
        
    elif kernel == 3:  # tiled+copy
        # Copy to contiguous buffer - perfect page utilization
        elements_per_tile = tile_size * tile_size
        pages_needed = (elements_per_tile + elements_per_page - 1) // elements_per_page
        return float(pages_needed), float(pages_needed), 1.0
    else:
        return float(N), 1.0, 1.0 / N
    
    a_loaded, a_needed, a_util = compute_pages_for_access(a_elements, a_stride, N)
    b_loaded, b_needed, b_util = compute_pages_for_access(b_elements, b_stride, N)
    
    avg_pages_loaded = (2 * a_loaded + b_loaded) / 3
    avg_pages_needed = (2 * a_needed + b_needed) / 3
    avg_util = (2 * a_util + b_util) / 3
    
    return avg_pages_loaded, avg_pages_needed, avg_util


def compute_effective_locality(row):
    """
    Combines spatial locality (stride) with temporal locality (cache fit).
    Lower is better.
    
    This metric penalizes large tiles that exceed cache capacity,
    even if their average stride is low.
    """
    import math
    
    kernel = row['kernel']
    thread_tile_sel = row['thread_tile_sel']
    tile_size = int(TILE_CONFIGS.get(thread_tile_sel, "4x4").split('x')[0])
    
    # Get base stride metric
    avg_stride = compute_average_stride(row)
    
    # For non-tiled kernels, working set is effectively the whole matrix row/column
    # For tiled kernels, working set is the tiles
    if kernel in [0, 1]:
        # Non-tiled: working set is essentially unbounded (whole rows/cols)
        # The stride already captures the penalty
        working_set_bytes = 32 * 8  # One row/column, 1 line at a time
    else:
        # Tiled: working set is 3 tiles (A tile, B tile, C tile)
        working_set_bytes = 3 * tile_size * tile_size * 32 * 8

    # How many cache lines needed
    num_complete_caches = (working_set_bytes + 32 * 1024 - 1) // (32 * 1024)
    
    # Combined metric: stride * cache penalties
    effective_locality = avg_stride * num_complete_caches
    
    return effective_locality


def main():
    # Parse data
    if len(sys.argv) < 2:
        print("Usage: python analyze_perf.py <csv_file>, default=results_cpu_v1/performance.csv")
        csv_file = "results_cpu_v1/performance.csv"
    else:
        csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    # Add human-readable columns
    df['kernel_name'] = df['kernel'].map(KERNELS)
    df['layout'] = df.apply(lambda r: LAYOUTS[(r['a_layout'], r['b_layout'])], axis=1)
    df['tile_size'] = df['thread_tile_sel'].map(TILE_CONFIGS)

    # Create description column
    def make_description(row):
        desc = f"{row['kernel_name']} | {row['layout']}"
        if row['kernel'] in [2, 3]:  # tiled or tiled+copy kernel
            desc += f" | tile={row['tile_size']}"
        return desc

    df['description'] = df.apply(make_description, axis=1)
    
    # Compute metrics
    df['reuse_distance'] = df.apply(compute_reuse_distance, axis=1)
    df['reuse_distance_w_penalty'] = df.apply(compute_reuse_distance_w_penalty, axis=1)
    closeness_span = df.apply(compute_tile_closeness, axis=1)
    df['tile_closeness'] = closeness_span.apply(lambda x: x[0])
    df['address_span'] = closeness_span.apply(lambda x: x[1])
    df['avg_stride'] = df.apply(compute_average_stride, axis=1)
    
    # Combined memory efficiency metric: cache_line_util * page_util
    df['cache_line_util'] = df.apply(compute_cache_line_utilization, axis=1)
    page_metrics = df.apply(compute_page_utilization, axis=1)
    df['page_util'] = page_metrics.apply(lambda x: x[2])
    df['mem_efficiency'] = df['cache_line_util'] * df['page_util']
    df.drop(columns=['cache_line_util', 'page_util'], inplace=True)
    
    # Effective locality: combines stride with cache working set penalty
    df['eff_locality'] = df.apply(compute_effective_locality, axis=1)

    # Define datasets to analyze (exclude tiled+copy kernel=3 from main analysis)
    datasets = [
        ("ALL KERNELS (excl. tiled+copy)", df[df['kernel'] != 2].copy()),
        ("NON-TILED KERNELS ONLY", df[df['kernel'].isin([0, 1])].copy()),
    ]

    for title, data in datasets:
        # Sort by mean runtime
        df_sorted = data.sort_values('mean_ms').reset_index(drop=True)
        df_sorted['runtime_rank'] = df_sorted.index + 1
        
        # Compute ranks (lower = better for most metrics, higher = better for utilization)
        df_sorted['reuse_rank'] = df_sorted['reuse_distance'].rank(method='min').astype(int)
        df_sorted['reuse_w_penalty_rank'] = df_sorted['reuse_distance_w_penalty'].rank(method='min').astype(int)
        df_sorted['closeness_rank'] = df_sorted['tile_closeness'].rank(method='min').astype(int)
        df_sorted['span_rank'] = df_sorted['address_span'].rank(method='min').astype(int)
        df_sorted['stride_rank'] = df_sorted['avg_stride'].rank(method='min').astype(int)
        # For efficiency, higher is better, so we rank in descending order
        df_sorted['mem_eff_rank'] = df_sorted['mem_efficiency'].rank(method='min', ascending=False).astype(int)
        # For effective locality, lower is better
        df_sorted['eff_loc_rank'] = df_sorted['eff_locality'].rank(method='min').astype(int)

        # Print results - FULL TABLE
        print("=" * 220)
        print(f"RUNTIME ORDERING (fastest to slowest) - {title}")
        print("=" * 220)
        print()
        header = (f"{'Rank':<5} {'Runtime':<22} {'Description':<45} "
                  f"{'ReuseDist':<10} {'ReuseRk':<8} "
                  f"{'ReuseDistWP':<10} {'ReuseWPRk':<8} "
                  f"{'Closeness':<12} {'CloseRk':<8} "
                  f"{'AddrSpan':<12} {'SpanRk':<8} "
                  f"{'AvgStride':<10} {'StrideRk':<8} "
                  f"{'MemEff':<10} {'MemEffRk':<8}")
        print(header)
        print("-" * 220)

        for _, row in df_sorted.iterrows():
            runtime_str = f"{row['mean_ms']:8.3f} ms (±{row['stdev_ms']:6.3f})"
            print(f"{row['runtime_rank']:<5} {runtime_str:<22} {row['description']:<45} "
                  f"{row['reuse_distance']:<10.1f} {row['reuse_rank']:<8} "
                  f"{row['reuse_distance_w_penalty']:<10.1f} {row['reuse_w_penalty_rank']:<8} "
                  f"{row['tile_closeness']:<12.1f} {row['closeness_rank']:<8} "
                  f"{row['address_span']:<12.1f} {row['span_rank']:<8} "
                  f"{row['avg_stride']:<10.1f} {row['stride_rank']:<8} "
                  f"{row['mem_efficiency']:<10.4f} {row['mem_eff_rank']:<8}")

        # Print results - SIMPLIFIED TABLE with Effective Locality
        print()
        print("=" * 150)
        print(f"SIMPLIFIED VIEW WITH EFFECTIVE LOCALITY - {title}")
        print("=" * 150)
        print()
        header2 = (f"{'Rank':<5} {'Runtime':<22} {'Description':<45} "
                   f"{'ReuseDist':<10} {'ReuseRk':<8} "
                   f"{'ReuseDistWP':<10} {'ReuseWPRk':<8} "
                   f"{'AvgStride':<10} {'StrideRk':<8} "
                   f"{'EffLocality':<12} {'EffLocRk':<8}")
        print(header2)
        print("-" * 150)

        for _, row in df_sorted.iterrows():
            runtime_str = f"{row['mean_ms']:8.3f} ms (±{row['stdev_ms']:6.3f})"
            print(f"{row['runtime_rank']:<5} {runtime_str:<22} {row['description']:<45} "
                  f"{row['reuse_distance']:<10.1f} {row['reuse_rank']:<8} "
                  f"{row['reuse_distance_w_penalty']:<10.1f} {row['reuse_w_penalty_rank']:<8} "
                  f"{row['avg_stride']:<10.1f} {row['stride_rank']:<8} "
                  f"{row['eff_locality']:<12.1f} {row['eff_loc_rank']:<8}")

        print()

        # All rank columns to evaluate
        rank_columns = [
            'reuse_rank',
            'reuse_w_penalty_rank', 
            'closeness_rank',
            'span_rank',
            'stride_rank',
            'mem_eff_rank',
            #'eff_loc_rank'
        ]

        runtime_rank = df_sorted['runtime_rank'].values

        print("=" * 90)
        print("RANK CORRELATION ANALYSIS: Metric Rank vs Runtime Rank")
        print("=" * 90)
        print()

        results = []

        for col in rank_columns:
            metric_rank = df_sorted[col].values
            
            # Spearman correlation
            spearman_rho, _ = spearmanr(runtime_rank, metric_rank)
            
            # Kendall tau
            kendall_tau, _ = kendalltau(runtime_rank, metric_rank)
            
            # Mean Absolute Rank Error
            mare = np.mean(np.abs(runtime_rank - metric_rank))
            
            # Root Mean Square Rank Error
            rmsre = np.sqrt(np.mean((runtime_rank - metric_rank) ** 2))
            
            # Top-5 accuracy
            top5_runtime = set(df_sorted.nsmallest(5, 'mean_ms').index)
            top5_metric = set(df_sorted.nsmallest(5, col).index)
            top5_overlap = len(top5_runtime & top5_metric)
            
            # Top-10 accuracy
            top10_runtime = set(df_sorted.nsmallest(10, 'mean_ms').index)
            top10_metric = set(df_sorted.nsmallest(10, col).index)
            top10_overlap = len(top10_runtime & top10_metric)
            
            results.append({
                'Metric': col.replace('_rank', ''),
                'Spearman ρ': spearman_rho,
                'Kendall τ': kendall_tau,
                'MARE': mare,
                'RMSRE': rmsre,
                'Top-5': top5_overlap,
                'Top-10': top10_overlap
            })

        # Summary table (sorted by Spearman descending)
        print(f"{'Metric':<22} {'Spearman':>10} {'Kendall':>10} {'MARE':>8} {'RMSRE':>8} {'Top-5':>7} {'Top-10':>8}")
        print("-" * 90)
        for r in sorted(results, key=lambda x: -x['Spearman ρ']):
            print(f"{r['Metric']:<22} {r['Spearman ρ']:>+10.4f} {r['Kendall τ']:>+10.4f} {r['MARE']:>8.2f} {r['RMSRE']:>8.2f} {r['Top-5']:>5}/5 {r['Top-10']:>6}/10")

        print()
        print("=" * 90)
        print("INTERPRETATION: Spearman/Kendall closer to +1.0 = better | MARE/RMSRE closer to 0 = better")
        print("=" * 90)


if __name__ == "__main__":
    main()