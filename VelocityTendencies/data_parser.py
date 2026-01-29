#!/usr/bin/env python3
"""
Simple ICON Data File Parser

Reads ICON model data files in Fortran dump format.

Usage:
    from icon_parser import read_array
    
    # Read array by name and timestep
    vn_data = read_array('vn', timestep=1, data_dir='./data')
    
    # Read connectivity data from structured files
    cell_idx = read_array('edges_cell_idx', timestep=1, data_dir='./data')
"""

import numpy as np
from pathlib import Path


def read_array(array_name, timestep, data_dir='.'):
    """
    Read array data for a given timestep.
    
    First tries: <array_name>.t0.<timestep>.data
    Then searches: p_patch, p_prog, p_metrics files
    
    Args:
        array_name: Name like 'vn', 'edges_cell_idx', 'nblks_c'
        timestep: Timestep number
        data_dir: Directory with data files
    
    Returns:
        numpy array or scalar value
    """
    data_dir = Path(data_dir)

    # Try direct file: array_name.t0.<timestep>.data
    direct_file = data_dir / f"{array_name}.t0.{timestep}.data"
    if direct_file.exists():
        return _parse_simple_file_for_array(direct_file)
    direct_file2 = data_dir / f"{array_name}.{timestep}.data"
    if direct_file2.exists():
        return _parse_simple_file_for_array(direct_file2)

    # Search structured files: p_patch, p_prog, p_metrics
    for base_name in ['p_patch', 'p_prog', 'p_metrics']:
        # Try with timestep
        for fp in [f"{base_name}.t0.{timestep}.data", f"{base_name}.{timestep}.data"]:
            filepath = data_dir / fp
            if not filepath.exists():
                # Try without timestep
                filepath = data_dir / f"{base_name}.data"
            
            if filepath.exists():
                result = _parse_structured_file_for_array(filepath)

                # Look for exact match or suffix match
                if array_name in result:
                    return result[array_name]
                
                for key in result.keys():
                    if key.endswith('_' + array_name):
                        return result[key]
    
    raise FileNotFoundError(f"Array '{array_name}' not found at timestep {timestep}")


def read_scalar(array_name, timestep, data_dir='.'):
    data_dir = Path(data_dir)
    
    # Try direct file: array_name.t0.timestep.data
    direct_file = data_dir / f"{array_name}.t0.{timestep}.data"
    if direct_file.exists():
        return _parse_simple_file_for_scalar(direct_file)
    direct_file2 = data_dir / f"{array_name}.{timestep}.data"
    if direct_file2.exists():
        return _parse_simple_file_for_scalar(direct_file2)

    # Search structured files: p_patch, p_prog, p_metrics
    for base_name in ['p_patch', 'p_prog', 'p_metrics']:
        # Try with timestep
        for fp in [f"{base_name}.t0.{timestep}.data", f"{base_name}.{timestep}.data"]:
            filepath = data_dir / fp
            if not filepath.exists():
                # Try without timestep
                filepath = data_dir / f"{base_name}.data"
            
            if filepath.exists():
                result = _parse_structured_file_for_scalar(filepath)

                # Look for exact match or suffix match
                if array_name in result:
                    return result[array_name]
                
                for key in result.keys():
                    if key.endswith('_' + array_name):
                        return result[key]
    
    raise FileNotFoundError(f"Array '{array_name}' not found at timestep {timestep}")


def _parse_simple_file_for_array(filepath):
    """Parse simple array file format."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    result = {}
    i = 0
    
    while i < len(lines):
        if lines[i].startswith('#'):
            section = lines[i][1:].strip()
            
            if section == 'rank':
                i += 1
                result['rank'] = int(lines[i])
                i += 1
                
            elif section == 'size':
                size = []
                for _ in range(result['rank']):
                    i += 1
                    size.append(int(lines[i]))
                result['size'] = tuple(size)
                i += 1
                
            elif section == 'lbound':
                for _ in range(result['rank']):
                    i += 1
                i += 1
                
            elif section == 'entries':
                i += 1
                data = []
                while i < len(lines) and not lines[i].startswith('#'):
                    data.append(float(lines[i]))
                    i += 1
                
                # Reshape
                expected = np.prod(result['size'])
                if len(data) < expected:
                    full = np.zeros(expected)
                    full[:len(data)] = data
                    data = full
                
                result['data'] = np.array(data).reshape(result['size'])
            else:
                i += 1
        else:
            i += 1

    return result['data']

def _parse_number(x):
    try:
        i = int(x)
        # Ensure things like "3.0" don't become int
        if str(i) == x or str(i) == x.strip():
            return i
    except ValueError:
        pass
    return float(x)

def _parse_simple_file_for_scalar(filepath):
    """Parse simple scalar file format."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    result = {}

    assert len(lines) == 1
    result['data'] = _parse_number(lines[0])

    return result['data']


def _parse_structured_file_for_array(filepath):
    """Parse structured file with multiple arrays."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    result = {}
    context = []
    i = 0
    n = len(lines)

    def parse_scalar(val):
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    while i < n:
        line = lines[i]

        if not line.startswith('#'):
            i += 1
            continue

        section = line[1:].strip()

        # ---- Scalar value ----
        if i + 1 < n and not lines[i + 1].startswith('#'):
            result[section] = parse_scalar(lines[i + 1])
            i += 2
            continue

        # ---- Array ----
        if i + 1 < n and (lines[i + 1] == '# assoc' or lines[i + 1] == '# alloc'):
            array_data = {}
            i += 2              # move to assoc value
            i += 1              # skip assoc value → next tag

            # Optional #missing
            if i < n and lines[i] == '# missing':
                i += 2          # skip '# missing' and its value

            # #rank
            if i < n and lines[i] == '# rank':
                i += 1
                array_data['rank'] = int(lines[i])
                i += 1
            else:
                raise ValueError("Expected # rank")

            # #size
            if i < n and lines[i] == '# size':
                size = []
                for _ in range(array_data['rank']):
                    i += 1
                    size.append(int(lines[i]))
                array_data['size'] = tuple(size)
                i += 1
            else:
                raise ValueError("Expected # size")

            # Optional #lbound
            if i < n and lines[i] == '# lbound':
                for _ in range(array_data['rank']):
                    i += 1
                i += 1

            # #entries
            if i < n and lines[i] == '# entries':
                i += 1
                data = []
                while i < n and not lines[i].startswith('#'):
                    data.append(float(lines[i]))
                    i += 1

                expected = int(np.prod(array_data['size']))
                if len(data) < expected:
                    full = np.zeros(expected)
                    full[:len(data)] = data
                    data = full

                array_data['data'] = np.array(data).reshape(array_data['size'])
            else:
                raise ValueError("Expected # entries")

            # Build key
            key = '_'.join(context + [section]) if context else section
            result[key] = array_data['data']
            context.clear()
            continue

        # ---- Context marker ----
        context.append(section)
        i += 1

    return result

def _parse_structured_file_for_scalar(filepath):
    """Parse structured file with multiple arrays."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    result = {}
    i = 0
    n = len(lines)

    def parse_scalar(val):
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    while i < n:
        line = lines[i]

        if not line.startswith('#'):
            i += 1
            continue

        section = line[1:].strip()

        # ---- Scalar value ----
        if i + 1 < n and not lines[i + 1].startswith('#'):
            result[section] = parse_scalar(lines[i + 1])
            i += 2
            continue

        i += 1

    return result


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Get data directory
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/primrose/Work/icon-artifacts/velocity/data_nproma20480/'
    
    print("=" * 70)
    print("ICON DATA PARSER - USAGE EXAMPLE")
    print("=" * 70)
    print()

    print("Example 0: Read istep scalar")
    print("-" * 70)
    try:
        istep = read_scalar('istep', timestep=1, data_dir=data_dir)
        print(f"✓ Read istep scalar")
        print(f"  Value: {istep}")
        print(f"  Type:  {type(istep).__name__}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
    print()

    # Example 1: Read a direct array file
    print("Example 1: Read velocity array")
    print("-" * 70)
    try:
        vn = read_array('vn', timestep=1, data_dir=data_dir)
        print(f"  Type:  {vn.dtype}")
        print(f"  Shape: {vn.shape}")
        print(f"  Type:  {vn.dtype}")
        print(f"  First 5 entries:")
        print(f"    {vn.ravel()[:5]}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
    print()

    # Example 2: Read scalar from structured file
    print("Example 2: Read scalar value")
    print("-" * 70)
    try:
        nblks_c = read_scalar('nblks_c', timestep=1, data_dir=data_dir)
        print(f"✓ Read nblks_c")
        print(f"  Value: {nblks_c}")
        print(f"  Type:  {type(nblks_c).__name__}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
    print()

    # Example 2: Read scalar from structured file
    print("Example 2.2: Read scalar value")
    print("-" * 70)
    try:
        nblks_e = read_scalar('nblks_e', timestep=1, data_dir=data_dir)
        print(f"✓ Read nblks_e")
        print(f"  Value: {nblks_e}")
        print(f"  Type:  {type(nblks_e).__name__}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
    print()
    
    # Example 3: Read connectivity array
    print("Example 3: Read connectivity array")
    print("-" * 70)
    try:
        cell_idx = read_array('edges_cell_idx', timestep=1, data_dir=data_dir)
        print(f"✓ Read edges_cell_idx")
        print(f"  Shape: {cell_idx.shape}")
        print(f"  Type:  {cell_idx.dtype}")
        print(f"  First 5 entries:")
        print(f"    {cell_idx.ravel()[:5]}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
    print()

    print("Example 4: Z kin hor e")
    print("-" * 70)
    try:
        z_kin_hor_e = read_array('z_kin_hor_e', timestep=1, data_dir=data_dir)
        print(f"✓ Read z_kin_hor_e")
        print(f"  Shape: {z_kin_hor_e.shape}")
        print(f"  Type:  {z_kin_hor_e.dtype}")
        print(f"  First 5 entries:")
        print(f"    {z_kin_hor_e.ravel()[:5]}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
    print()
