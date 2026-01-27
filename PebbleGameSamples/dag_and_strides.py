import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Array dimensions
I = 4  # rows
J = 5  # columns

print("=" * 60)
print("SIMPLE DATA DEPENDENCY EXAMPLES")
print("=" * 60)

# Initialize array with boundary conditions
A = np.arange(I * J).reshape(I, J).astype(float)

print("\n" + "=" * 60)
print("VERSION 1: i outer loop, j inner loop")
print("Dependency: A[i,j] = A[i,j] + A[i,j-1]  (LEFT neighbor)")
print("j is unit stride (contiguous in memory)")
print("=" * 60)

A_v1 = A.copy()
print("\nInitial array:")
print(A_v1)
print()

for i in range(I):
    for j in range(1, J):  # j starts at 1 (depends on j-1)
        A_v1[i, j] = A_v1[i, j] + A_v1[i, j-1]
        print(f"A[{i},{j}] = A[{i},{j}] + A[{i},{j-1}]")

print("\nResult (Version 1):")
print(A_v1)

print("\n" + "=" * 60)
print("VERSION 2: j outer loop, i inner loop")
print("Dependency: A[i,j] = A[i,j] + A[i-1,j]  (BOTTOM neighbor)")
print("i is NOT unit stride (strided access)")
print("=" * 60)

A_v2 = A.copy()
print("\nInitial array:")
print(A_v2)
print()

for j in range(J):
    for i in range(1, I):  # i starts at 1 (depends on i-1)
        A_v2[i, j] = A_v2[i, j] + A_v2[i-1, j]
        print(f"A[{i},{j}] = A[{i},{j}] + A[{i-1},{j}]")

print("\nResult (Version 2):")
print(A_v2)

print("\n" + "=" * 60)
print("DATA DEPENDENCY GRAPHS")
print("=" * 60)

# Create both dependency DAGs
def create_dag_v1():
    """Version 1: depends on left neighbor (j-1)"""
    G = nx.DiGraph()
    
    for i in range(I):
        for j in range(J):
            node_id = f"({i},{j})"
            G.add_node(node_id, pos=(j*2.5, -i*2), i=i, j=j)
            
            # Dependency on left neighbor
            if j > 0:
                G.add_edge(f"({i},{j-1})", node_id)
    
    return G

def create_dag_v2():
    """Version 2: depends on bottom neighbor (i-1)"""
    G = nx.DiGraph()
    
    for i in range(I):
        for j in range(J):
            node_id = f"({i},{j})"
            G.add_node(node_id, pos=(j*2.5, -i*2), i=i, j=j)
            
            # Dependency on bottom neighbor (previous i)
            if i > 0:
                G.add_edge(f"({i-1},{j})", node_id)
    
    return G

G1 = create_dag_v1()
G2 = create_dag_v2()

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

def draw_dependency_dag(G, ax, title, edge_color, dep_type):
    pos = nx.get_node_attributes(G, 'pos')
    
    # Find boundary nodes (no dependencies)
    boundary_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    interior_nodes = [n for n in G.nodes() if G.in_degree(n) > 0]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color,
                           arrows=True, arrowsize=20, width=2.5, 
                           alpha=0.7, arrowstyle='->')
    
    # Draw boundary nodes (blue circles - slow memory)
    nx.draw_networkx_nodes(G, pos, nodelist=boundary_nodes, ax=ax,
                          node_color='lightblue', node_size=1200,
                          node_shape='o', edgecolors='blue', linewidths=2)
    
    # Draw interior nodes (gray - not yet pebbled)
    nx.draw_networkx_nodes(G, pos, nodelist=interior_nodes, ax=ax,
                          node_color='lightgray', node_size=1200,
                          node_shape='o', edgecolors='gray', linewidths=2)
    
    # Draw labels
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')
    
    # Add text annotation
    textstr = f'Dependency: {dep_type}\n'
    textstr += f'Blue nodes (slow memory): {len(boundary_nodes)}\n'
    textstr += f'Gray nodes (to compute): {len(interior_nodes)}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))

draw_dependency_dag(G1, axes[0], 
                   'Version 1: Inner loop j\nA[i,j] += A[i,j-1]',
                   'blue', 'LEFT neighbor (j-1)')

draw_dependency_dag(G2, axes[1],
                   'Version 2: Inner loop i\nA[i,j] += A[i-1,j]',
                   'red', 'BOTTOM neighbor (i-1)')

plt.tight_layout()
plt.savefig('dependency_dag.png', dpi=150, bbox_inches='tight')
print("\nDependency DAGs saved as 'dependency_dag.png'")
plt.show()

# Analysis
print("\n" + "=" * 60)
print("DEPENDENCY STRUCTURE")
print("=" * 60)

print("\nVersion 1 (inner loop j, depends on j-1):")
print("  Each ROW is independent")
print("  Within a row: sequential dependency left-to-right")
for i in range(I):
    deps = [f"({i},{j})" for j in range(J)]
    print(f"  Row {i}: {' → '.join(deps)}")

print("\nVersion 2 (inner loop i, depends on i-1):")
print("  Each COLUMN is independent")
print("  Within a column: sequential dependency top-to-bottom")
for j in range(J):
    deps = [f"({i},{j})" for i in range(I)]
    print(f"  Column {j}: {' → '.join(deps)}")

print("\n" + "=" * 60)
print("MEMORY ACCESS PATTERNS")
print("=" * 60)

print("\nVersion 1 (i outer, j inner - unit stride):")
print("  • Accesses within same row (contiguous memory)")
print("  • Reading A[i,j-1] then A[i,j] → sequential access")
print("  • ✓ GOOD cache performance")
print(f"  • Can parallelize across {I} rows")

print("\nVersion 2 (j outer, i inner - non-unit stride):")
print("  • Accesses within same column (strided memory)")
print(f"  • Reading A[i-1,j] then A[i,j] → stride of {J} elements")
print("  • ✗ POOR cache performance (column-major access in row-major storage)")
print(f"  • Can parallelize across {J} columns")

print("\n" + "=" * 60)