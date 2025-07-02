from pathlib import Path
import networkx as nx
import scipy.io # Used for mmread
import threading # Import threading module
import ast # For literal_eval to parse list-like strings

# This module provides a function to load a graph from a file.
def load_graph(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"No file found at the path: {path}")

    def is_matrix_market_format(p):
        with open(p, 'r') as f:
            first_line = f.readline()
        return first_line.startswith('%%MatrixMarket')

    try:
        if path.endswith(".mtx") and is_matrix_market_format(path):
            mtx = scipy.io.mmread(path)
            G = nx.from_scipy_sparse_array(mtx)
            G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes()})
            print(f"Successfully loaded Matrix Market graph: {Path(path).name}")
        else:
            edges = []
            with open(path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line.startswith('%') or line == "":
                        continue
                    parts = line.replace(',', ' ').replace('\t', ' ').split()
                    if len(parts) >= 2:
                        edges.append((parts[0], parts[1]))
                    else:
                        print(f"Warning: Line {line_num+1} has fewer than two node parts: '{line}'")

            G = nx.Graph()
            G.add_edges_from(edges)
            print(f"Successfully loaded edge list graph: {Path(path).name}")
    except Exception as e:
        raise ValueError(f"Graph loading failed: {e}")

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # --- RE-INSTATED: Handle disconnected graphs by extracting largest connected component ---
    if not nx.is_connected(G):
        print("Graph is disconnected. Extracting largest connected component for analysis.")
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        except ValueError:
            print("No valid components found. Graph is empty after component extraction.")
            G = nx.Graph() # Ensure G is an empty graph if no components

    if G.number_of_nodes() == 0:
        raise ValueError("Final graph is empty after preprocessing. Please verify the file format.")

    return G
