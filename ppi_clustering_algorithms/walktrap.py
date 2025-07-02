try:
    import igraph as ig # For Walktrap and Fast Greedy
    # Note: python-igraph is typically installed as 'python-igraph', but imported as 'igraph'
except ImportError:
    ig = None # Set to None if not available, functions will raise error later


def nx_to_igraph(nx_graph):
    """Converts a NetworkX graph to an iGraph graph, handling node mapping."""
    if ig is None:
        raise ImportError("igraph package not found. Install with: pip install python-igraph")

    # Create a mapping from NetworkX node names to iGraph integer IDs
    node_map = {node: i for i, node in enumerate(nx_graph.nodes())}
    reverse_node_map = {i: node for node, i in node_map.items()}

    # Create iGraph graph
    ig_graph = ig.Graph(directed=False) # Assuming undirected for community detection
    ig_graph.add_vertices(len(nx_graph.nodes()))
    ig_graph.add_edges([(node_map[u], node_map[v]) for u, v in nx_graph.edges()])
    
    # Store original node names as attributes for later retrieval
    ig_graph.vs["name"] = [reverse_node_map[i] for i in range(len(nx_graph.nodes()))]
    
    return ig_graph, reverse_node_map


def run_walktrap(G):
    """Runs the Walktrap community detection algorithm."""
    if not G.nodes():
        return []
    
    if ig is None:
        raise ImportError("igraph package not found. Install with: pip install python-igraph")

    if G.number_of_edges() == 0:
        return [[n] for n in G.nodes()]

    ig_graph, reverse_node_map = nx_to_igraph(G)

    print("Running Walktrap algorithm...")
    try:
        dendrogram = ig_graph.community_walktrap(steps=4) 
        partition = dendrogram.as_clustering()
        
        clusters = []
        for cluster_indices in partition:
            clusters.append([reverse_node_map[idx] for idx in cluster_indices])
        
        return clusters
    except Exception as e:
        raise RuntimeError(f"Walktrap clustering failed: {e}") from e
