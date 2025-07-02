import community as community_louvain  # type: ignore
import networkx as nx
def run_louvain(G):
    """Runs the Louvain community detection algorithm."""
    if not G.nodes(): # Handle empty graph
        return []
    partition = community_louvain.best_partition(G)
    clusters = {}
    for node, comm_id in partition.items():
        clusters.setdefault(comm_id, []).append(node)
    return list(clusters.values())
