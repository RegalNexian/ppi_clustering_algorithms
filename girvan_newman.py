from networkx.algorithms.community import girvan_newman
def run_girvan_newman(G):
    """
    Runs the Girvan-Newman algorithm.
    Note: This algorithm is very slow on large graphs.
    It returns the communities after the first iteration (one level of division).
    For a more refined partition, one would typically iterate and evaluate modularity,
    which is computationally very expensive and beyond the scope of this GUI's responsiveness.
    """
    if G.number_of_edges() == 0 or G.number_of_nodes() < 2:
        return [list(G.nodes())] if G.nodes() else []

    comp = girvan_newman(G)
    try:
        # Get the communities from the first split (first item from the generator)
        return [list(c) for c in next(comp)]
    except StopIteration:
        return [list(G.nodes())]
