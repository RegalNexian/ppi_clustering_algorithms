import networkx as nx

import markov_clustering as mc  # type: ignore
from scipy.sparse import SparseEfficiencyWarning # Import the specific warning type

def run_mcl(G):
    """Runs the Markov Clustering (MCL) algorithm on an unweighted graph with string or integer nodes."""
    # Initialize index_to_node here to ensure it's always defined, even if empty
    node_list = []
    node_to_index = {}
    index_to_node = {}

    try:
        # Handle trivial graphs
        if G.number_of_nodes() < 2:
            return [[n] for n in G.nodes()]
        
        if G.number_of_edges() == 0:
            return [[n] for n in G.nodes()]

        # Get sorted node list for consistent ordering
        node_list = sorted(G.nodes())
        n_nodes = len(node_list)
        
        print(f"Processing graph with {n_nodes} nodes and {G.number_of_edges()} edges")
        print(f"Node types: {type(node_list[0]) if node_list else 'N/A'}")
        
        # Create node mapping (works with both string and integer nodes)
        node_to_index = {node: i for i, node in enumerate(node_list)}
        index_to_node = {i: node for i, node in enumerate(node_list)}

        # Create adjacency matrix directly from the NetworkX graph
        try:
            print("Creating adjacency matrix...")
            # Explicitly use nx.to_scipy_sparse_array to guarantee CSR format and float dtype
            matrix = nx.to_scipy_sparse_array(G, nodelist=node_list, dtype=float, format='csr')
            
            matrix.eliminate_zeros() # Remove explicit zeros that might exist
            matrix.sort_indices()  # Sort indices for better performance
            
            print(f"Matrix creation successful:")
            print(f"  Shape: {matrix.shape}")
            print(f"  Non-zero elements: {matrix.nnz}")
            print(f"  Data type: {matrix.dtype}")
            print(f"  Format: {matrix.format}")
            
        except Exception as matrix_error:
            print(f"Matrix creation failed: {matrix_error}")
            raise ValueError(f"Could not create adjacency matrix: {matrix_error}")

        # Validate matrix dimensions
        if matrix.ndim != 2:
            raise ValueError(f"Matrix has {matrix.ndim} dimensions, expected 2")
        
        if matrix.shape[0] != n_nodes or matrix.shape[1] != n_nodes:
            raise ValueError(f"Matrix shape {matrix.shape} doesn't match number of nodes {n_nodes}")
        
        if matrix.nnz == 0:
            raise ValueError("Matrix has no non-zero elements (no edges)")

        # Additional matrix preprocessing for MCL compatibility
        try:
            # Ensure the matrix is symmetric and unweighted (all non-zeros are 1.0)
            matrix_sym = matrix + matrix.T 
            if len(matrix_sym.data) > 0: 
                matrix_sym.data[matrix_sym.data > 0] = 1.0 
            matrix = matrix_sym
            
            # Ensure no self-loops (diagonal should be zero)
            matrix.setdiag(0)
            matrix.eliminate_zeros()
            
            print(f"Matrix preprocessing completed. Final nnz: {matrix.nnz}")
            
        except Exception as prep_error:
            print(f"Matrix preprocessing failed: {prep_error}")

        print("Running MCL algorithm...")
        

        
        result = None # Initialize result to None, will be updated if successful
        
        # Try different approaches to fix the 0-dimensional array issue
        try:
            # Method 1: Standard MCL call
            print("Attempting Method 1: Standard MCL...")
            result = mc.run_mcl(matrix, inflation=2.0, expansion=2)
            
        except Exception as mcl_error1:
            print(f"Method 1 failed: {mcl_error1}")
            
            try:
                # Method 2: Convert to dense matrix (for smaller graphs)
                if n_nodes <= 2000:
                    print("Attempting Method 2: Dense matrix MCL...")
                    dense_matrix = matrix.toarray()
                    result = mc.run_mcl(dense_matrix, inflation=2.0, expansion=2)
                else:
                    print("Graph too large for dense conversion, skipping Method 2.") # Print instead of raise
                    
            except Exception as mcl_error2:
                print(f"Method 2 failed: {mcl_error2}")
                
                try:
                    # Method 3: Ensure matrix is in COO format first, then CSR
                    print("Attempting Method 3: COO->CSR conversion...")
                    coo_matrix = matrix.tocoo()
                    csr_matrix = coo_matrix.tocsr()
                    csr_matrix.eliminate_zeros()
                    result = mc.run_mcl(csr_matrix, inflation=2.0, expansion=2)
                    
                except Exception as mcl_error3:
                    print(f"Method 3 failed: {mcl_error3}")
                    # If all methods fail, result remains None
        
        if result is None:
            raise ValueError("All MCL methods failed to produce a valid result from markov_clustering.run_mcl.")
        
        print("MCL algorithm completed successfully")

        # Extract clusters
        try:
            cluster_indices = mc.get_clusters(result)
            print(f"Found {len(cluster_indices)} clusters (raw MCL output)")
            
            # Convert cluster indices back to original node labels
            clusters = []
            for cluster_idx_list in cluster_indices:
                cluster = []
                for idx in cluster_idx_list:
                    # Safety check for index validity (should be covered by MCL output though)
                    if 0 <= idx < len(index_to_node):
                        cluster.append(index_to_node[idx])
                    else:
                        print(f"Warning: MCL returned invalid node index {idx}. Skipping.")
                if cluster: # Only add non-empty clusters
                    clusters.append(cluster)
            
            # Sort clusters by size (largest first)
            clusters.sort(key=len, reverse=True)
            
            # Remove any empty clusters (already handled above, but good for final cleanup)
            clusters = [cluster for cluster in clusters if cluster]
            
            print(f"Successfully created {len(clusters)} clusters from MCL result.")
            if clusters:
                print(f"Cluster sizes: {[len(c) for c in clusters[:5]]}...")
            
            return clusters

        except Exception as get_clusters_error:
            # This specific error handling helps identify if the problem is in mc.get_clusters
            raise ValueError(f"Failed to extract clusters from MCL result object: {get_clusters_error}")

    except Exception as e:
        # This catches any errors from the matrix setup, MCL running, or cluster extraction.
        print(f"MCL clustering error: {e}")
        # When MCL fails, falling back to connected components is a robust option.
        print("Falling back to connected components clustering (as MCL failed to process the graph).")
        try:
            components = list(nx.connected_components(G))
            result = [list(comp) for comp in components]
            print(f"Connected components fallback successful: {len(result)} components")
            return result
        except Exception as fallback_error:
            print(f"Fallback clustering also failed: {fallback_error}")
            return [[n] for n in G.nodes()] 
        