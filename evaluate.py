
from networkx.algorithms.community.quality import modularity
import numpy as np
# --- Suppress SciPy Sparse Efficiency Warning ---
import warnings  # Import the warnings module
from scipy.sparse import SparseEfficiencyWarning  # Import the specific warning type


# --- Suppress specific SciPy Sparse Efficiency Warning ---
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

# --- Unified Evaluation Function ---
def evaluate_clusters(G, clusters):
    """Evaluates clustering by calculating modularity and cluster statistics."""
    stats = {
        "num_clusters": 0,
        "avg_cluster_size": 0,
        "max_cluster_size": 0,
        "min_cluster_size": 0,
        "modularity": 0.0
    }

    if not clusters or len(clusters) == 0:
        return stats

    sizes = [len(c) for c in clusters if len(c) > 0]
    if not sizes:
        return stats

    stats["num_clusters"] = len(sizes)
    stats["avg_cluster_size"] = float(np.mean(sizes))
    stats["max_cluster_size"] = max(sizes)
    stats["min_cluster_size"] = min(sizes)

    try:
        communities = [set(c) for c in clusters if c]
        stats["modularity"] = modularity(G, communities)
    except Exception:
        stats["modularity"] = 0.0

    return stats
