"""k-Core Decomposition Model."""
from typing import Dict
import numpy as np
import networkx as nx


class KCoreModel:
    """
    k-Core Decomposition Model.
    Features: max core number, avg core number, core density, degeneracy.
    """

    @staticmethod
    def compute_kcore_features(G: nx.DiGraph) -> Dict[str, float]:
        """Extract k-core features from graph."""
        if G.number_of_nodes() < 2:
            return {
                "max_core_number": 0.0,
                "avg_core_number": 0.0,
                "std_core_number": 0.0,
                "core_density": 0.0,
                "degeneracy": 0.0,
                "core_concentration": 0.0,
            }

        G_und = G.to_undirected()

        try:
            core_numbers = nx.core_number(G_und)
            core_values = list(core_numbers.values())

            max_core = float(max(core_values))
            avg_core = float(sum(core_values) / len(core_values)) if core_values else 0.0
            std_core = float(np.std(core_values)) if len(core_values) > 1 else 0.0

            max_k = int(max_core)
            core_nodes = {n for n, k in core_numbers.items() if k == max_k}
            core_subgraph = G_und.subgraph(core_nodes)
            n_core = len(core_nodes)
            n_core_edges = core_subgraph.number_of_edges()
            max_edges = n_core * (n_core - 1) / 2
            core_density = float(n_core_edges / max_edges) if max_edges > 0 else 0.0
            core_concentration = float(n_core / G.number_of_nodes()) if G.number_of_nodes() > 0 else 0.0

            return {
                "max_core_number": max_core,
                "avg_core_number": avg_core,
                "std_core_number": std_core,
                "core_density": core_density,
                "degeneracy": max_core,
                "core_concentration": core_concentration,
            }
        except Exception:
            return {
                "max_core_number": 0.0,
                "avg_core_number": 0.0,
                "std_core_number": 0.0,
                "core_density": 0.0,
                "degeneracy": 0.0,
                "core_concentration": 0.0,
            }
