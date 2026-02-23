"""Community Detection Model (Louvain)."""
from typing import Dict
import numpy as np
import networkx as nx

try:
    import community.community_louvain as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


class CommunityModel:
    """Community Detection Model (Louvain algorithm)."""

    @staticmethod
    def compute_community_features(G: nx.DiGraph) -> Dict[str, float]:
        """Extract community detection features from graph."""
        if G.number_of_nodes() < 2:
            return {
                "modularity": 0.0,
                "n_communities": 1.0,
                "community_entropy": 0.0,
                "inter_community_edge_ratio": 0.0,
                "bridging_ratio": 0.0,
                "community_score": 0.0,
                "avg_community_size": 0.0,
                "max_community_size": 0.0,
            }

        G_und = G.to_undirected()

        if not HAS_LOUVAIN:
            components = list(nx.connected_components(G_und))
            return {
                "modularity": 0.0,
                "n_communities": float(len(components)),
                "community_entropy": 0.0,
                "inter_community_edge_ratio": 0.0,
                "bridging_ratio": 0.0,
                "community_score": 0.0,
                "avg_community_size": float(len(G.nodes()) / len(components)) if components else 0.0,
                "max_community_size": float(max(len(c) for c in components)) if components else 0.0,
            }

        try:
            partition = community_louvain.best_partition(G_und)
            modularity = community_louvain.modularity(partition, G_und)

            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)

            n_communities = len(communities)
            community_sizes = [len(c) for c in communities.values()]
            
            n_nodes = G_und.number_of_nodes()
            community_probs = [size / n_nodes for size in community_sizes]
            entropy = -sum(p * np.log(p + 1e-10) for p in community_probs)

            inter_edges = 0
            total_edges = G_und.number_of_edges()
            for u, v in G_und.edges():
                if partition[u] != partition[v]:
                    inter_edges += 1

            inter_ratio = inter_edges / total_edges if total_edges > 0 else 0.0
            bridging_ratio = float(inter_ratio)

            mod_norm = float(np.clip((modularity + 1) / 2, 0, 1))
            ent_max = np.log(n_nodes + 1)
            ent_norm = float(np.clip(entropy / (ent_max + 1e-10), 0, 1))
            community_score = float(np.clip((mod_norm + ent_norm + bridging_ratio) / 3, 0, 1))

            return {
                "modularity": float(modularity),
                "n_communities": float(n_communities),
                "community_entropy": float(entropy),
                "inter_community_edge_ratio": float(inter_ratio),
                "bridging_ratio": bridging_ratio,
                "community_score": community_score,
                "avg_community_size": float(np.mean(community_sizes)),
                "max_community_size": float(np.max(community_sizes)),
            }
        except Exception:
            return {
                "modularity": 0.0,
                "n_communities": 1.0,
                "community_entropy": 0.0,
                "inter_community_edge_ratio": 0.0,
                "bridging_ratio": 0.0,
                "community_score": 0.0,
                "avg_community_size": 0.0,
                "max_community_size": 0.0,
            }
