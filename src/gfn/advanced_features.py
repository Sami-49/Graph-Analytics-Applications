"""Advanced graph features for per-event feature table."""
from typing import Dict
import numpy as np
import networkx as nx

from .graph_builder import GraphBuilder
from .centralization import CentralizationModel


def _safe_float(x: float, default: float = 0.0) -> float:
    v = float(x)
    return v if np.isfinite(v) else default


def _gini(values: np.ndarray) -> float:
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    v = np.sort(values.astype(float))
    n = len(v)
    return float((2 * np.sum((np.arange(1, n + 1) * v)) - (n + 1) * np.sum(v)) / (n * np.sum(v) + 1e-10))


def _entropy(p: np.ndarray) -> float:
    if len(p) == 0 or np.sum(p) <= 0:
        return 0.0
    p = p[p > 0] / np.sum(p)
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + 1e-10)))


class AdvancedFeatures:
    """10 advanced graph features with safe fallbacks."""

    @staticmethod
    def compute(G: nx.DiGraph, basic_stats: Dict = None) -> Dict[str, float]:
        if G.number_of_nodes() < 2:
            return {
                "assortativity": 0.0,
                "transitivity": 0.0,
                "avg_clustering": 0.0,
                "diameter_lcc": 0.0,
                "avg_shortest_path_lcc": 0.0,
                "degree_gini": 0.0,
                "pagerank_entropy": 0.0,
                "kcore_max": 0.0,
                "depth": 0.0,
                "breadth": 0.0,
            }

        G_und = G.to_undirected()
        stats = basic_stats or GraphBuilder.compute_basic_stats(G)
        features = {}

        try:
            deg_assort = nx.degree_assortativity_coefficient(G_und)
            features["assortativity"] = _safe_float(deg_assort)
        except (nx.NetworkXError, Exception):
            features["assortativity"] = 0.0

        try:
            features["transitivity"] = _safe_float(nx.transitivity(G_und))
        except Exception:
            features["transitivity"] = 0.0

        try:
            features["avg_clustering"] = _safe_float(nx.average_clustering(G_und))
        except Exception:
            features["avg_clustering"] = 0.0

        try:
            lcc = max(nx.connected_components(G_und), key=len)
            G_lcc = G_und.subgraph(lcc)
            n_lcc = G_lcc.number_of_nodes()
            if n_lcc >= 2:
                features["diameter_lcc"] = _safe_float(nx.diameter(G_lcc))
                features["avg_shortest_path_lcc"] = _safe_float(nx.average_shortest_path_length(G_lcc))
            else:
                features["diameter_lcc"] = 0.0
                features["avg_shortest_path_lcc"] = 0.0
        except (nx.NetworkXError, Exception):
            features["diameter_lcc"] = 0.0
            features["avg_shortest_path_lcc"] = 0.0

        try:
            degs = np.array([d for _, d in G_und.degree()])
            features["degree_gini"] = _safe_float(_gini(degs))
        except Exception:
            features["degree_gini"] = 0.0

        try:
            pr = nx.pagerank(G_und)
            pr_vals = np.array(list(pr.values()))
            features["pagerank_entropy"] = _safe_float(_entropy(pr_vals))
        except Exception:
            features["pagerank_entropy"] = 0.0

        try:
            core_nums = nx.core_number(G_und)
            features["kcore_max"] = float(max(core_nums.values())) if core_nums else 0.0
        except Exception:
            features["kcore_max"] = 0.0

        features["depth"] = float(stats.get("depth", 0))
        features["breadth"] = float(stats.get("breadth", 0))

        return features
