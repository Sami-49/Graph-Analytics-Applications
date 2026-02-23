"""Propagation Heterogeneity Score (HPS) - layer-based cascade diversity."""
from typing import Dict
import numpy as np
import networkx as nx
from collections import deque

from .graph_builder import GraphBuilder


def _gini(x: np.ndarray) -> float:
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x.astype(float))
    n = len(x)
    cum = np.cumsum(x)
    return float((2 * np.sum((np.arange(1, n + 1) * x)) - (n + 1) * np.sum(x)) / (n * np.sum(x) + 1e-10))


def _normalized_entropy(p: np.ndarray) -> float:
    if len(p) == 0 or np.sum(p) == 0:
        return 0.0
    p = p / np.sum(p)
    p = p[p > 0]
    n = len(p)
    if n <= 1:
        return 0.0
    max_ent = np.log(n)
    ent = -np.sum(p * np.log(p + 1e-10))
    return float(ent / max_ent) if max_ent > 0 else 0.0


class HybridPropagationScoreModel:
    """
    Propagation Heterogeneity Score (HPS).
    BFS layers from root, entropy + gini of layer sizes.
    """

    @staticmethod
    def compute_hps_features(G: nx.DiGraph, basic_stats: Dict = None,
                             spectral_features: Dict = None, kcore_features: Dict = None,
                             community_features: Dict = None,
                             centralization_features: Dict = None) -> Dict[str, float]:
        """
        Compute HPS as propagation heterogeneity: layer entropy + gini.
        Uses undirected graph for BFS distances.
        """
        zeros = {
            "hps_entropy": 0.0,
            "hps_gini": 0.0,
            "hps_score": 0.0,
        }

        if G.number_of_nodes() < 2:
            return zeros

        G_und = G.to_undirected()
        root = GraphBuilder.find_root(G)
        if root is None:
            return zeros

        levels = GraphBuilder.compute_levels_undirected(G_und, root)
        if not levels:
            return zeros

        layer_sizes = {}
        for _, level in levels.items():
            layer_sizes[level] = layer_sizes.get(level, 0) + 1

        layer_arr = np.array([layer_sizes.get(d, 0) for d in range(max(layer_sizes.keys()) + 1)])
        if len(layer_arr) == 0 or np.sum(layer_arr) == 0:
            return zeros

        hps_entropy = _normalized_entropy(layer_arr)
        hps_gini = _gini(layer_arr)
        hps_score = float(0.5 * hps_entropy + 0.5 * hps_gini)
        hps_score = float(np.clip(hps_score, 0, 1))

        return {
            "hps_entropy": float(hps_entropy),
            "hps_gini": float(hps_gini),
            "hps_score": hps_score,
        }
