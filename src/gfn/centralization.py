"""Centralization-Based Model."""
from typing import Dict
import networkx as nx
import numpy as np


class CentralizationModel:
    """Centralization-Based Model for measuring hub concentration."""

    @staticmethod
    def compute_centralization_features(G: nx.DiGraph) -> Dict[str, float]:
        """Extract centralization features from graph."""
        if G.number_of_nodes() < 2:
            return {
                "degree_centralization": 0.0,
                "degree_gini": 0.0,
                "betweenness_centralization": 0.0,
                "betweenness_gini": 0.0,
                "pagerank_variance": 0.0,
                "pagerank_gini": 0.0,
            }

        G_und = G.to_undirected()
        features = {}

        try:
            degrees = [G_und.degree(n) for n in G_und.nodes()]
            degree_cent = nx.degree_centralization(G_und)
            degree_gini = CentralizationModel._gini_coefficient(degrees)
            features["degree_centralization"] = float(degree_cent)
            features["degree_gini"] = float(degree_gini)
        except Exception:
            features["degree_centralization"] = 0.0
            features["degree_gini"] = 0.0

        try:
            betweenness = nx.betweenness_centrality(G_und)
            betweenness_cent = nx.betweenness_centralization(G_und)
            betweenness_vals = list(betweenness.values())
            betweenness_gini = CentralizationModel._gini_coefficient(betweenness_vals)
            features["betweenness_centralization"] = float(betweenness_cent)
            features["betweenness_gini"] = float(betweenness_gini)
        except Exception:
            features["betweenness_centralization"] = 0.0
            features["betweenness_gini"] = 0.0

        try:
            pagerank = nx.pagerank(G_und)
            pr_values = list(pagerank.values())
            features["pagerank_variance"] = float(np.var(pr_values)) if pr_values else 0.0
            features["pagerank_gini"] = float(CentralizationModel._gini_coefficient(pr_values))
        except Exception:
            features["pagerank_variance"] = 0.0
            features["pagerank_gini"] = 0.0

        return features

    @staticmethod
    def _gini_coefficient(values: list) -> float:
        """Calculate Gini coefficient for inequality measure."""
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals + 1e-10)
        return (2 * np.sum(np.arange(1, n+1) * sorted_vals)) / (n * np.sum(sorted_vals + 1e-10)) - (n + 1) / n
