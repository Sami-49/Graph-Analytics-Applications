"""Structural Virality Model."""
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
from collections import deque


class ViralityModel:
    """Structural Virality Model for measuring cascade shape and propagation."""

    @staticmethod
    def compute_wiener_index_approx(G: nx.DiGraph, sample_size: int = 100) -> float:
        """Approximate Wiener index via sampling."""
        if G.number_of_nodes() < 2:
            return 0.0

        G_und = G.to_undirected()
        nodes = list(G_und.nodes())
        n = len(nodes)

        if n <= sample_size:
            wiener = 0.0
            for source in nodes:
                lengths = nx.single_source_shortest_path_length(G_und, source)
                wiener += sum(lengths.values())
            return wiener / 2

        sample_nodes = np.random.choice(nodes, size=min(sample_size, n), replace=False)
        wiener_sample = 0.0
        for source in sample_nodes:
            lengths = nx.single_source_shortest_path_length(G_und, source)
            wiener_sample += sum(lengths.values())

        wiener_approx = wiener_sample * (n / len(sample_nodes)) / 2
        return wiener_approx

    @staticmethod
    def compute_virality_features(G: nx.DiGraph) -> Dict[str, float]:
        """Extract virality features from graph."""
        from .graph_builder import GraphBuilder

        if G.number_of_nodes() < 2:
            return {
                "wiener_index": 0.0,
                "avg_shortest_path": 0.0,
                "depth": 0.0,
                "breadth": 0.0,
                "structural_virality": 0.0,
                "depth_breadth_product": 0.0,
                "virality_span": 0.0,
            }

        basic_stats = GraphBuilder.compute_basic_stats(G)
        depth = basic_stats["depth"]
        breadth = basic_stats["breadth"]
        avg_short_path = basic_stats["avg_shortest_path"]

        wiener = ViralityModel.compute_wiener_index_approx(G)
        
        structural_virality = (depth * breadth) / (avg_short_path + 1) if avg_short_path >= 0 else 0.0
        depth_breadth_product = float(depth * breadth)
        virality_span = float((depth + breadth) / 2) if (depth + breadth) > 0 else 0.0

        return {
            "wiener_index": float(wiener),
            "avg_shortest_path": float(avg_short_path),
            "depth": float(depth),
            "breadth": float(breadth),
            "structural_virality": float(structural_virality),
            "depth_breadth_product": depth_breadth_product,
            "virality_span": virality_span,
        }
