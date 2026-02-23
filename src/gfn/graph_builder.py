"""Graph construction from cascade edges."""
from typing import List, Tuple, Optional
import networkx as nx
from collections import deque


class GraphBuilder:
    """Build and analyze cascade graphs."""

    @staticmethod
    def build_graph(edges: List[Tuple[str, str]]) -> nx.DiGraph:
        """Build directed graph from edge list."""
        G = nx.DiGraph()
        G.add_edges_from(edges)
        return G

    @staticmethod
    def find_root(G: nx.DiGraph) -> str:
        """Find root node (in-degree 0, or highest out-degree fallback)."""
        if not G.nodes():
            return None

        for node in G.nodes():
            if G.in_degree(node) == 0:
                return node

        return max(G.nodes(), key=lambda n: G.out_degree(n))

    @staticmethod
    def compute_levels(G: nx.DiGraph, root: str) -> dict:
        """BFS from root to compute levels (depth) via directed edges."""
        levels = {root: 0}
        queue = deque([root])

        while queue:
            node = queue.popleft()
            current_level = levels[node]
            for neighbor in G.successors(node):
                if neighbor not in levels:
                    levels[neighbor] = current_level + 1
                    queue.append(neighbor)

        return levels

    @staticmethod
    def compute_levels_undirected(G_und: nx.Graph, root: str) -> dict:
        """BFS from root on undirected graph for distance-based layers."""
        levels = {root: 0}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            current_level = levels[node]
            for neighbor in G_und.neighbors(node):
                if neighbor not in levels:
                    levels[neighbor] = current_level + 1
                    queue.append(neighbor)
        return levels

    @staticmethod
    def compute_basic_stats(G: nx.DiGraph) -> dict:
        """Compute basic graph statistics."""
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes == 0:
            return {"n_nodes": 0, "n_edges": 0}

        G_undirected = G.to_undirected()

        root = GraphBuilder.find_root(G)
        levels = GraphBuilder.compute_levels(G, root) if root else {}
        depth = max(levels.values()) if levels else 0
        nodes_per_level = {}
        for node, level in levels.items():
            nodes_per_level[level] = nodes_per_level.get(level, 0) + 1
        breadth = max(nodes_per_level.values()) if nodes_per_level else 0

        lcc_size = len(max(nx.weakly_connected_components(G), key=len)) if G.nodes() else 0

        avg_degree = (2 * n_edges / n_nodes) if n_nodes > 0 else 0
        max_degree = max(dict(G.degree()).values()) if G.nodes() else 0

        density = nx.density(G_undirected) if n_nodes > 1 else 0

        avg_clustering = nx.average_clustering(G_undirected) if n_nodes > 2 else 0

        if G.nodes() and lcc_size > 1:
            lcc_nodes = set(max(nx.weakly_connected_components(G), key=len))
            G_lcc = G.subgraph(lcc_nodes).to_undirected()
            diameter = nx.diameter(G_lcc)
            avg_short_path = nx.average_shortest_path_length(G_lcc)
        else:
            diameter = 0
            avg_short_path = 0

        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "density": density,
            "lcc_size": lcc_size,
            "avg_clustering": avg_clustering,
            "diameter": diameter,
            "avg_shortest_path": avg_short_path,
            "depth": depth,
            "breadth": breadth,
            "breadth_depth_ratio": breadth / (depth + 1) if depth >= 0 else 0,
        }
