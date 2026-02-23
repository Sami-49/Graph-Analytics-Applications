"""Unit tests for spectral model."""
import numpy as np
import networkx as nx
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gfn.spectral import SpectralModel


def test_star_graph():
    G = nx.star_graph(5)
    G_dir = nx.DiGraph(G)
    features = SpectralModel.compute_spectral_features(G_dir)
    assert features["spectral_radius"] > 0
    assert features["fiedler_value"] >= 0
    assert features["laplacian_energy"] >= 0

    score = SpectralModel.spectral_score_from_components(
        spectral_radius=features["spectral_radius"],
        fiedler_value=features["fiedler_value"],
        laplacian_energy=features["laplacian_energy"],
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_path_graph():
    G = nx.path_graph(5)
    G_dir = nx.DiGraph(G)
    features = SpectralModel.compute_spectral_features(G_dir)
    assert features["spectral_radius"] > 0
    assert features["fiedler_value"] >= 0
    assert features["laplacian_energy"] >= 0

    score = SpectralModel.spectral_score_from_components(
        spectral_radius=features["spectral_radius"],
        fiedler_value=features["fiedler_value"],
        laplacian_energy=features["laplacian_energy"],
    )
    assert 0.0 <= score <= 1.0


def test_complete_graph():
    G = nx.complete_graph(4)
    G_dir = nx.DiGraph(G)
    features = SpectralModel.compute_spectral_features(G_dir)
    assert features["spectral_radius"] > 0
    assert features["fiedler_value"] >= 0
    assert features["laplacian_energy"] >= 0

    score = SpectralModel.spectral_score_from_components(
        spectral_radius=features["spectral_radius"],
        fiedler_value=features["fiedler_value"],
        laplacian_energy=features["laplacian_energy"],
    )
    assert 0.0 <= score <= 1.0


def test_two_component_uses_lcc():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    G_dir = nx.DiGraph(G)
    features = SpectralModel.compute_spectral_features(G_dir)
    assert features["spectral_radius"] > 0
    assert features["laplacian_energy"] >= 0

    score = SpectralModel.spectral_score_from_components(
        spectral_radius=features["spectral_radius"],
        fiedler_value=features["fiedler_value"],
        laplacian_energy=features["laplacian_energy"],
    )
    assert 0.0 <= score <= 1.0


def test_small_graph_returns_zeros():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    features = SpectralModel.compute_spectral_features(G)
    assert features["spectral_radius"] == 0
    assert features["fiedler_value"] == 0
    assert features["laplacian_energy"] == 0

    score = SpectralModel.spectral_score_from_components(
        spectral_radius=features["spectral_radius"],
        fiedler_value=features["fiedler_value"],
        laplacian_energy=features["laplacian_energy"],
    )
    assert score == 0.0


def test_empty_or_single_node():
    G = nx.DiGraph()
    G.add_node("a")
    features = SpectralModel.compute_spectral_features(G)
    assert features["spectral_radius"] == 0
    assert features["fiedler_value"] == 0
    assert features["laplacian_energy"] == 0
