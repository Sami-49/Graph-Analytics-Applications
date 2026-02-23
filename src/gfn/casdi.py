"""
Community-Aware Spectral Diffusion Index (CASDI).
CASDI = α*z(spectral_radius) + β*z(core_density) + γ*z(bridging_ratio)
     + δ*z(centralization_index) + ε*z(virality)
Robust z-score: z=(x-median)/IQR, fallback to std if IQR=0.
"""
from typing import Dict, List, Optional
import numpy as np
import networkx as nx
from dataclasses import dataclass

from .spectral import SpectralModel
from .kcore import KCoreModel
from .community import CommunityModel
from .centralization import CentralizationModel
from .virality import ViralityModel


def _robust_zscore(x: np.ndarray, median: float, iqr: float, std: float) -> np.ndarray:
    denom = iqr if iqr > 1e-10 else (std if std > 1e-10 else 1.0)
    return (x - median) / (denom + 1e-10)


def _robust_z(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    m = float(np.median(a))
    q75, q25 = np.percentile(a, 75), np.percentile(a, 25)
    iqr = float(q75 - q25)
    std = float(np.std(a))
    return _robust_zscore(a, m, iqr, std)


def _stack_components(components: "CASDIComponents") -> np.ndarray:
    z_sr = _robust_z(components.spectral_radius)
    z_cd = _robust_z(components.core_density)
    z_br = _robust_z(components.bridging_ratio)
    z_ci = _robust_z(components.centralization_index)
    z_vir = _robust_z(components.virality)
    return np.vstack([z_sr, z_cd, z_br, z_ci, z_vir]).T


@dataclass
class CASDIComponents:
    spectral_radius: np.ndarray
    core_density: np.ndarray
    bridging_ratio: np.ndarray
    centralization_index: np.ndarray
    virality: np.ndarray


class CASDIModel:
    """CASDI with robust z-score normalization and ablations."""

    @staticmethod
    def compute_community_bridging_score(G: nx.DiGraph) -> float:
        inter_edges = 0
        G_und = G.to_undirected()
        total_edges = G_und.number_of_edges()
        if total_edges == 0:
            return 0.0

        try:
            import community.community_louvain as community_louvain
            partition = community_louvain.best_partition(G_und)
        except ImportError:
            partition = {}
            for i, comp in enumerate(nx.connected_components(G_und)):
                for node in comp:
                    partition[node] = i

        for u, v in G_und.edges():
            if partition.get(u) != partition.get(v):
                inter_edges += 1
        return inter_edges / total_edges

    @staticmethod
    def compute_centralization_index(G: nx.DiGraph) -> float:
        G_und = G.to_undirected()
        try:
            degree_cent = nx.degree_centralization(G_und)
            pagerank = nx.pagerank(G_und)
            pr_values = np.array(list(pagerank.values()))
            pr_var = np.var(pr_values) if len(pr_values) > 0 else 0.0
            pr_cent = min(pr_var / 0.01, 1.0) if pr_var > 0 else 0.0
            return float(0.6 * degree_cent + 0.4 * pr_cent)
        except Exception:
            return 0.0

    @staticmethod
    def compute_casdi_scores(
        components: CASDIComponents,
        alpha: float, beta: float, gamma: float, delta: float, epsilon: float,
    ) -> Dict[str, np.ndarray]:
        sr, cd, br, ci, vir = (
            components.spectral_radius, components.core_density,
            components.bridging_ratio, components.centralization_index,
            components.virality,
        )

        z_sr = _robust_z(sr)
        z_cd = _robust_z(cd)
        z_br = _robust_z(br)
        z_ci = _robust_z(ci)
        z_vir = _robust_z(vir)

        full = alpha * z_sr + beta * z_cd + gamma * z_br + delta * z_ci + epsilon * z_vir

        return {
            "CASDI_full": full,
            "CASDI_minus_spectral": beta * z_cd + gamma * z_br + delta * z_ci + epsilon * z_vir,
            "CASDI_minus_core": alpha * z_sr + gamma * z_br + delta * z_ci + epsilon * z_vir,
            "CASDI_minus_bridge": alpha * z_sr + beta * z_cd + delta * z_ci + epsilon * z_vir,
            "CASDI_minus_centralization": alpha * z_sr + beta * z_cd + gamma * z_br + epsilon * z_vir,
            "CASDI_minus_virality": alpha * z_sr + beta * z_cd + gamma * z_br + delta * z_ci,
        }

    @staticmethod
    def fit_casdi_v2_weights(
        *,
        components: CASDIComponents,
        y: np.ndarray,
        train_mask: np.ndarray,
        random_state: int = 42,
    ) -> Dict[str, object]:
        """Fit CASDI_v2 weights on train only using a logistic regression decision function.

        - Inputs are robust-z normalized components.
        - We return coefficients and intercept.
        """

        from sklearn.linear_model import LogisticRegression

        X = _stack_components(components)
        y2 = np.asarray(y, dtype=int)
        m = np.asarray(train_mask, dtype=bool)

        m = m & np.isin(y2, [0, 1])
        if int(np.sum(m)) < 10 or len(np.unique(y2[m])) < 2:
            return {
                "ok": False,
                "reason": "not_enough_labeled_train",
                "coef": np.zeros(X.shape[1], dtype=float),
                "intercept": 0.0,
            }

        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=int(random_state),
        )
        clf.fit(X[m], y2[m])

        coef = np.asarray(clf.coef_, dtype=float).reshape(-1)
        intercept = float(np.asarray(clf.intercept_, dtype=float).reshape(-1)[0])
        return {
            "ok": True,
            "reason": "fit",
            "coef": coef,
            "intercept": intercept,
        }

    @staticmethod
    def casdi_v2_scores_from_fit(
        *,
        components: CASDIComponents,
        fit: Dict[str, object],
    ) -> Dict[str, np.ndarray]:
        """Compute CASDI_v2 decision scores for all events + ablations."""

        X = _stack_components(components)
        coef = np.asarray(fit.get("coef", np.zeros(X.shape[1], dtype=float)), dtype=float).reshape(-1)
        intercept = float(fit.get("intercept", 0.0))

        # full decision score
        full = X @ coef + intercept

        # ablations: set one component to 0 (removing its contribution)
        out: Dict[str, np.ndarray] = {"CASDI_v2_full": full}
        names = ["spectral", "core", "bridge", "centralization", "virality"]
        for j, nm in enumerate(names):
            X2 = X.copy()
            X2[:, j] = 0.0
            out[f"CASDI_v2_minus_{nm}"] = X2 @ coef + intercept
        return out

    @staticmethod
    def compute_casdi_features(G: nx.DiGraph, config) -> Dict[str, float]:
        if G.number_of_nodes() < 2:
            return {
                "casdi_spectral_component": 0.0,
                "casdi_core_component": 0.0,
                "casdi_bridging_component": 0.0,
                "casdi_centralization_component": 0.0,
                "casdi_virality_component": 0.0,
                "casdi_score": 0.0,
            }

        sf = SpectralModel.compute_spectral_features(G)
        kf = KCoreModel.compute_kcore_features(G)
        cf = CommunityModel.compute_community_features(G)
        cif = CentralizationModel.compute_centralization_features(G)
        vf = ViralityModel.compute_virality_features(G)

        spec_radius = sf.get("spectral_radius", 0.0)
        core_density = kf.get("core_density", 0.0)
        bridging = cf.get("bridging_ratio", cf.get("inter_community_edge_ratio", 0.0))
        cent_idx = CASDIModel.compute_centralization_index(G)
        virality = vf.get("structural_virality", 0.0)

        return {
            "casdi_spectral_component": float(spec_radius),
            "casdi_core_component": float(core_density),
            "casdi_bridging_component": float(bridging),
            "casdi_centralization_component": float(cent_idx),
            "casdi_virality_component": float(virality),
            "casdi_score": 0.0,
        }
