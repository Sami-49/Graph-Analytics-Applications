"""Spectral Graph Analysis Model - stable computations on LCC."""
from typing import Dict
import numpy as np
import networkx as nx

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _safe_float(x: float, default: float = 0.0) -> float:
    v = float(x)
    return v if np.isfinite(v) else default


def _extract_lcc_undirected(G: nx.DiGraph) -> nx.Graph:
    G_und = G.to_undirected()
    if G_und.number_of_edges() == 0:
        return G_und
    comps = list(nx.connected_components(G_und))
    lcc = max(comps, key=len)
    return G_und.subgraph(lcc).copy()


class SpectralModel:
    """
    Spectral Graph Analysis Model.
    spectral_radius, fiedler_value (2nd smallest Laplacian), laplacian_energy.
    Uses undirected LCC; sparse eigs for n>200, dense for small graphs.
    """

    @staticmethod
    def compute_spectral_features(G: nx.DiGraph) -> Dict[str, float]:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        zeros = {
            "spectral_radius": 0.0,
            "fiedler_value": 0.0,
            "laplacian_energy": 0.0,
        }

        if n < 3 or m == 0:
            return zeros

        G_lcc = _extract_lcc_undirected(G)
        n_lcc = G_lcc.number_of_nodes()
        m_lcc = G_lcc.number_of_edges()
        if n_lcc < 3 or m_lcc == 0:
            return zeros

        avg_degree = 2.0 * m_lcc / n_lcc if n_lcc > 0 else 0.0

        features: Dict[str, float] = {}

        try:
            A = nx.adjacency_matrix(G_lcc)
            if n_lcc > 200 and HAS_SCIPY:
                vals, _ = eigsh(A.astype(float), k=1, which="LA", return_eigenvectors=True)
                spectral_radius = float(np.real(vals[-1]))
            else:
                A_dense = np.asarray(A.todense(), dtype=float)
                vals = np.linalg.eigvalsh(A_dense)
                spectral_radius = float(np.max(vals))
            features["spectral_radius"] = _safe_float(spectral_radius)
        except Exception:
            features["spectral_radius"] = 0.0

        try:
            L_norm = nx.normalized_laplacian_matrix(G_lcc)
            L = nx.laplacian_matrix(G_lcc)
            k_eig = min(n_lcc - 1, max(2, n_lcc // 2))
            k_eig = max(2, k_eig)

            if n_lcc > 200 and HAS_SCIPY:
                vals_norm, _ = eigsh(L_norm.astype(float), k=k_eig, which="SM", return_eigenvectors=False)
                vals_norm = np.sort(np.real(vals_norm))
                fiedler = float(vals_norm[1]) if len(vals_norm) > 1 else 0.0
            else:
                try:
                    vals_norm = np.linalg.eigvalsh(np.asarray(L_norm.todense(), dtype=float))
                    vals_norm = np.sort(vals_norm)
                    fiedler = float(vals_norm[1]) if len(vals_norm) > 1 else 0.0
                except Exception:
                    vals_std = np.linalg.eigvalsh(np.asarray(L.todense(), dtype=float))
                    vals_std = np.sort(vals_std)
                    fiedler = float(vals_std[1]) if len(vals_std) > 1 else 0.0

            features["fiedler_value"] = _safe_float(fiedler)
        except Exception:
            try:
                L = nx.laplacian_matrix(G_lcc)
                if n_lcc > 200 and HAS_SCIPY:
                    vals_std, _ = eigsh(L.astype(float), k=min(n_lcc - 1, 20), which="SM", return_eigenvectors=False)
                    vals_std = np.sort(np.real(vals_std))
                else:
                    vals_std = np.linalg.eigvalsh(np.asarray(L.todense(), dtype=float))
                    vals_std = np.sort(vals_std)
                fiedler = float(vals_std[1]) if len(vals_std) > 1 else 0.0
                features["fiedler_value"] = _safe_float(fiedler)
            except Exception:
                features["fiedler_value"] = 0.0

        try:
            L = nx.laplacian_matrix(G_lcc)
            if n_lcc > 200 and HAS_SCIPY:
                vals_L, _ = eigsh(L.astype(float), k=min(n_lcc - 1, 50), which="LM", return_eigenvectors=False)
                vals_L = np.real(vals_L)
            else:
                vals_L = np.linalg.eigvalsh(np.asarray(L.todense(), dtype=float))
            laplacian_energy = float(np.sum(np.abs(vals_L - avg_degree)))
            features["laplacian_energy"] = _safe_float(laplacian_energy)
        except Exception:
            features["laplacian_energy"] = 0.0

        return features

    @staticmethod
    def spectral_score_from_components(
        *,
        spectral_radius: float,
        fiedler_value: float,
        laplacian_energy: float,
    ) -> float:
        """Stable, bounded spectral score in [0,1].

        Uses monotone transforms to avoid extreme scaling:
        - spectral_radius: r/(1+r)
        - fiedler_value: 1-exp(-lambda2)
        - laplacian_energy: E/(1+E)
        """

        r = float(spectral_radius) if np.isfinite(spectral_radius) else 0.0
        l2 = float(fiedler_value) if np.isfinite(fiedler_value) else 0.0
        e = float(laplacian_energy) if np.isfinite(laplacian_energy) else 0.0

        r_n = r / (1.0 + max(r, 0.0)) if r >= 0 else 0.0
        l2_n = 1.0 - float(np.exp(-max(l2, 0.0)))
        e_n = e / (1.0 + max(e, 0.0)) if e >= 0 else 0.0

        score = (r_n + l2_n + e_n) / 3.0
        return float(np.clip(score, 0.0, 1.0))
