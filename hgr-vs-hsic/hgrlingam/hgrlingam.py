"""hgr_lingam.py
Standalone implementation of Direct LiNGAM with a kernel‑HGR independence test.
Compatible with lingam ≥ 1.10 and maxcorr 0.1–0.3.

Usage
-----
>>> from hgr_lingam import HGRDirectLiNGAM, shd
>>> mdl = HGRDirectLiNGAM(kernel_sizes=(0.3, 0.3), random_state=0)
>>> mdl.fit(X)
>>> B_hat = mdl.adjacency_matrix_
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import scale
from lingam.base import _BaseLiNGAM
from maxcorr import hgr as _hgr_builder

__all__ = [
    "HGRDirectLiNGAM",
    "shd",
]

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def shd(A: np.ndarray, B: np.ndarray, tol: float = 1e-2) -> int:
    """Structural Hamming Distance between two adjacency matrices."""
    return int(np.sum((np.abs(A) > tol) != (np.abs(B) > tol)))


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class HGRDirectLiNGAM(_BaseLiNGAM):
    """Direct LiNGAM that uses a *kernel‑HGR* independence test.

    Parameters
    ----------
    kernel_sizes : tuple[float, float] | None, default=None
        Bandwidths (σₓ, σᵧ) for the Gaussian kernels applied to (x, y).
        If ``None`` the maxcorr median‑heuristic is used.
    rank : int, default=100
        Nyström feature rank in *Double‑Kernel* HGR. Higher ⇒ slower but
        slightly more accurate.
    random_state, prior_knowledge, apply_prior_knowledge_softly
        Forwarded to the original LiNGAM implementation.
    """

    _hgr_calls: int = 0  # class‑level debug counter

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        kernel_sizes: tuple[float, float] | None = None,
        rank: int = 100,
        random_state: int | None = None,
        prior_knowledge: np.ndarray | None = None,
        apply_prior_knowledge_softly: bool = False,
    ) -> None:
        super().__init__(random_state=random_state)
        self._kernel_sizes = kernel_sizes
        self._rank = rank
        self._Aknw = prior_knowledge
        self._apply_prior_knowledge_softly = apply_prior_knowledge_softly

    # ------------------------------------------------------------------
    # Low‑level HGR wrapper (handles every maxcorr API version) ----------
    # ------------------------------------------------------------------
    def _hgr(self, x: np.ndarray, y: np.ndarray) -> float:
        """Kernel‑HGR statistic for two 1‑D variables.

        Works with all known *maxcorr* APIs. Bandwidths are assigned
        **after** construction via the public ``alpha``/``beta`` attrs so the
        user can keep exactly the same naming convention they used in other
        projects.
        """
        HGRDirectLiNGAM._hgr_calls += 1

        # Build the indicator first – pass only algorithm/backend.
        est = _hgr_builder(algorithm="dk", backend="numpy")  # semantics="hgr" set internally

        # Set Gaussian kernel bandwidths if the user provided them.
        if self._kernel_sizes is not None:
            est.f = int(self._kernel_sizes[0])  # bandwidth for X
            est.g  = int(self._kernel_sizes[1])  # bandwidth for Y

        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)

        # --- maxcorr 0.1.x --------------------------------------------------
        if hasattr(est, "fit"):
            est.fit(X, Y)
            return float(getattr(est, "score_", est.score_))

        # --- maxcorr ≥0.2 ---------------------------------------------------
        if hasattr(est, "learn"):
            est.learn(X, Y)
            return float(getattr(est, "score", est.score))

        # --- compute()/calc() fallbacks -------------------------------------
        if hasattr(est, "compute"):
            return float(est.compute(X, Y))
        if hasattr(est, "calc"):
            return float(est.calc(X, Y))

        # --- callable estimator --------------------------------------------
        if callable(est):
            res = est(X, Y)
            if isinstance(res, (float, int, np.floating)):
                return float(res)
            if hasattr(res, "score"):
                return float(res.score)
            if hasattr(res, "score_"):
                return float(res.score_)

        raise AttributeError("maxcorr indicator did not expose a usable score API")

    # ------------------------------------------------------------------
    # OLS residual (unchanged from official LiNGAM) ----------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        beta = np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)
        return xi - beta * xj

    # ------------------------------------------------------------------
    # Candidate search (trimmed copy of official logic) -----------------
    # ------------------------------------------------------------------
    def _search_candidate(self, U: np.ndarray):
        if self._Aknw is None:
            return U, []
        Uc = [j for j in U if np.nansum(self._Aknw[j][U[U != j]]) == 0]
        return (np.array(Uc), []) if Uc else (U, [])

    # ------------------------------------------------------------------
    # Exogeneity score loop using kernel‑HGR ----------------------------
    # ------------------------------------------------------------------
    def _search_causal_order_hgr(self, X: np.ndarray, U: np.ndarray) -> int:
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        scores: list[float] = []
        for i in Uc:
            s = 0.0
            for j in U:
                if i == j:
                    continue
                xi_std = (X[:, i] - X[:, i].mean()) / X[:, i].std()
                xj_std = (X[:, j] - X[:, j].mean()) / X[:, j].std()
                ri_j = xi_std if (i in Vj and j in Uc) else self._residual(xi_std, xj_std)
                s += self._hgr(xi_std, ri_j) ** 2
            scores.append(s)
        return Uc[int(np.argmin(scores))]

    # ------------------------------------------------------------------
    # Public API – fit --------------------------------------------------
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray):
        X = np.asarray(X)
        U = np.arange(X.shape[1])
        K: list[int] = []
        X_std = scale(X)

        while len(U):
            m = self._search_causal_order_hgr(X_std, U)
            for i in U:
                if i != m:
                    X_std[:, i] = self._residual(X_std[:, i], X_std[:, m])
            K.append(m)
            U = U[U != m]

        self._causal_order = K
        return self._estimate_adjacency_matrix(X)
