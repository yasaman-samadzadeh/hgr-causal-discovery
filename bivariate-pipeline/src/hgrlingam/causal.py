"""
LiNGAM helper utilities
=======================

run_lingam(Z_train, algo_name="direct", **kwargs)
    → ordering, B_est, residuals

*2025‑05‑16 fix*: avoid calling `DirectLiNGAM._residual`, which expects
arguments; instead compute residuals manually when `model.residuals_` is not a
ready‐made array/DataFrame.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from lingam import DirectLiNGAM, ICALiNGAM

# PairwiseLiNGAM is optional in older versions — import defensively
try:
    from lingam import PairwiseLiNGAM
except ImportError:
    PairwiseLiNGAM = None

ALGOS = {
    "direct":   DirectLiNGAM,
    "ica":      ICALiNGAM,
    "pairwise": PairwiseLiNGAM,
}


def run_lingam(
    Z_train: np.ndarray,
    algo_name: str = "direct",
    **kwargs,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Fit LiNGAM and return (ordering, B_est, residuals)."""
    model_cls = ALGOS.get(algo_name)
    if model_cls is None:
        raise ValueError(f"algorithm '{algo_name}' not available in this lingam version")

    model = model_cls(**kwargs)
    model.fit(Z_train)

    ordering = model.causal_order_
    B_est = model.adjacency_matrix_

    # ---------- retrieve residuals robustly -------------------------
    residuals = None

    # Newer lingam versions store residuals_ as ndarray / DataFrame
    if hasattr(model, "residuals_"):
        res = model.residuals_
        if isinstance(res, pd.DataFrame):
            residuals = res.values
        elif isinstance(res, np.ndarray):
            residuals = res

    # Older versions: no ready residuals; compute manually
    if residuals is None:
        I_minus_B = np.eye(B_est.shape[0]) - B_est
        residuals = (I_minus_B @ Z_train.T).T

    return ordering, B_est, residuals
