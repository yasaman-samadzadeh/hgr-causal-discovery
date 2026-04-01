import numpy as np
from typing import Optional

def poly_ridge_residual(x_i: np.ndarray, x_j: np.ndarray, degree: int = 4, lam: Optional[float] = None) -> np.ndarray:
    """Return r_{j|i} = x_j − ĝ(x_i) where ĝ is a degree‑`degree` polynomial.
    A tiny ridge term (default 1e‑3·var(x_j)) avoids numerical blow‑ups."""
    x_i = x_i.ravel()
    x_j = x_j.ravel()
    V = np.vander(x_i, N=degree + 1, increasing=True)           # n × (d+1)
    if lam is None:
        lam = 1e-3 * np.var(x_j)
    A = V.T @ V + lam * np.eye(degree + 1)
    alpha = np.linalg.solve(A, V.T @ x_j)
    return x_j - V @ alpha
