# direct_lingam_hgr_injection.py (revision 2)
"""
Direct LiNGAM augmented with **user‑configurable non‑linear residuals and double‑kernel HGR**.

🎯 **What changed in this revision**
* You can now pass **`deg_res`** (polynomial degree of the residualiser) **and** two separate
  kernel degrees **`kernel_a`** and **`kernel_b`** for the HGR statistic.
* Updated CLI flags: `--deg-res 5 --kernel-a 4 --kernel-b 6`.
* No other behaviour altered; defaults keep the original (4, 4, 4).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from typing import List, Tuple, Optional

import numpy as np
from maxcorr import indicator
from lingam import DirectLiNGAM
from numpy.random import RandomState
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------
# 0.  Helper: polynomial‑ridge residual (same implementation)
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# 1.  Double‑kernel HGR scorer with separate (a, b) degrees
# ---------------------------------------------------------------------

class HGRScore:
    def __init__(self, kernel_a: int = 4, kernel_b: Optional[int] = None, backend: str = "numpy") -> None:
        if kernel_b is None:
            kernel_b = kernel_a
        self._hgr = indicator(
            semantics="hgr",
            algorithm="dk",
            backend=backend,
            kernel_a=kernel_a,
            kernel_b=kernel_b,
        )

    def __call__(self, u: np.ndarray, v: np.ndarray) -> float:
        """Return sqrt(λ₁) ∈ [0,1] – smaller ⇒ more independent."""
        return float(self._hgr.compute(u.ravel(), v.ravel()))

# ---------------------------------------------------------------------
# 2.  DirectLiNGAM subclass with full parametrisation
# ---------------------------------------------------------------------

class DirectLiNGAM_HGR(DirectLiNGAM):
    """Direct LiNGAM + polynomial residuals + double‑kernel HGR.

    Parameters
    ----------
    deg_res : int, default 4
        Degree of the polynomial used to residualise *x_j on x_i*.
    kernel_a, kernel_b : int, optional
        Polynomial degrees of the two kernels inside the HGR statistic.
        If *kernel_b* is None the two degrees are identical.
    backend : {'numpy','torch','tensorflow'}, default 'numpy'
        Computation backend used by **maxcorr**.
    **kwargs : forwarded to parent DirectLiNGAM (random_state, prior knowledge …).
    """

    def __init__(
        self,
        deg_res: int = 4,
        kernel_a: int = 4,
        kernel_b: Optional[int] = None,
        backend: str = "numpy",
        **kwargs,
    ) -> None:
        super().__init__(measure="pwling", **kwargs)  # dummy measure, overridden below
        self._deg_res = deg_res
        self._score = HGRScore(kernel_a=kernel_a, kernel_b=kernel_b, backend=backend)

    # ---------- overrides ---------- #
    def _residual(self, xj, xi):  # noqa: N802  (keep parent signature)
        return poly_ridge_residual(xi, xj, degree=self._deg_res)

    def _search_causal_order(self, X, U):  # noqa: N802
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        X_std = (X - X.mean(0)) / X.std(0)
        scores = []
        for i in Uc:
            xi = X_std[:, i]
            acc = 0.0
            for j in U:
                if i == j:
                    continue
                xj = X_std[:, j]
                rij = xj if (i in Vj and j in Uc) else self._residual(xj, xi)
                acc += self._score(xi, rij) ** 2
            scores.append(acc)
        return Uc[int(np.argmin(scores))]

# ---------------------------------------------------------------------
# 3.  Synthetic DAG, metrics, benchmark – unchanged, only add CLI flags
# ---------------------------------------------------------------------

def simulate_nonlinear_dag(n: int = 1000, rng: RandomState | None = None):
    """Generate a *simpler* 5‑node polynomial DAG (all functions are low‑order polynomials).

    Structure (same shape, no trig/exp):
        x0  → x1  → x3
        x0  → x2  → x3
        x2  → x4
    All noises are Laplace(0, 0.5) to keep the non‑Gaussian assumption.
    """
    if rng is None:
        rng = np.random.RandomState()

    # exogenous root
    x0 = rng.uniform(-2.0, 2.0, size=n)

    # i.i.d. Laplace noises (non‑Gaussian)
    e = rng.laplace(scale=0.5, size=(n, 4))

    # purely *polynomial* mechanisms
    x1 = 1.2 * x0           + e[:, 0]          # linear (degree 1)
    x2 = 0.5 * x0**2        + e[:, 1]          # quadratic
    x3 = 0.8 * x1 + 0.6*x2  + e[:, 2]          # linear mix of parents
    x4 = 0.3 * x2**2        + e[:, 3]          # quadratic of x2

    X = np.column_stack([x0, x1, x2, x3, x4])

    # ground‑truth adjacency (1 if parent → child)
    B = np.zeros((5, 5))
    B[1, 0] = 1  # x0 → x1
    B[2, 0] = 1  # x0 → x2
    B[3, 1] = 1  # x1 → x3
    B[3, 2] = 1  # x2 → x3
    B[4, 2] = 1  # x2 → x4
    return X, B

def structural_hamming_distance(B_true, B_est):
    gt = (np.abs(B_true) > 0).astype(int)
    est = (np.abs(B_est) > 0).astype(int)
    diff = gt ^ est
    return int(diff.sum() - np.multiply(gt, est).sum())

def adjacency_f1(B_true, B_est):
    gt = (np.abs(B_true) > 0).ravel()
    est = (np.abs(B_est) > 0).ravel()
    return f1_score(gt, est)

# -------------------------------------------------------
# 4.  CLI benchmark driver (adds flags)
# -------------------------------------------------------

def run_experiment(runs=10, n=1000, seed=0, deg_res=4, k_a=4, k_b=None):
    rng = RandomState(seed)
    shd_pw, shd_k, shd_h = [], [], []
    f1_pw, f1_k, f1_h = [], [], []
    for _ in range(runs):
        X, B = simulate_nonlinear_dag(n, rng)
        m_pw = DirectLiNGAM(random_state=rng.randint(2**31), measure="pwling").fit(X)
        m_k  = DirectLiNGAM(random_state=rng.randint(2**31), measure="kernel").fit(X)
        m_h  = DirectLiNGAM_HGR(deg_res=deg_res, kernel_a=k_a, kernel_b=k_b, random_state=rng.randint(2**31)).fit(X)
        for lst, f in [(shd_pw, structural_hamming_distance), (f1_pw, adjacency_f1)]:
            lst.append(f(B, m_pw.adjacency_matrix_))
        for lst, f in [(shd_k, structural_hamming_distance), (f1_k, adjacency_f1)]:
            lst.append(f(B, m_k.adjacency_matrix_))
        for lst, f in [(shd_h, structural_hamming_distance), (f1_h, adjacency_f1)]:
            lst.append(f(B, m_h.adjacency_matrix_))
    print(f"\nMean over {runs} runs (n={n}):")
    for name, shd, f1 in [
        ("PW-ling", np.mean(shd_pw), np.mean(f1_pw)),
        ("Kernel-MI", np.mean(shd_k), np.mean(f1_k)),
        ("HGR-DK", np.mean(shd_h), np.mean(f1_h)),
    ]:
        print(f"{name:<11}: SHD={shd:5.2f}  F1={f1:5.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DirectLiNGAM + HGR with user degrees")
    ap.add_argument("--bench", type=int, metavar="N", help="run N Monte‑Carlo simulations")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--deg-res", type=int, default=4, help="degree of regression polynomial")
    ap.add_argument("--kernel-a", type=int, default=4, help="degree of HGR kernel a")
    ap.add_argument("--kernel-b", type=int, default=None, help="degree of HGR kernel b")
    args = ap.parse_args()
    if args.bench:
        run_experiment(args.bench, args.samples, deg_res=args.deg_res, k_a=args.kernel_a, k_b=args.kernel_b)
    else:
        X, _ = simulate_nonlinear_dag(args.samples, np.random.RandomState(42))
        model = DirectLiNGAM_HGR(deg_res=args.deg_res, kernel_a=args.kernel_a, kernel_b=args.kernel_b, random_state=42)
        model.fit(X)
        print("Causal order:", model.causal_order_)
        print("Adjacency matrix:\n", np.round(model.adjacency_matrix_, 3))
