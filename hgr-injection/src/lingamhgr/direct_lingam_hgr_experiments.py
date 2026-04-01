# direct_lingam_hgr_experiments.py (revision 4 – plots show **true vs. predicted**)
"""
This revision adds *side‑by‑side* DAG plots:
• Left panel  = ground truth • Right panel = graph estimated by **HGR‑DK**

Usage unchanged — just pass `--plots out_dir`. Each PNG now contains two
subplots so you can eyeball mistakes at a glance.
"""

from __future__ import annotations
import argparse, pathlib, numpy as np
from typing import Tuple
from numpy.random import RandomState
from sklearn.metrics import f1_score
from maxcorr import indicator
try:
    import networkx as nx, matplotlib.pyplot as plt
except ImportError:
    nx = plt = None
from lingam import DirectLiNGAM

# ───────── Residual & HGR helpers ───────────────────────────────────

def poly_ridge_residual(x_i, x_j, degree=4, lam=None):
    x_i, x_j = x_i.ravel(), x_j.ravel()
    V = np.vander(x_i, degree + 1, increasing=True)
    if lam is None:
        lam = 1e-3 * np.var(x_j)
    alpha = np.linalg.solve(V.T @ V + lam * np.eye(degree + 1), V.T @ x_j)
    return x_j - V @ alpha

class HGRScore:
    def __init__(self, k_a=4, k_b=None, backend="numpy"):
        if k_b is None:
            k_b = k_a
        self._hgr = indicator(semantics="hgr", algorithm="dk", backend=backend,
                              kernel_a=k_a, kernel_b=k_b)
    def __call__(self, u, v):
        return float(self._hgr.compute(u.ravel(), v.ravel()))

# ───────── DirectLiNGAM‑HGR subclass ────────────────────────────────

class DirectLiNGAM_HGR(DirectLiNGAM):
    def __init__(self, deg_res=4, kernel_a=4, kernel_b=None, backend="numpy", **kw):
        super().__init__(measure="pwling", **kw)
        self._deg_res = deg_res
        self._score   = HGRScore(kernel_a, kernel_b, backend)
    def _residual(self, xj, xi):
        return poly_ridge_residual(xi, xj, self._deg_res)
    def _search_causal_order(self, X, U):
        Uc, Vj = self._search_candidate(U)
        if len(Uc)==1:
            return Uc[0]
        Xs = (X - X.mean(0)) / X.std(0)
        scores = []
        for i in Uc:
            xi = Xs[:, i]
            acc = 0.0
            for j in U:
                if i == j: continue
                xj = Xs[:, j]
                rij = xj if (i in Vj and j in Uc) else self._residual(xj, xi)
                acc += self._score(xi, rij)**2
            scores.append(acc)
        return Uc[int(np.argmin(scores))]

# ───────── Random polynomial DAG generator ──────────────────────────

def random_poly_sem(p: int, sparsity: float, max_deg: int, n: int, rng: RandomState
                    ) -> Tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(p)
    B = np.zeros((p,p))
    for j in range(1,p):
        for i in range(j):
            if rng.rand() < sparsity:
                B[order[j], order[i]] = 1
    X = np.zeros((n,p))
    E = rng.laplace(scale=0.5, size=(n,p))
    roots = np.where(B.sum(1)==0)[0]
    X[:, roots] = rng.uniform(-2,2,size=(n,len(roots)))
    for idx in order:
        parents = np.where(B[idx]==1)[0]
        if parents.size==0:
            continue
        val = np.zeros(n)
        for pa in parents:
            deg  = rng.randint(1, max_deg+1)
            sign = rng.choice((-1,1))
            val += rng.uniform(0.3,1.0) * sign * (X[:,pa] ** deg)
        X[:, idx] = val + E[:, idx]
    return X, B

# ───────── Metrics & plotting ───────────────────────────────────────

def shd(Bt, Be):
    diff = (np.abs(Bt)>0) ^ (np.abs(Be)>0)
    return int(diff.sum() - np.multiply(np.abs(Bt)>0, np.abs(Be)>0).sum())

def adj_f1(Bt, Be):
    return f1_score((np.abs(Bt)>0).ravel(), (np.abs(Be)>0).ravel())

def plot_true_vs_pred(B_true, B_pred, path):
    if nx is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    for ax, B, title in zip(axes, (B_true,B_pred), ("True DAG","Predicted DAG")):
        g = nx.DiGraph()
        g.add_nodes_from(range(B.shape[0]))
        g.add_edges_from([(i,j) for i,j in zip(*np.where(B==1))])
        pos = nx.spring_layout(g, seed=0)
        nx.draw(g, pos, with_labels=True, node_size=600, arrowsize=12, ax=ax)
        ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ───────── Single run helper ────────────────────────────────────────

def run_single(X, B, deg_res, k_a, k_b, rng):
    pw  = DirectLiNGAM(random_state=rng.randint(2**31), measure="pwling").fit(X)
    hs  = DirectLiNGAM(random_state=rng.randint(2**31), measure="kernel").fit(X)
    hgr = DirectLiNGAM_HGR(deg_res, k_a, k_b, random_state=rng.randint(2**31)).fit(X)
    return (
        [
            (shd(B, pw.adjacency_matrix_),  adj_f1(B, pw.adjacency_matrix_)),
            (shd(B, hs.adjacency_matrix_),  adj_f1(B, hs.adjacency_matrix_)),
            (shd(B, hgr.adjacency_matrix_), adj_f1(B, hgr.adjacency_matrix_)),
        ],
        hgr.adjacency_matrix_,  # return for plotting
    )

# ───────── CLI entry ────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--sparsity", type=float, default=0.3)
    ap.add_argument("--max-degree", type=int, default=2)
    ap.add_argument("--deg-res", type=int, default=4)
    ap.add_argument("--kernel-a", type=int, default=4)
    ap.add_argument("--kernel-b", type=int, default=None)
    ap.add_argument("--batch", type=int, metavar="N")
    ap.add_argument("--plots", type=str)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = RandomState(args.seed)

    if args.batch:
        res = np.zeros((args.batch, 3, 2))
        out_dir = pathlib.Path(args.plots) if args.plots else None
        if out_dir: out_dir.mkdir(parents=True, exist_ok=True)
        for r in range(args.batch):
            X,B = random_poly_sem(args.nodes, args.sparsity, args.max_degree, args.samples, rng)
            metrics, B_pred = run_single(X,B,args.deg_res,args.kernel_a,args.kernel_b,rng)
            res[r] = metrics
            if out_dir: plot_true_vs_pred(B, (np.abs(B_pred)>0).astype(int), out_dir / f"dag_{r}.png")
        labels = ["PW-ling","HSIC-kernel","HGR-DK"]
        print(f"\nMean over {args.batch} DAGs (p={args.nodes}, n={args.samples})\n")
        for lbl,(shd_m,f1_m) in zip(labels, res.mean(0)):
            print("\nSingle-DAG results (SHD, F1):")
            print("PW-ling     :", pw[0], pw[1])
            print("HSIC-kernel :", hs[0], hs[1])
            print("HGR-DK      :", hgr[0], hgr[1])

    else:
        X, B = random_poly_sem(
            p=args.nodes, sparsity=args.sparsity,
            max_deg=args.max_degree, n=args.samples, rng=rng
        )
        metrics, B_pred = run_single(
            X, B, args.deg_res, args.kernel_a, args.kernel_b, rng
        )

        if args.plots:
            out = pathlib.Path(args.plots)
            out.mkdir(parents=True, exist_ok=True)
            plot_true_vs_pred(B, (np.abs(B_pred) > 0).astype(int),
                              out / "single_dag.png")
            print(f"Saved DAG plot → {out/'single_dag.png'}")

        (pw, hs, hgr) = metrics
        print("\nSingle-DAG results (SHD, F1):")
        print(f"PW-ling     : {pw[0]:6.2f}, {pw[1]:.3f}")
        print(f"HSIC-kernel : {hs[0]:6.2f}, {hs[1]:.3f}")
        print(f"HGR-DK      : {hgr[0]:6.2f}, {hgr[1]:.3f}")

        print("\nGround-truth adjacency matrix:\n", B.astype(int))

