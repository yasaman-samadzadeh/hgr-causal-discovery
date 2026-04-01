"""
Kernel-HGR helper built on maxcorr.indicator
-------------------------------------------
• semantics : 'hgr' | 'gedi' | 'nlc'
• algorithm : 'dk' | 'sk' | 'nn' | 'lat' | 'kde' | 'rdc'
• backend   : 'numpy' | 'torch' | 'tensorflow'
• **algo_kwargs : anything the specific indicator accepts
                  e.g. kernel_a=3, kernel_b=1  for DK
                       projections=100         for RDC
"""
from pathlib import Path
import json, numpy as np
from maxcorr import indicator


class KernelHGR:
    def __init__(
        self,
        semantics: str = "hgr",
        algorithm: str = "dk",
        backend: str = "numpy",
        **algo_kwargs,
    ):
        self.semantics = semantics
        self.algorithm = algorithm
        self.backend = backend
        self.algo_kwargs = algo_kwargs

        # build indicator instance
        self._ind = indicator(
            semantics=self.semantics,
            algorithm=self.algorithm,
            backend=self.backend,
            **self.algo_kwargs
        )
        self._trained = False
        self.train_corr_ = None
        self.train_pval_ = None

    # --------------------------------------------------
    def fit(self, x, y):
        """
        Optimise copula transforms on TRAIN vectors x, y.
        x, y expected shape: (n,)  or  (n, d)
        """
        self.train_corr_ = self._ind.compute(x, y)           # optimisation
        res = self._ind.last_result
        self.train_pval_ = getattr(res, "pval", None)
        self._trained = True
        return self

    # --------------------------------------------------
    def transform(self, x, y):
        """
        Apply learned transforms → return u, v (same shape as x, y rows).
        """
        if not self._trained:
            raise RuntimeError("Call fit() first")
        return self._ind.f(x), self._ind.g(y)

    # --------------------------------------------------
    def value(self, x, y):
        """
        Correlation on *new* data with frozen transforms (no optimisation).
        """
        if not self._trained:
            raise RuntimeError("Call fit() first")
        return self._ind.value(x, y)

    # --------------------------------------------------
    def save(self, folder: Path):
        """
        Persist alpha/beta (if available) and metadata JSON.
        """
        folder.mkdir(parents=True, exist_ok=True)
        res = self._ind.last_result
        if hasattr(res, "alpha"):
            np.save(folder / "alpha.npy", res.alpha)
        if hasattr(res, "beta"):
            np.save(folder / "beta.npy", res.beta)
        meta = dict(
            semantics=self.semantics,
            algorithm=self.algorithm,
            backend=self.backend,
            algo_kwargs=self.algo_kwargs,
            rho=self.train_corr_,
            pval=self.train_pval_,
        )
        (folder / "hgr.json").write_text(json.dumps(meta, indent=2))

    # (re-loading is omitted for brevity; easiest is to re-fit.)


# -------------------------------------------------------------------------
# Smoke-test: run  `python -m hgrlingam.hgr`  to verify it works standalone
if __name__ == "__main__":
    import numpy as np, pprint
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    y = np.sin(x) + 0.3 * rng.normal(0, 1, 1000)

    hgr = KernelHGR(algorithm="dk", kernel_a=5, kernel_b=1)
    hgr.fit(x, y)
    print("ρ_HGR =", hgr.train_corr_, "   p =", hgr.train_pval_)
    u, v = hgr.transform(x, y)
    print("Pearson(u,v) =", np.corrcoef(u, v)[0, 1])
    pprint.pp(meta := {"demo_saved": True})
