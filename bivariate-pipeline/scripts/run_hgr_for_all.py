#!/usr/bin/env python
"""
Compute kernel-HGR for every dataset and for every combination of

    kernel_a × kernel_b × backend × algorithm

declared in configs/hgr.yaml  (lists allowed).

Outputs:
results/hgr/ka{a}_kb{b}_{backend}_{algo}/{dataset}/
    Z_train.npy, Z_test.npy, hgr.json, alpha.npy, beta.npy
"""
from pathlib import Path
import argparse, json, itertools, yaml, tqdm, numpy as np
from types import SimpleNamespace
from hgrlingam.data import load_dataset
from hgrlingam.hgr  import KernelHGR

ROOT         = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
RESULTS_ROOT = ROOT / "results" / "hgr"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
def load_grid(path: Path):
    """Return a dict with list-fields cast to plain Python lists."""
    cfg = yaml.safe_load(open(path))
    cfg["algorithm"] = [str(a) for a in cfg.get("algorithm", ["dk"])]
    cfg["backend"]   = [str(b) for b in cfg.get("backend",  ["numpy"])]
    cfg["kernel_a"]  = list(cfg.get("kernel_a", [3]))
    cfg["kernel_b"]  = list(cfg.get("kernel_b", [3]))
    return cfg


# ----------------------------------------------------------------------
def process_dataset(ds_name: str, cfg_obj, algo_kwargs, out_dir: Path):
    """Fit HGR for a single dataset and save outputs."""
    X_tr, Y_tr = load_dataset(ds_name, "train")
    X_te, Y_te = load_dataset(ds_name, "test")

    hgr = KernelHGR(
        semantics=cfg_obj.semantics,
        algorithm=cfg_obj.algorithm,
        backend  =cfg_obj.backend,
        **algo_kwargs,
    )
    hgr.fit(X_tr.squeeze(), Y_tr.squeeze())

    u_tr, v_tr = hgr.transform(X_tr.squeeze(), Y_tr.squeeze())
    u_te, v_te = hgr.transform(X_te.squeeze(), Y_te.squeeze())

    folder = out_dir / ds_name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "Z_train.npy", np.c_[u_tr, v_tr])
    np.save(folder / "Z_test.npy",  np.c_[u_te, v_te])

    res = hgr._ind.last_result
    if hasattr(res, "alpha"):
        np.save(folder / "alpha.npy", res.alpha)
    if hasattr(res, "beta"):
        np.save(folder / "beta.npy", res.beta)

    meta = dict(
        rho_train=hgr.train_corr_,
        rho_test =hgr.value(X_te.squeeze(), Y_te.squeeze()),
        semantics=cfg_obj.semantics,
        algorithm=cfg_obj.algorithm,
        backend  =cfg_obj.backend,
        algo_kwargs=algo_kwargs,
    )
    (folder / "hgr.json").write_text(json.dumps(meta, indent=2))
    return hgr.train_corr_


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/hgr.yaml",
                        help="Path to YAML grid with kernel_a/b, backend, algorithm")
    args = parser.parse_args()

    grid = load_grid(Path(args.cfg))
    kernels_a = grid["kernel_a"]
    kernels_b = grid["kernel_b"]
    backends  = grid["backend"]
    algos     = grid["algorithm"]
    datasets  = sorted(d.name for d in DATASETS_DIR.iterdir() if d.is_dir())

    for ka, kb, backend, algo in itertools.product(kernels_a, kernels_b, backends, algos):
        tag = f"ka{ka}_kb{kb}_{backend}_{algo}"
        out_root = RESULTS_ROOT / tag
        out_root.mkdir(parents=True, exist_ok=True)

        cfg_obj = SimpleNamespace(
            semantics = grid.get("semantics", "hgr"),
            algorithm = algo,
            backend   = backend,
        )
        extra = grid.get("extras", {}).get(algo, {})
        algo_kwargs = dict(kernel_a=ka, kernel_b=kb, **extra)

        tqdm.tqdm.write(f"\n## HGR: ka={ka}, kb={kb}, backend={backend}, algo={algo} ##")
        for ds in tqdm.tqdm(datasets, desc=tag):
            rho = process_dataset(ds, cfg_obj, algo_kwargs, out_root)
            tqdm.tqdm.write(f"{ds:35s}  ρ_train={rho:.3f}")
