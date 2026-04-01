#!/usr/bin/env python
"""
Fit LiNGAM on each HGR‑transformed **train** split *once*, then compute residuals
on the **test** split with the **same B ̂**. This avoids the empty residual files
caused by re‑fitting on the test data.

Folder layout (unchanged)
─────────────────────────
results/lingam/<hgr_tag>_<lingamAlgo>/<dataset>/
    B_est.npy
    residuals_train.npy
    residuals_test.npy
    ordering.json
    metrics.json   (direction_acc)
"""
from pathlib import Path
import argparse, json, itertools, tqdm, yaml, numpy as np, pandas as pd
from hgrlingam.causal import run_lingam
from hgrlingam.data   import DATASETS

ROOT        = Path(__file__).resolve().parents[1]
HGR_ROOT    = ROOT / "results" / "hgr"
LINGAM_ROOT = ROOT / "results" / "lingam"
DATASETS_ROOT = ROOT/ "datasets"
LINGAM_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------

def load_lingam_grid(path: Path):
    cfg = yaml.safe_load(open(path))
    cfg["algorithm"]    = [str(a) for a in cfg.get("algorithm", ["direct"])]
    cfg["random_state"] = list(cfg.get("random_state", [0]))
    return cfg

def direction_accuracy(B_est: np.ndarray, adj_matrix_path: Path) -> int:
    """
    Compare LiNGAM B_est (B[i,j] means j → i) to ground-truth adjacency matrix
    (adj[i,j] means i → j).
    """
    true_adj = pd.read_csv(adj_matrix_path, header=None).values

    # Convert B_est to binary matrix: j → i → transpose → i → j
    pred_adj = (np.abs(B_est) > 1e-6).astype(int).T

    return int((pred_adj == true_adj).all())

# def direction_accuracy(ordering):
#     """After HGR linearisation, ground‑truth arrow is always 0 → 1."""
#     return int(list(ordering) == [0, 1])

# -------------------------------------------------------------------------

def main(grid_yaml: str):
    cfg = load_lingam_grid(Path(grid_yaml))

    hgr_tags   = sorted(p.name for p in HGR_ROOT.iterdir() if p.is_dir())
    lingam_alg = cfg["algorithm"]
    seeds      = cfg["random_state"]

    for tag, algo, rs in itertools.product(hgr_tags, lingam_alg, seeds):
        hgr_dir  = HGR_ROOT / tag
        out_root = LINGAM_ROOT / f"{tag}_{algo}"
        out_root.mkdir(parents=True, exist_ok=True)

        tqdm.tqdm.write(f"\n## LiNGAM={algo}  on  HGR={tag}  (seed={rs}) ##")
        for ds_dir in tqdm.tqdm(sorted(hgr_dir.iterdir()), desc=f"{tag}_{algo}"):
            if not (ds_dir / "Z_train.npy").exists():
                tqdm.tqdm.write(f"⚠️  {ds_dir.name}: missing Z_train → skipped")
                continue

            dataset_name = ds_dir.name
            dataset_root = DATASETS_ROOT / dataset_name
            adj_path = dataset_root / "adj_matrix.csv"

            if not adj_path.exists():
                tqdm.tqdm.write(f"⚠️  {dataset_name}: missing adj_matrix.csv → skipped")
                continue

            Z_tr = np.load(ds_dir / "Z_train.npy")
            Z_te = np.load(ds_dir / "Z_test.npy")

            # ---- fit LiNGAM ONCE on TRAIN --------------------------------
            ordering, B_est, resid_tr = run_lingam(
                Z_tr, algo_name=algo, random_state=rs
            )

            # ---- compute TEST residuals with SAME B_est -------------------
            I_minus_B = np.eye(B_est.shape[0]) - B_est
            resid_te  = (I_minus_B @ Z_te.T).T

            # ensure numeric arrays (no pickled objects)
            resid_tr = np.asarray(resid_tr, dtype=np.float64)
            resid_te = np.asarray(resid_te, dtype=np.float64)
            B_est    = np.asarray(B_est,    dtype=np.float64)

            # ✅ use new version
            acc = direction_accuracy(B_est, adj_path)

            # Save results
            dest = out_root / dataset_name
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest / "B_est.npy", B_est)
            np.save(dest / "residuals_train.npy", resid_tr)
            np.save(dest / "residuals_test.npy",  resid_te)
            json.dump([int(i) for i in ordering], open(dest / "ordering.json", "w"))
            json.dump({"direction_acc": acc}, open(dest / "metrics.json", "w"), indent=2)

            tqdm.tqdm.write(f"{dataset_name:50s}  acc={acc}")


# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/lingam.yaml",
                        help="YAML with 'algorithm' (list) and 'random_state' (list)")
    args = parser.parse_args()
    main(args.cfg)
