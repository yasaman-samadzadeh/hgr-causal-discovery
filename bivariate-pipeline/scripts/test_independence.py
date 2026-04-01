#!/usr/bin/env python
"""
Independence checker using kernel‑HGR for raw noise or residuals.

Usage:
  python scripts/test_independence.py --mode raw      # HGR(x, ε)
  python scripts/test_independence.py --mode residual # HGR(x, ε̂)

Outputs:
  results/independence/raw_noise.csv
  results/independence/residuals.csv
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from hgrlingam.hgr import KernelHGR

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
LINGAM_ROOT = ROOT / "results" / "lingam"
OUT_DIR = ROOT / "results" / "independence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Kernel-HGR setup
HGR_TEST = KernelHGR(algorithm="dk", kernel_a=5, kernel_b=1)

def hgr_corr(x: np.ndarray, y: np.ndarray) -> float:
    return HGR_TEST._ind.compute(x.reshape(-1, 1), y.reshape(-1, 1))

# Structural functions used in raw mode
FUNCS: Dict[str, callable] = {
    "cubic": lambda x: 0.489 * (x + 0.3 * x ** 3),
    "quintic": lambda x: 0.268 * (x + 0.1 * x ** 5),
    "sine": lambda x: 1.209 * (
        np.sin(1.2 * x) + 0.3 * np.cos(0.8 * x) + 0.25 * np.tanh(0.8 * x)
    ),
}

# --- Raw Mode ---------------------------------------------------------

def raw_noise_rows():
    rows = []
    for ds in sorted(d.name for d in DATASETS_DIR.iterdir() if d.is_dir()):
        folder = DATASETS_DIR / ds
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta["f_type"] not in FUNCS:
            print(f"⏭️  Skipping non-causal dataset: {ds}")
            continue

        f = FUNCS[meta["f_type"]]
        df = pd.read_csv(folder / "train.csv")
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        eps = y - f(x)
        rho = hgr_corr(x, eps)

        rows.append({
            "dataset": ds,
            "hgr_x_eps": rho,
            "f_type": meta.get("f_type", "unknown"),
            "noise_type": meta.get("noise_type", "unknown"),
            "is_independent": meta.get("is_independent", False),
            "is_noiseless": meta.get("is_noiseless", False)
        })
    return rows

# --- Residual Mode ----------------------------------------------------

def parse_kernel(tag: str) -> Tuple[int, int]:
    bits = tag.split("_")
    return int(bits[0][2:]), int(bits[1][2:])

def residual_rows():
    rows = []
    for res_dir in sorted(LINGAM_ROOT.iterdir()):
        if not res_dir.is_dir() or not res_dir.name.endswith("_direct"):
            continue
        tag = res_dir.name[:-7]  # strip _direct
        ka, kb = parse_kernel(tag)

        for ds_dir in res_dir.iterdir():
            fpath = ds_dir / "residuals_test.npy"
            if not fpath.exists():
                continue

            arr = np.load(fpath, allow_pickle=True)
            eps_hat = arr[:, 1] if arr.dtype != object else arr.item()["y"]

            ds = ds_dir.name
            x = pd.read_csv(DATASETS_DIR / ds / "test.csv")["x"].to_numpy()
            rho = hgr_corr(x, eps_hat)
            acc = json.load(open(ds_dir / "metrics.json"))["direction_acc"]

            # Load meta.json
            meta = {}
            meta_path = DATASETS_DIR / ds / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())

            rows.append({
                "dataset": ds,
                "ka": ka,
                "kb": kb,
                "hgr_x_epshat": rho,
                "acc": acc,
                "f_type": meta.get("f_type", "unknown"),
                "noise_type": meta.get("noise_type", "noiseless" if meta.get("is_noiseless") else "unknown"),
                "is_independent": meta.get("is_independent", False),
                "is_noiseless": meta.get("is_noiseless", False)
            })
    return rows

# --- Main -------------------------------------------------------------

def main(mode: str):
    if mode == "raw":
        rows = raw_noise_rows()
        out = OUT_DIR / "raw_noise.csv"
    else:
        rows = residual_rows()
        out = OUT_DIR / "residuals.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"✔ saved {len(rows)} rows → {out.relative_to(ROOT)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "residual"], default="raw")
    main(ap.parse_args().mode)
