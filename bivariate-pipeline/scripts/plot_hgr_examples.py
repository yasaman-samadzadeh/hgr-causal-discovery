#!/usr/bin/env python
"""
Generate (x,y) vs (u,v) scatter PNGs for **all** HGR combos and datasets.

Run:
    python scripts/plot_hgr_all.py
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tqdm
from hgrlingam.data import load_dataset

ROOT        = Path(__file__).resolve().parents[1]
HGR_ROOT    = ROOT / "results" / "hgr"
PLOT_ROOT   = ROOT / "plots" / "hgr"
PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def plot_pair(x, y, ax, title):
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")


def save_plot(hgr_tag: str, ds_name: str):
    # raw (unstandardised) train data
    df_raw = pd.read_csv(ROOT / "datasets" / ds_name / "train.csv")
    x_raw, y_raw = df_raw["x"].values, df_raw["y"].values

    # transformed data
    z_path = HGR_ROOT / hgr_tag / ds_name / "Z_train.npy"
    if not z_path.exists():
        tqdm.tqdm.write(f"⚠️  {z_path} missing, skip")
        return
    u, v = np.load(z_path).T

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), tight_layout=True)
    plot_pair(x_raw, y_raw, axes[0], "original (x, y)")
    plot_pair(u, v, axes[1], "HGR (u, v)")
    fig.suptitle(ds_name, fontsize=8)

    out_dir = PLOT_ROOT / hgr_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{ds_name}.png", dpi=140)
    plt.close(fig)


def main():
    hgr_tags = sorted(p.name for p in HGR_ROOT.iterdir() if p.is_dir())

    for tag in hgr_tags:
        ds_names = sorted(p.name for p in (HGR_ROOT / tag).iterdir() if p.is_dir())
        tqdm.tqdm.write(f"\n## plotting {tag}  (datasets={len(ds_names)}) ##")
        for ds in tqdm.tqdm(ds_names, desc=tag):
            save_plot(tag, ds)


if __name__ == "__main__":
    main()
