#!/usr/bin/env python
"""
Aggregate results by kernel pair (ka, kb) ***using the new HGR‑based
independence metric*** (`hgr_x_epshat`).

For each combination we report:
    n_runs      – total datasets evaluated
    pass_rate   – share of runs that satisfy (acc == 1 **and** HGR < thr)
    acc_sum     – number of runs with correct direction (acc == 1)
    indep_mean  – mean HGR(X, ε̂) across runs
    indep_max   – max  HGR across runs
    fail_acc    – runs with wrong direction (acc == 0)
    fail_both   – runs where acc == 0 **and** HGR ≥ thr

Run after regenerating `results/independence/residuals.csv` with the
updated independence checker:
    python scripts/aggregate_results.py --thr 0.06
"""
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESID_CSV = ROOT / "results" / "independence" / "residuals.csv"
OUT_CSV   = ROOT / "results" / "summary_by_kernel.csv"

# ----------------------------------------------------------------------

def main(thr: float):
    df = pd.read_csv(RESID_CSV)

    # column is now 'hgr_x_epshat'
    df["abs_rho"] = df["hgr_x_epshat"].abs()

    # Flags
    df["pass"]      = (df["acc"] == 1) & (df["abs_rho"] < thr)
    df["fail_acc"]  = (df["acc"] == 0)
    df["fail_both"] = (df["acc"] == 0) & (df["abs_rho"] >= thr)

    summary = (
        df.groupby(["ka", "kb"], as_index=False)
          .agg(
              n_runs     = ("dataset", "count"),
            #  pass_rate  = ("pass", "mean"),
              acc_sum    = ("acc", "sum"),
              indep_mean = ("abs_rho", "mean"),
              indep_max  = ("abs_rho", "max"),
              fail_acc   = ("fail_acc", "sum"),
              fail_both  = ("fail_both", "sum"),
          )
          .sort_values(["ka", "kb"])
    )

    summary.to_csv(OUT_CSV, index=False)
    print(f"✔  summary written to {OUT_CSV.relative_to(ROOT)}  (thr={thr})")

# ----------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.06,
                    help="HGR threshold for independence (default 0.06)")
    args = ap.parse_args()
    main(args.thr)
