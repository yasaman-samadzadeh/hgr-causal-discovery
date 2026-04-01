import pandas as pd
import json
from pathlib import Path

# Load residual evaluation
res_file = Path("results/independence/residuals.csv")
df = pd.read_csv(res_file)

# Load metadata for each dataset
def get_meta_info(dataset):
    meta_file = Path("datasets") / dataset / "meta.json"
    try:
        meta = json.loads(meta_file.read_text())
        return pd.Series({
            "f_type": meta.get("f_type", "unknown"),
            "is_independent": meta.get("is_independent", False),
            "is_noiseless": meta.get("is_noiseless", False)
        })
    except:
        return pd.Series({"f_type": "unknown", "is_independent": False, "is_noiseless": False})

meta_df = df["dataset"].apply(get_meta_info)
df = pd.concat([df, meta_df], axis=1)
print(meta_df.head())
print(meta_df.dtypes)
# Mark success
threshold = 0.06
df["worked"] = (df["acc"] == 1) & (df["hgr_x_epshat"] < threshold)

# --- Step 1: Select top 3 kernels by total success rate
kernel_summary = (
    df.groupby(["ka", "kb"])
      .agg(total_runs=("dataset", "count"), successes=("worked", "sum"))
      .assign(success_rate=lambda d: d["successes"] / d["total_runs"])
      .sort_values("success_rate", ascending=False)
)

top3 = kernel_summary.head(3).reset_index()
print("\n🔥 Top 3 kernel combinations:")
print(top3)

# --- Step 2: Analyze performance of top kernels
top_kernels = list(top3[["ka", "kb"]].itertuples(index=False, name=None))
top_df = df[df[["ka", "kb"]].apply(tuple, axis=1).isin(top_kernels)]

# Summary by kernel and data type
summary = (
    top_df.groupby(["ka", "kb", "f_type", "is_independent", "is_noiseless"])
    .agg(
        n_runs=("dataset", "count"),
        n_correct=("acc", "sum"),
        n_passed=("worked", "sum"),
        avg_residual_hgr=("hgr_x_epshat", "mean")
    )
    .reset_index()
)

# Save results
summary_path = Path("results/independence/top_kernel_analysis.csv")
summary.to_csv(summary_path, index=False)
print(f"\n📊 Analysis saved to {summary_path}")
