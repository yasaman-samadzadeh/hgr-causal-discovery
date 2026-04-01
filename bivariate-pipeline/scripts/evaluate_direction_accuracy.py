import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the extended residuals.csv (with f_type, noise_type, is_independent, is_noiseless)
residuals_path = Path("results/independence/residuals.csv")
df = pd.read_csv(residuals_path)

# Add group classification from metadata
def classify(row):
    if row.get("is_independent", False):
        return "independent"
    elif row.get("is_noiseless", False):
        return "causal_noiseless"
    else:
        return "causal_with_noise"

df["group"] = df.apply(classify, axis=1)
df["kernel"] = df["ka"].astype(str) + "-" + df["kb"].astype(str)

# --- STEP 1: Compute overall accuracy across all data for each kernel ---
kernel_global_accuracy = (
    df.groupby("kernel")
    .agg(total=("dataset", "count"), correct=("acc", "sum"))
    .assign(overall_accuracy=lambda d: d["correct"] / d["total"])
    .sort_values("overall_accuracy", ascending=False)
    .head(3)
    .reset_index()
)

top_kernels = kernel_global_accuracy["kernel"].tolist()

# --- STEP 2: Keep only data from top kernels (across all groups) ---
df_top = df[df["kernel"].isin(top_kernels)].copy()

# --- STEP 3: Compute accuracy per group for each of those kernels ---
summary = (
    df_top.groupby(["kernel", "group"])
    .agg(
        total_datasets=("dataset", "count"),
        correct_predictions=("acc", "sum"),
        accuracy_rate=("acc", "mean")
    )
    .reset_index()
)

# --- STEP 4: Prepare pivot table for grouped bar plot ---
groups = ["causal_with_noise", "causal_noiseless", "independent"]
pivot = summary.pivot(index="kernel", columns="group", values="accuracy_rate").fillna(0)
for group in groups:
    if group not in pivot.columns:
        pivot[group] = 0
pivot = pivot[groups]  # Ensure consistent column order

# --- STEP 5: Plot grouped bar chart ---
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.25
x = range(len(pivot))

for i, group in enumerate(groups):
    ax.bar(
        [xi + i * width for xi in x],
        pivot[group],
        width=width,
        label=group.replace("_", " ").title(),
        color={"causal_with_noise": "green", "causal_noiseless": "orange", "independent": "red"}[group]
    )

# Styling
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(pivot.index)
ax.set_ylabel("Accuracy Rate")
ax.set_title("True Direction Accuracy by Kernel and Dataset Group (Top 3 Overall Kernels)")
ax.legend()
plt.tight_layout()
plt.show()
