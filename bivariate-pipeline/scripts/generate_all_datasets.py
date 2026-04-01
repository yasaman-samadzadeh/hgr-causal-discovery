"""
Generate every dataset combination defined in configs/dataset_grid.yaml
Supports dependent (causal), independent (non-causal), and noiseless variants.
"""
import itertools, yaml, pathlib
from types import SimpleNamespace
from hgrlingam.data import make_synthetic, make_independent

ROOT = pathlib.Path(__file__).resolve().parents[1]
grid = yaml.safe_load((ROOT / "configs" / "dataset_grid.yaml").read_text())

# Define the types
independent_f_types = {"noise_only", "shuffled", "random_pair"}
noiseless_f_types = {"cubic_noiseless", "quintic_noiseless", "sine_noiseless"}

# Identify keys
noise_key = "noise_scale"
noise_scales = grid.get(noise_key, [0.3])
list_keys = [k for k, v in grid.items() if isinstance(v, list) and k != noise_key]
scalar_k  = {k: v for k, v in grid.items() if k not in list_keys and k != noise_key}

# Generate combinations
prod = []
for vals in itertools.product(*[grid[k] for k in list_keys]):
    partial_combo = dict(zip(list_keys, vals))

    f_type = partial_combo["f_type"]

    if f_type in noiseless_f_types:
        combo = {**partial_combo, **scalar_k, noise_key: None}
        prod.append(combo)
    else:
        for ns in noise_scales:
            combo = {**partial_combo, **scalar_k, noise_key: ns}
            prod.append(combo)

# Generate datasets
count = 0
for combo in prod:
    f_type = combo["f_type"]
    noise_type = combo["noise_type"]
    x_dist = combo["x_dist"]
    seed = combo["seed"]
    noise_scale = combo.get("noise_scale")

    # Set name based on whether it's noiseless
    if f_type in noiseless_f_types:
        combo["name"] = f'{grid["name_prefix"]}_{f_type}_{noise_type}_{x_dist}_{seed}'
    else:
        combo["name"] = f'{grid["name_prefix"]}_{f_type}_{noise_type}_{noise_scale}_{x_dist}_{seed}'

    ns = SimpleNamespace(**combo)

    # Dispatch to generator
    if f_type in independent_f_types:
        make_independent(ns)
    else:
        make_synthetic(ns)

    count += 1

print(f"✔ generated {count} datasets")
