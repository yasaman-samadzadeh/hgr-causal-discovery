from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from causalgen import Generator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS  = REPO_ROOT / "datasets"
DATASETS.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# INDEPENDENT DATASETS
# ----------------------------------------------------------------------

def make_independent(cfg: SimpleNamespace | dict) -> Path:
    """Create datasets with no causal link between x and y."""
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)

    rng = np.random.default_rng(cfg.seed)
    n_train, n_test = cfg.n_train, cfg.n_test
    noise_scale = cfg.noise_scale

    def gen_noise(n, kind):
        if kind == "logistic":
            return rng.logistic(0, noise_scale * (np.sqrt(3) / np.pi), size=n)
        elif kind == "laplace":
            return rng.laplace(0, noise_scale, size=n)
        else:
            raise ValueError(f"Unsupported noise type: {kind}")

    # x is always standard normal
    x_train = rng.normal(0, 1, n_train)
    x_test  = rng.normal(0, 1, n_test)

    f = lambda x: 0.489 * (x + 0.3 * x**3)  # a sample function used in "shuffled"

    if cfg.f_type == "noise_only":
        y_train = gen_noise(n_train, cfg.noise_type)
        y_test  = gen_noise(n_test,  cfg.noise_type)

    elif cfg.f_type == "shuffled":
        shuffled_x_train = rng.permutation(x_train)
        shuffled_x_test = rng.permutation(x_test)
        y_train = f(shuffled_x_train) + gen_noise(n_train, cfg.noise_type)
        y_test  = f(shuffled_x_test)  + gen_noise(n_test,  cfg.noise_type)

    elif cfg.f_type == "random_pair":
        y_train = rng.normal(0, 1, n_train)
        y_test  = rng.normal(0, 1, n_test)

    else:
        raise ValueError(f"Unknown independent f_type: {cfg.f_type}")

    folder = DATASETS / f"{cfg.name}"
    #_{cfg.seed}_{cfg.f_type}_{cfg.noise_type}"
    folder.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"x": x_train, "y": y_train}).to_csv(folder / "train.csv", index=False)
    pd.DataFrame({"x": x_test,  "y": y_test}).to_csv(folder / "test.csv",  index=False)

    pd.DataFrame([[0, 0], [0, 0]], dtype=int).to_csv(folder / "adj_matrix.csv", header=False, index=False)

    meta = {
        "f_type": cfg.f_type,
        "noise_type": cfg.noise_type,
        "noise_scale": noise_scale,
        "n_train": n_train,
        "n_test": n_test,
        "seed": cfg.seed,
        "is_independent": True,
        "is_noiseless": False
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"✔ independent dataset ({cfg.f_type}) saved to {folder}")
    return folder

# ----------------------------------------------------------------------
# DEPENDENT DATASETS (WITH AND WITHOUT NOISE)
# ----------------------------------------------------------------------

def make_synthetic(cfg: SimpleNamespace | dict) -> Path:
    """
    Build a 2‑node SEM x → y and save in CSuite layout.

    cfg must contain:
      name, seed, f_type, noise_type, noise_scale, n_train, n_test
    """
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)

    g = Generator(seed=cfg.seed)

    # root node
    if cfg.x_dist == "normal":
        X = g.normal(0, 1, name="x")

    # structural function
    if cfg.f_type in ["cubic", "cubic_noiseless"]:
        f = lambda x: 0.489 * (x + 0.3 * x**3)
    elif cfg.f_type in ["quintic", "quintic_noiseless"]:
        f = lambda x: 0.268 * (x + 0.1 * x**5)
    elif cfg.f_type in ["sine", "sine_noiseless"]:
        f = lambda x: 1.209 * (
            np.sin(1.2 * x) + 0.3 * np.cos(0.8 * x) + 0.25 * np.tanh(0.8 * x)
        )
    else:
        raise ValueError(cfg.f_type)

    def noise(x):
        if cfg.noise_type == "logistic":
            scale = cfg.noise_scale * (np.sqrt(3) / np.pi)
            return g.random.logistic(loc=0, scale=scale, size=x.size)
        else:
            raise ValueError(cfg.noise_type)

    # Generate y
    if "noiseless" in cfg.f_type:
        Y = g.descendant(lambda x: f(x), parents=[X], name="y")
    else:
        Y = g.descendant(lambda x: f(x) + noise(x), parents=[X], name="y")

    folder = DATASETS / f"{cfg.name}"
    #_{cfg.seed}_{cfg.f_type}_{cfg.noise_type}"
    folder.mkdir(parents=True, exist_ok=True)

    g.generate(cfg.n_train).to_csv(folder / "train.csv", index=False)
    g.generate(cfg.n_test).to_csv(folder / "test.csv", index=False)
    
    if "noiseless" in cfg.f_type:
        pd.DataFrame([[0, 0], [0, 0]], dtype=int).to_csv(
            folder / "adj_matrix.csv", header=False, index=False
        )
    else:
        pd.DataFrame([[0, 1], [0, 0]], dtype=int).to_csv(
        folder / "adj_matrix.csv", header=False, index=False
    )

    meta = {
        "f_type": cfg.f_type,
        "noise_type": cfg.noise_type,
        "noise_scale": cfg.noise_scale,
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "seed": cfg.seed,
        "is_independent": False,
        "is_noiseless": "noiseless" in cfg.f_type
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"✔ dataset saved to {folder}")
    return folder

# ----------------------------------------------------------------------
# Dataset Loader
# ----------------------------------------------------------------------

def load_dataset(name: str, split: str = "all"):
    """Return X, Y arrays (standardised) from datasets/{name}/"""
    folder = DATASETS / name
    split_map = {
        "train": folder / "train.csv",
        "test": folder / "test.csv",
    }
    if split == "all":
        df = pd.concat([pd.read_csv(p) for p in split_map.values()], ignore_index=True)
    else:
        df = pd.read_csv(split_map[split])

    return df[["x"]].values, df[["y"]].values
