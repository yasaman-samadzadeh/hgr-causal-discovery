# HGR-LiNGAM: Nonlinear Causal Discovery via Kernel-HGR Enhanced LiNGAM

Empirical evaluation of **Hirschfeld-Gebelein-Renyi (HGR) kernel transformations** as a drop-in enhancement for **LiNGAM** causal discovery, targeting nonlinear, non-Gaussian structural equation models.

---

## Motivation

[LiNGAM](https://github.com/cdt15/lingam) recovers causal DAGs under linearity + non-Gaussianity assumptions. Real-world data is rarely linear. This project asks:

> **Can kernel-based HGR transformations (via [maxcorr](https://github.com/giuluck/maxcorr)) make LiNGAM work on nonlinear data?**

We attack this from three angles -- preprocessing transforms, internal algorithm injection, and independence-test replacement -- and benchmark each against standard LiNGAM variants across synthetic datasets with known ground truth.

## Repository Structure

```
.
├── bivariate-pipeline/        # Experiment 1: Config-driven bivariate HGR → LiNGAM pipeline
│   ├── configs/               #   YAML grids for datasets, HGR params, LiNGAM variants
│   ├── src/hgrlingam/         #   Core library (data generation, HGR wrapper, LiNGAM runner)
│   ├── scripts/               #   Runnable pipeline steps (generate, transform, evaluate)
│   └── pyproject.toml
│
├── hgr-injection/             # Experiment 2: DirectLiNGAM subclass with HGR internals
│   └── src/lingamhgr/
│       ├── direct_lingam_hgr_injection.py    # DirectLiNGAM_HGR: poly residuals + DK-HGR scorer
│       └── direct_lingam_hgr_experiments.py  # Random polynomial DAG benchmarks + DAG plots
│
├── hgr-vs-hsic/               # Experiment 3: HGR vs HSIC independence test comparison
│   └── hgrlingam/
│       └── hgrlingam.py       # HGRDirectLiNGAM built on _BaseLiNGAM
│
├── multivariate-benchmarks/   # Experiment 4: Multivariate DAG benchmarks at scale
│   ├── final.ipynb            #   NL-HGR comparative analysis (linear/cubic/sine/tanh/square)
│   ├── multivariate.ipynb     #   Full multivariate experiments with DAG visualisation
│   └── multivariate1.ipynb    #   Extended runs with kernel sweep (k ∈ {1,3}, {1,5}, …)
│
├── final.ipynb                # Experiment 5: End-to-end NL-HGR vs baselines comparison
├── multivariate1.ipynb        # Colab notebook: multivariate HGR-LiNGAM experiments
├── multivariate12.ipynb       # Colab notebook: extended multivariate experiments
│
├── docs/
│   ├── project_summary.md     # High-level project overview and timeline
│   └── experiment_report.md   # Detailed experiment report with code snippets
│
├── references/                # Papers and reading materials (gitignored)
└── .gitignore
```

## Experiments

### 1. Bivariate Pipeline (`bivariate-pipeline/`)

A full config-driven pipeline for the simplest causal problem: two variables, one edge.

- **Data generation**: Synthetic `x → y` SEMs with configurable structural functions (cubic, quintic, sine), noise families (logistic, Laplace), and noise scales
- **HGR transformation**: Double-Kernel HGR via `maxcorr` with grid search over `kernel_a` × `kernel_b` ∈ {3,4,5} × {1,2,3,4,5}
- **Causal evaluation**: Run DirectLiNGAM on transformed data, measure direction accuracy
- **Independence testing**: Verify residual ⊥ input post-transform

```bash
cd bivariate-pipeline
pip install -e .

python scripts/generate_all_datasets.py
python scripts/run_hgr_for_all.py
python scripts/run_lingam_for_all.py
python scripts/evaluate_direction_accuracy.py
```

### 2. HGR Injection (`hgr-injection/`)

Replaces LiNGAM's internal independence test and residual computation with nonlinear alternatives:

- **Polynomial ridge residuals** instead of OLS (configurable degree)
- **Double-Kernel HGR** as the independence scorer (configurable `kernel_a`, `kernel_b`)
- Benchmarks `DirectLiNGAM_HGR` against `PW-ling` and `HSIC-kernel` on random polynomial DAGs

```bash
cd hgr-injection/src
python -m lingamhgr.direct_lingam_hgr_injection --bench 20 --samples 1000 --deg-res 5 --kernel-a 4 --kernel-b 6
```

### 3. HGR vs HSIC (`hgr-vs-hsic/`)

A standalone `HGRDirectLiNGAM` implementation built directly on `lingam._BaseLiNGAM`, replacing the exogeneity score loop with kernel-HGR. Designed for clean head-to-head comparison against HSIC-based kernels.

```python
from hgrlingam import HGRDirectLiNGAM
model = HGRDirectLiNGAM(kernel_sizes=(0.3, 0.3), random_state=0)
model.fit(X)
print(model.causal_order_)
```

### 4. Multivariate Benchmarks (`multivariate-benchmarks/`)

Large-scale evaluation across multiple DAG sizes, function types, and kernel configurations. Generates:

- SHD / F1 comparisons across methods (PW-ling, HSIC-kernel, HGR-DK)
- Kernel parameter sweeps (k ∈ {1,3}, {1,5}, {3,3}, {3,5}, {5,5})
- Runtime comparisons
- Heatmaps and bar charts for presentations

### 5. End-to-End Comparison (`final.ipynb`)

The top-level notebook implementing the full **NL-HGR** method: an end-to-end nonlinear LiNGAM variant with HGR-based components, compared against linear baselines.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| [`lingam`](https://github.com/cdt15/lingam) | LiNGAM causal discovery algorithms |
| [`maxcorr`](https://github.com/giuluck/maxcorr) | Kernel-HGR computation (Double Kernel) |
| [`causalgen`](https://github.com/giuluck/causalgen) | Synthetic causal data generation |
| `numpy`, `scipy`, `scikit-learn` | Numerical computation |
| `matplotlib`, `seaborn`, `networkx` | Visualisation |
| `omegaconf` | Config management (bivariate pipeline) |

```bash
pip install lingam maxcorr causalgen numpy scipy scikit-learn matplotlib seaborn networkx omegaconf
```

## Results Summary

Key findings from the experiments:

- **Bivariate**: HGR transformation before LiNGAM significantly improves direction accuracy for nonlinear functions (cubic, quintic, sine) compared to raw LiNGAM
- **Multivariate**: `DirectLiNGAM_HGR` with polynomial residuals + DK-HGR achieves lower SHD and higher F1 than vanilla PW-ling on polynomial DAGs
- **Kernel sensitivity**: Performance varies with kernel degree configuration; asymmetric kernels (e.g., `kernel_a=4, kernel_b=2`) sometimes outperform symmetric ones
- **Runtime**: HGR-based methods are slower than PW-ling but competitive with HSIC-kernel

## License

Research code -- not yet released under a formal license.
