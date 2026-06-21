<div align="center">

# ePINN-AF

### Enhanced Physics-Informed Neural Networks with Attention–Fuzzy Logic

*A drop-in PINN architecture that mitigates spectral bias and gradient pathologies
by adaptively partitioning the input domain through soft fuzzy rules and
softmax attention.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-pep8-000000.svg)]()
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)]()

</div>

---

## Table of Contents

1. [Why ePINN-AF?](#why-epinn-af)
2. [Method at a glance](#method-at-a-glance)
3. [Repository structure](#repository-structure)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Benchmarks](#benchmarks)
7. [How it works (deeper dive)](#how-it-works-deeper-dive)
8. [Reproducing the paper](#reproducing-the-paper)
9. [Citation](#citation)
10. [Contact](#contact)
11. [Acknowledgments](#acknowledgments)
12. [License](#license)

---

## Why ePINN-AF?

Physics-Informed Neural Networks (PINNs, Raissi et al. 2019) are an elegant
framework for solving PDEs: parameterize the solution by an MLP, take
analytic derivatives via autograd, and minimize the residual at collocation
points. In practice three well-documented pathologies make plain PINNs fragile:

| Pathology                                | Symptom                                                                                       | Fix in ePINN-AF                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Spectral bias**                         | Network learns low-frequency components and ignores high-frequency residuals (NTK eigen gap). | Per-rule heads `h_j(z)` over a shared backbone give each rule its own spectrum. |
| **Gradient pathologies** (Wang et al. 2021) | ‖∇θ L_pde‖ ≫ ‖∇θ L_data‖, optimisation stalls in the residual term.                          | Soft fuzzy gating `γ_j = α_j·μ_j` localizes residual gradients per region.      |
| **Multi-regime / stiff problems**         | One MLP must fit pre-shock + post-shock or several disparate length-scales simultaneously.    | Fuzzy rules partition the input space; attention selects which rules fire where. |

ePINN-AF keeps the **same loss form** as a standard PINN
(`L = MSE_data + MSE_pde`, no adaptive weights, no curriculum) and addresses
the above purely through the architecture. The architectural cost is modest
— a few extra hundred parameters for the attention sub-net and `M` fuzzy
heads — and the training pipeline (Adam → L-BFGS) is identical.

---

## Method at a glance

For an input `z ∈ ℝ^d` (e.g. `(x, t)` or `(x, y, t)`), the network outputs

```
û(z)  =  Σ_{j=1}^{M}  α_j(z) · μ_j(z) · h_j(z; θ_h)   +  b
       └─────────┬───────────┘└──────┬──────┘
            adaptive gate     per-rule head
```

where

| Symbol           | Definition                                                                                | Role                                              |
| ---------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `μ_j(z)`         | `exp(-½ Σᵢ ((zᵢ - c_{ji}) / σ_{ji})²)`                                                    | Soft Gaussian rule over the input domain          |
| `α_j(z)`         | `softmax_j(W₂ · tanh(W₁ z + b₁) + b₂)`                                                    | Attention weights — what each rule should focus on |
| `γ_j(z)`         | `μ_j(z) · α_j(z)`                                                                         | Combined gating per rule                          |
| `h(z; θ_h)`      | MLP backbone, tanh on every hidden layer                                                  | Shared latent features                            |
| `h_j(z)`         | `W_j · h(z)`                                                                              | Per-rule projection of the shared features        |
| `b`              | Trainable output bias                                                                     | —                                                  |

The fuzzy centers `c_j` and widths `σ_j` are **learnable** — the network
discovers where to place its rules. With `M = 4–16` rules ePINN-AF already
matches or beats much larger plain PINNs on every benchmark in this repo.

Two optional switches (off by default for 1-D PDEs, on for Navier-Stokes):

- **`partition_dims`** — restrict `μ_j` to a subset of input axes.
  Example: `partition_dims=[2]` on a `(x, y, t)` problem makes rules localize
  along time only, ideal for periodically-shedding flows.
- **`use_direct_head`** — adds a parallel `W_d · h(z)` path that bypasses
  the fuzzy gate. Guarantees a clean gradient highway and helps on
  multi-output problems.

The exact variant differs slightly by PDE (these evolved incrementally
across experiments, and each folder keeps the version that was actually
used for that problem) — see [Repository structure](#repository-structure)
below for which folder uses which.

---

## Repository structure

```
ePINN-AF/
├── forward/                    # known coefficients, learn the solution u
│   ├── README.md
│   ├── burgers/                model.py  pinn.py  data.py  utils.py  train.py  README.md
│   ├── kdv/                    (same five files)
│   ├── allen_cahn/             (same five files)
│   ├── poisson/                (same five files; data.py builds a manufactured BVP, no dataset)
│   └── navier_stokes/          (same five files)
├── inverse/                    # noisy data on u, recover unknown PDE coefficients
│   ├── README.md
│   ├── burgers/                (same five-file layout)
│   ├── kdv/
│   ├── allen_cahn/
│   └── navier_stokes/
├── datasets/
│   └── README.md                download links for the four required .mat files
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md                    (this file)
```

Every PDE folder is self-contained:

| File         | Role                                                              |
| ------------ | ------------------------------------------------------------------ |
| `model.py`   | `AttentionFuzzyLayer` + the ePINN-AF network for that PDE           |
| `pinn.py`    | PDE residual, loss function, Adam→L-BFGS training loop, prediction |
| `data.py`    | Data loading / sampling / noise injection for that PDE              |
| `utils.py`   | `seed_torch`, `get_device`                                          |
| `train.py`   | The script you actually run: load data → train → evaluate → save   |

---

## Installation

```bash
git clone https://github.com/amiin10/ePINN-AF.git
cd ePINN-AF
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.9–3.12, PyTorch ≥ 2.0 on both CPU and CUDA.
A single mid-range GPU (e.g. T4, 3060) is enough for every experiment;
runtime per script ranges from ~3 minutes (Burgers) to ~40 minutes
(Navier-Stokes with 20,000 Adam iterations).

### Datasets

Four `.mat` files are needed (Burgers, Allen-Cahn, KdV, Navier-Stokes).
The Poisson experiments use a manufactured analytical solution and need
no data. See [`datasets/README.md`](datasets/README.md) for download
links.

---

## Quick start

Each PDE folder is independent — `cd` into one and run its `train.py`:

```bash
cd forward/burgers
python train.py --data /path/to/burgers_shock.mat

cd ../../inverse/kdv
python train.py --data /path/to/KdV.mat --noise 2
```

`train.py` prints training progress (loss, and for inverse problems the
recovered coefficients), then pickles everything useful — predictions,
the exact/reference solution, relative-L2 and pointwise errors, loss and
gradient-norm histories, and the architecture config — to a `.pkl` file
in the current directory (e.g. `burgers_epinn_af_results.pkl`). Load it
with `pickle.load()` and plot whatever you need with matplotlib; no
bundled plotting script is included.

Every script accepts `--n-iter`, `--seed`, and `--out`; inverse scripts
additionally accept `--noise`. Run `python train.py --help` in any
folder for the full list, or see that folder's `README.md`.

### Using the model in your own code

There's no installable `ePINN-AF` package — each PDE folder is a
standalone pair of `model.py` / `pinn.py` files you can read top to
bottom. From inside e.g. `forward/kdv/`:

```python
import torch
from model import AFPINN
from utils import seed_torch, get_device

seed_torch(0)
device = get_device()

model = AFPINN(
    backbone_layers = [200, 200, 200, 200],    # tanh on every hidden layer
    n_rules         = 8,                       # M fuzzy rules
    attn_hidden     = 64,                      # width of attention sub-net
    output_dim      = 1,                       # scalar PDE
    input_dim       = 2,                       # e.g. (x, t)
    lb              = [-1.0, 0.0],             # domain bounds (for normalization)
    ub              = [ 1.0, 1.0],
).to(device)

u_hat = model(torch.randn(128, 2, device=device))   # [128, 1]
```

The exact constructor signature (whether `lb`/`ub` or `partition_dims`
are accepted, the output dimension, etc.) varies slightly by PDE — see
that folder's `model.py` and `README.md`. Pair the network with a
PDE-specific residual (a handful of `torch.autograd.grad` calls, as in
`pinn.py`) and an optimizer to reproduce the full training loop.

---

## Benchmarks

> Exact error/runtime figures depend on hardware, PyTorch version, and
> seed — run the script for your PDE of interest to get numbers for your
> setup. All errors reported by the scripts are relative L² unless noted.

| PDE                     | Type    | Folder                  |
| ------------------------ | ------- | ------------------------ |
| Burgers                  | Forward | `forward/burgers/`        |
| KdV                       | Forward | `forward/kdv/`            |
| Allen-Cahn                | Forward | `forward/allen_cahn/`     |
| Poisson (2D / 3D)         | Forward | `forward/poisson/`        |
| Navier-Stokes (cyl. wake) | Forward | `forward/navier_stokes/`  |
| Burgers                  | Inverse | `inverse/burgers/`         |
| KdV                       | Inverse | `inverse/kdv/`            |
| Allen-Cahn                | Inverse | `inverse/allen_cahn/`     |
| Navier-Stokes (cyl. wake) | Inverse | `inverse/navier_stokes/`  |

Refer to the manuscript for head-to-head comparisons against plain PINN,
APINN, FPINN, SA-PINN and CausalPINN under identical settings.

---

## How it works (deeper dive)

**The trick is in `γ_j(z) = α_j(z) · μ_j(z)`.**

The fuzzy term `μ_j(z)` is *spatially* local — it lights up only where the
input is close to the rule's learned center `c_j`. The attention term
`α_j(z)` is *content-driven* — it can suppress or amplify a rule based on
the full input. Their product gives a soft, learnable partition of the
input space:

- In smooth regions where one rule's center is dominant, that rule's head
  `h_j(z)` carries the prediction.
- Near sharp features (shocks, interfaces, vortices), several rules can
  overlap and their weighted sum captures the local behaviour at higher
  effective resolution than a single MLP.

Because the gating is fully differentiable, gradients flow back into the
centers `c_j`, widths `σ_j`, attention weights, backbone parameters, and
per-rule heads all at once. There are **no auxiliary losses, no manual
weight tuning, no curriculum**.

**Spectral bias.** The per-rule heads `h_j(z) = W_j · h(z)` give each rule
its own linear projection of the backbone features. In the NTK regime this
amounts to widening the effective spectral support of the network: rather
than `K_uu` being a single rank-D kernel, it becomes a mixture of M
rank-D kernels selected per location by `γ_j`. The result is a flatter NTK
eigenvalue decay and a lower condition number, both of which correlate
with the residual term being well-conditioned (Wang, Wang & Perdikaris,
2022).

---

## Reproducing the paper

The training scripts in this repo are simplified to make ePINN-AF easy to
read and re-use. The original full-experiment scripts (including baselines
PINN/APINN/FPINN/SA-PINN/CausalPINN and the editor-requested gradient-flow
/ loss-landscape / NTK diagnostics used in the rebuttal) were used to
produce the manuscript's tables and figures. Neither the baseline codes
nor the diagnostic tooling are included here — this repository is focused
on the proposed method only, kept as small and readable as possible.

If you would like the comparison or diagnostic scripts, please open an
issue or contact the author (see [Contact](#contact)).

---

## Citation

If you use this code in your research, please cite both the software and
the paper once it is published. The GitHub "Cite this repository" button
reads from [`CITATION.cff`](CITATION.cff).

```bibtex
@software{hosseini_epinn_af,
  author  = {Hosseini, Amin},
  title   = {ePINN-AF: Enhanced Physics-Informed Neural Networks with Attention-Fuzzy Logic},
  url     = {https://github.com/amiin10/ePINN-AF},
  year    = {2026}
}
```

A companion paper is in preparation; this entry will be updated with the
full bibliographic details once it is available.

---

## Contact

**Aminhosseini**
✉ &nbsp; *amin.hosseini1@aut.ac.ir*

🐙 &nbsp; [github.com/Amiin10](https://github.com/amiin10)

🔬 &nbsp; *Department of Mechanical Engineering, Amirkabir University of Technology (Tehran Polytechnic)*

Bug reports, feature requests, and questions are very welcome via
[GitHub Issues](https://github.com/amiin10/ePINN-AF/issues).

---

## Acknowledgments

- The benchmark datasets (`burgers_shock.mat`, `AC.mat`, `KdV.mat`,
  `cylinder_nektar_wake.mat`) come from the original PINN and HFM
  repositories by Maziar Raissi and collaborators.

---

## License

This project is released under the [MIT License](LICENSE).
The benchmark datasets remain under their original licenses; please
refer to the upstream repositories listed in
[`datasets/README.md`](datasets/README.md).

---

<div align="center">

*Built with ❤ for the scientific-machine-learning community.*

</div>
