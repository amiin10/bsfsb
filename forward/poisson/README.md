# ePINN-AF — Poisson equation (forward BVP)

Solves the boundary-value problem

```
Delta u = f(x)   on Omega = [-5, 5]^d     (d = 2 or 3)
u = g(x)         on d(Omega)              (Dirichlet)
```

with manufactured solution `u* = sin(x) sin(y)` (2D) or
`u* = sin(x) sin(y) sin(z)` (3D), so the exact solution / forcing term are
generated in closed form (no external dataset needed).

This is a boundary-value problem rather than a time-dependent PDE, so the
loss has a different structure: `L = MSE_boundary + MSE_residual` (no
initial condition / no separate collocation-vs-data split — see `pinn.py`).

## Files

| File             | Contents                                                  |
|------------------|-------------------------------------------------------------|
| `model.py`       | `AttentionFuzzyLayer` + `AFPINN` (per-rule-head variant)     |
| `pinn.py`        | `PhysicsInformedNN` — boundary + residual loss, training loop |
| `data.py`        | Exact solution, forcing term, Laplacian, boundary/interior sampling |
| `utils.py`       | `seed_torch`, `get_device`                                    |
| `train.py`       | Entry point: generate data, train, evaluate, save a results pickle |

## Run

```bash
pip install -r ../../requirements.txt
python train.py --dim 2
python train.py --dim 3
```

No dataset download needed — data is generated from the manufactured
solution. Optional flags: `--n-iter` (default 10000), `--seed`.
