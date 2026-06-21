# ePINN-AF — Navier-Stokes (forward, 2D cylinder wake)

Solves the incompressible Navier-Stokes momentum equations on the classic
cylinder-wake benchmark (Re=100), with known `LAMBDA_1=1, LAMBDA_2=0.01`.
The network predicts a stream function `psi` and pressure `p`; velocities
are recovered as `u = psi_y`, `v = -psi_x`, which satisfies the continuity
equation exactly by construction.

This is the most elaborate ePINN-AF variant in the repo:
- per-rule output heads (like KdV/Poisson)
- a **direct head** in parallel with the fuzzy-gated sum, so the network
  always keeps a clean MLP path
- **time-only fuzzy partitioning** (`partition_dims=[2]`) — rule
  memberships specialise on the periodic vortex-shedding phase rather
  than on spatial regions

## Files

| File             | Contents                                                  |
|------------------|-------------------------------------------------------------|
| `model.py`       | `AttentionFuzzyLayer` + `AFPINN` (direct head + partition_dims) |
| `pinn.py`        | `NSBase`, `ns_forward` (psi -> u, v via autograd) + `PhysicsInformedNN` |
| `data.py`        | Loads `cylinder_nektar_wake.mat`, builds train/collocation sets |
| `utils.py`       | `seed_torch`, `get_device`                                    |
| `train.py`       | Entry point: load data, train, evaluate, save a results pickle |

## Data

Download `cylinder_nektar_wake.mat` (Raissi et al.'s original PINN
benchmark data), e.g. from https://github.com/maziarraissi/PINNs
(`appendix/Data/cylinder_nektar_wake.mat`).

## Run

```bash
pip install -r ../../requirements.txt
python train.py --data /path/to/cylinder_nektar_wake.mat
```

Optional flags: `--n-iter` (default 20000), `--n-train` (default 5000),
`--seed`. Training is significantly heavier than the 1D examples — a
GPU is strongly recommended.
