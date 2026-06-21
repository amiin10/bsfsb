# ePINN-AF — Navier-Stokes (inverse, 2D cylinder wake)

Given noisy measurements of velocity `(u, v)` ONLY (pressure is withheld
during training and used purely for evaluation afterwards), recover the
two PDE coefficients

```
u_t + lambda_1*(u u_x + v u_y) + p_x - lambda_2*(u_xx + u_yy) = 0
v_t + lambda_1*(u v_x + v v_y) + p_y - lambda_2*(v_xx + v_yy) = 0
```

True values: `lambda_1 = 1.0`, `lambda_2 = 0.01`.

This folder uses an independent re-implementation of the ePINN-AF
architecture (`EPINNAFNetwork` in `model.py`) — written directly with
`nn.ModuleList` rule heads and a sigmoid-gated direct head — rather than
reusing `forward/navier_stokes/model.py`'s `AFPINN`. Both encode the same
attention-fuzzy-gated-mixture idea; this version is specific to the
inverse codebase it was developed in.

## Files

| File             | Contents                                                  |
|------------------|-------------------------------------------------------------|
| `model.py`       | `EPINNAFNetwork` — attention-fuzzy mixture-of-rules backbone  |
| `pinn.py`        | `_InverseBase` (trainable lambdas) + `PhysicsInformedNN`, `inverse_ns_residual` |
| `data.py`        | Loads `cylinder_nektar_wake.mat`, withholds pressure, adds noise to (u, v) |
| `utils.py`       | `seed_torch`, `get_device`                                    |
| `train.py`       | Entry point: load data, train, recover lambdas, save results  |

## Data

Download `cylinder_nektar_wake.mat`, e.g. from
https://github.com/maziarraissi/PINNs
(`appendix/Data/cylinder_nektar_wake.mat`).

## Run

```bash
pip install -r ../../requirements.txt
python train.py --data /path/to/cylinder_nektar_wake.mat --noise 0.01
```

Note: `--noise` here is a **fraction** of `std(u)`/`std(v)` (e.g. `0.01`
= 1%), unlike the percentage convention used in the other `inverse/*`
folders. Other flags: `--n-iter` (default 20000), `--n-train` (default
5000), `--seed`. A GPU is strongly recommended.
