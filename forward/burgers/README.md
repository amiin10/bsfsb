# ePINN-AF — Burgers equation (forward)

Solves the viscous Burgers equation

```
u_t + u * u_x - (0.01/pi) * u_xx = 0,   (x, t) in [-1, 1] x [0, 1]
```

given noiseless supervised data on `u`, with the ePINN-AF architecture.

## Files

| File            | Contents                                                      |
|-----------------|----------------------------------------------------------------|
| `model.py`      | `AttentionFuzzyLayer` + `AFPINN` — the ePINN-AF network (Eqs 3.5-3.10) |
| `pinn.py`       | `PhysicsInformedNN` — loss function, Adam+L-BFGS training loop, prediction |
| `data.py`       | Loads `burgers_shock.mat` and builds the training/collocation sets |
| `utils.py`      | `seed_torch`, `get_device`                                     |
| `train.py`      | Entry point: load data, train, evaluate, save a results pickle |

## Data

Download `burgers_shock.mat` (Raissi et al.'s original PINN benchmark data),
e.g. from https://github.com/maziarraissi/PINNs (`appendix/Data/burgers_shock.mat`).

## Run

```bash
pip install -r ../../requirements.txt
python train.py --data /path/to/burgers_shock.mat
```

Optional flags: `--n-iter` (Adam iterations, default 10000), `--n-train`
(number of supervised points, default 2000), `--seed`.

Results (predictions, errors, loss/gradient histories) are pickled to
`burgers_epinn_af_results.pkl`.
