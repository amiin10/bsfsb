# ePINN-AF — Allen-Cahn equation (inverse / parameter identification)

Given noisy measurements of `u(x, t)`, recover the two PDE coefficients

```
u_t - lambda_1 * u_xx + lambda_2 * u^3 - lambda_2 * u = 0
```

True values: `lambda_1 = 0.0001`, `lambda_2 = 5.0`. `lambda_1` and
`lambda_2` are `torch.nn.Parameter`s trained jointly with the network.

## Files

| File             | Contents                                                  |
|------------------|-------------------------------------------------------------|
| `model.py`       | `AttentionFuzzyLayer` + `AFPINN` (same architecture as forward) |
| `pinn.py`        | `_InverseMixin` (trainable lambdas) + `PhysicsInformedNN`     |
| `data.py`        | Loads `AC.mat`, adds Gaussian measurement noise               |
| `utils.py`       | `seed_torch`, `get_device`                                    |
| `train.py`       | Entry point: load data, train, recover lambdas, save results  |

## Run

```bash
pip install -r ../../requirements.txt
python train.py --data /path/to/AC.mat --noise 3
```

`--noise` is the measurement-noise level, in percent of `std(u)` (default
0 = clean data). Other flags: `--n-iter` (default 10000), `--n-train`
(default 8000), `--seed`.
