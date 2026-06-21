# ePINN-AF — KdV equation (inverse / parameter identification)

Given noisy measurements of `u(x, t)`, recover the two PDE coefficients

```
u_t + lambda_1 * u * u_x + lambda_2 * u_xxx = 0
```

True values: `lambda_1 = 1.0`, `lambda_2 = 0.0025`. Both are trained
jointly with the network as `torch.nn.Parameter`s.

## Files

Same layout as `inverse/allen_cahn`: `model.py`, `pinn.py`,
`data.py`, `utils.py`, `train.py`.

## Run

```bash
pip install -r ../../requirements.txt
python train.py --data /path/to/KdV.mat --noise 2
```

`--noise` is the measurement-noise level, in percent of `std(u)` (default
0). Other flags: `--n-iter` (default 10000), `--n-train` (default 8000),
`--seed`.
