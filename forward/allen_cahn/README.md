# ePINN-AF — Allen-Cahn equation (forward)

Solves

```
u_t - 0.0001 * u_xx + 5 * u^3 - 5 * u = 0
```

given noiseless supervised data on `u`, with the ePINN-AF architecture
(simple scalar-gate variant, same as `forward/burgers`).

## Files

Same layout as `forward/burgers`: `model.py`, `pinn.py`, `data.py`, `utils.py`, `train.py`.

## Data

An `AC.mat` file with `x`, `tt`, `uu` (spatial grid, time grid, exact
solution) — the standard Allen-Cahn PINN benchmark dataset.

## Run

```bash
pip install -r ../../requirements.txt
python train.py --data /path/to/AC.mat
```

Optional flags: `--n-iter` (default 10000), `--n-train` (default 8000),
`--seed`.
