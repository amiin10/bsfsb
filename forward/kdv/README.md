# ePINN-AF — Korteweg-de Vries (KdV) equation (forward)

Solves

```
u_t + u * u_x + 0.0025 * u_xxx = 0
```

given noiseless supervised data on `u`, with the ePINN-AF architecture
(per-rule-head variant, with input normalisation to `[-1, 1]` via the
domain bounds).

## Files

Same layout as `forward/burgers`: `model.py`, `pinn.py`, `data.py`, `utils.py`, `train.py`.

## Data

A `KdV.mat` file with `x`, `tt`, `uu` (spatial grid, time grid, exact
solution) — the standard KdV PINN benchmark dataset.

## Run

```bash
pip install -r ../../requirements.txt
python train.py --data /path/to/KdV.mat
```

Optional flags: `--n-iter` (default 10000), `--n-train` (default 8000),
`--seed`.
