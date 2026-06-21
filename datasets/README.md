# Datasets

Four `.mat` files are needed to run the benchmarks in this repo. The
Poisson experiments (`forward/poisson/`) use a manufactured analytical
solution and need no data file.

| File                          | Used by                                              | Source |
|-------------------------------|-------------------------------------------------------|--------|
| `burgers_shock.mat`           | `forward/burgers`, `inverse/burgers`                  | [PINNs repo](https://github.com/maziarraissi/PINNs) — `appendix/Data/burgers_shock.mat` |
| `AC.mat`                      | `forward/allen_cahn`, `inverse/allen_cahn`            | [PINNs repo](https://github.com/maziarraissi/PINNs) — `main/Data/AC.mat` |
| `KdV.mat`                     | `forward/kdv`, `inverse/kdv`                          | [PINNs repo](https://github.com/maziarraissi/PINNs) — `main/Data/KdV.mat` |
| `cylinder_nektar_wake.mat`    | `forward/navier_stokes`, `inverse/navier_stokes`      | [PINNs repo](https://github.com/maziarraissi/PINNs) — `appendix/Data/cylinder_nektar_wake.mat` |

Download whichever files you need and pass the path directly via each
script's `--data` flag, e.g.:

```bash
cd forward/burgers
python train.py --data /path/to/burgers_shock.mat
```

There's no requirement to put the files in this folder specifically —
every `train.py` takes an explicit `--data` path. Placing them here
(`datasets/burgers_shock.mat`, etc.) is just a convenient convention if
you're running several experiments from a shared checkout.

These datasets were produced by Maziar Raissi and collaborators for the
original PINN (Raissi, Perdikaris & Karniadakis, 2019) and HFM (Raissi,
Yazdani & Karniadakis, 2020) papers; please refer to those repositories
for their original license terms.
