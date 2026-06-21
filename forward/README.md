# Forward problems

Each subfolder solves one PDE given known coefficients and noiseless
supervised data, with the ePINN-AF architecture.

| Folder           | Equation                                          |
|------------------|-----------------------------------------------------|
| `burgers/`       | `u_t + u u_x - (0.01/pi) u_xx = 0`                  |
| `kdv/`           | `u_t + u u_x + 0.0025 u_xxx = 0`                    |
| `allen_cahn/`    | `u_t - 0.0001 u_xx + 5u^3 - 5u = 0`                 |
| `poisson/`       | `Delta u = f`  (2D or 3D boundary-value problem)    |
| `navier_stokes/` | 2D incompressible Navier-Stokes, cylinder wake      |

See each folder's `README.md` for data requirements and the exact run
command.
