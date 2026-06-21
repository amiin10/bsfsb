# Inverse problems

Each subfolder recovers unknown PDE coefficients (`lambda_1`, `lambda_2`)
from noisy supervised data, jointly with the network, using the ePINN-AF
architecture. `lambda_1`/`lambda_2` are `torch.nn.Parameter`s trained
alongside the network weights.

| Folder           | Equation                                                    | True coefficients |
|------------------|---------------------------------------------------------------|--------------------|
| `burgers/`       | `u_t + lambda_1 u u_x - lambda_2 u_xx = 0`                    | `1.0`, `0.01/pi`   |
| `kdv/`           | `u_t + lambda_1 u u_x + lambda_2 u_xxx = 0`                   | `1.0`, `0.0025`    |
| `allen_cahn/`    | `u_t - lambda_1 u_xx + lambda_2 u^3 - lambda_2 u = 0`         | `0.0001`, `5.0`    |
| `navier_stokes/` | 2D Navier-Stokes momentum (cylinder wake), velocity-only data | `1.0`, `0.01`      |

See each folder's `README.md` for data requirements, noise convention,
and the exact run command.
