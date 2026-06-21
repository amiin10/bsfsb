"""
Data loading for the 2D Navier-Stokes (cylinder wake) forward benchmark.

Expects the classic Raissi et al. `cylinder_nektar_wake.mat` file, with:
    U_star : [N, 2, T]   (u, v) velocity field
    p_star : [N, T]      pressure field
    t      : [T, 1]      time grid
    X_star : [N, 2]      (x, y) spatial coordinates

Download, e.g. from https://github.com/maziarraissi/PINNs
(appendix/Data/cylinder_nektar_wake.mat).
"""
import numpy as np
import scipy.io


def load_data(data_path, n_train=5000, seed=None):
    data = scipy.io.loadmat(data_path)

    U_star = data['U_star']            # [N, 2, T]
    P_star = data['p_star']            # [N, T]
    t_star = data['t'].flatten()[:, None]
    X_star = data['X_star']            # [N, 2]

    N = X_star.shape[0]
    T = t_star.shape[0]

    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(t_star, (1, N)).T

    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p_true = PP.flatten()[:, None]

    X_star_full = np.hstack((x, y, t))
    lb = X_star_full.min(0)
    ub = X_star_full.max(0)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    idx = rng.choice(X_star_full.shape[0], n_train, replace=False)
    X_train = X_star_full[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]
    p_train = p_true[idx, :]

    idx_f = rng.choice(X_star_full.shape[0], n_train, replace=False)
    X_f = X_star_full[idx_f, :]

    return {
        'X_star_full': X_star_full, 'X_star_2D': X_star,
        'u': u, 'v': v, 'p_true': p_true,
        'UU': UU, 'VV': VV, 'PP': PP,
        'x': X_star[:, 0:1], 'y': X_star[:, 1:2], 't': t_star,
        'lb': lb, 'ub': ub, 'N': N, 'T': T,
        'X_train': X_train, 'u_train': u_train, 'v_train': v_train, 'p_train': p_train,
        'X_f': X_f, 'N_train': n_train,
    }
