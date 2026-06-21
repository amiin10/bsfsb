"""
Data loading for the inverse 2D Navier-Stokes (cylinder wake) benchmark:
same `cylinder_nektar_wake.mat` file as the forward problem. Only (u, v)
are supervised here (pressure is withheld, and only used afterwards for
evaluation) -- the inverse problem must recover lambda_1, lambda_2 purely
from noisy velocity measurements plus the PDE residual.
"""
import numpy as np
import scipy.io


def load_data(data_path, n_train=5000, noise_level=0.0, seed=None):
    """Load cylinder_nektar_wake.mat and build the (u, v)-only training set
    plus a separate collocation set.

    noise_level is a FRACTION (e.g. 0.01 = 1%) of std(u_train)/std(v_train),
    matching the convention used by the original inverse-NS experiments
    (NOT a percentage like the other inverse PDE folders).
    """
    data = scipy.io.loadmat(data_path)

    U_star = data['U_star']
    P_star = data['p_star']
    t_star = data['t'].flatten()[:, None]
    X_star = data['X_star']

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
    u_train_clean = u[idx, :]
    v_train_clean = v[idx, :]
    p_train_ref = p_true[idx, :]   # withheld from training; eval reference only

    if noise_level > 0.0:
        u_std = float(np.std(u_train_clean))
        v_std = float(np.std(v_train_clean))
        u_train = u_train_clean + noise_level * u_std * rng.randn(*u_train_clean.shape)
        v_train = v_train_clean + noise_level * v_std * rng.randn(*v_train_clean.shape)
    else:
        u_train = u_train_clean
        v_train = v_train_clean

    idx_f = rng.choice(X_star_full.shape[0], n_train, replace=False)
    X_f = X_star_full[idx_f, :]

    return {
        'X_star_full': X_star_full, 'X_star_2D': X_star,
        'u': u, 'v': v, 'p_true': p_true,
        'UU': UU, 'VV': VV, 'PP': PP,
        'lb': lb, 'ub': ub, 'N': N, 'T': T,
        'X_train': X_train,
        'u_train': u_train, 'v_train': v_train,
        'u_train_clean': u_train_clean, 'v_train_clean': v_train_clean,
        'p_train_ref': p_train_ref, 'noise_level': noise_level,
        'X_f': X_f, 'N_train': n_train,
    }
