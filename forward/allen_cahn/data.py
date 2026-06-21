"""
Data loading for the forward Allen-Cahn benchmark.

Expects an `AC.mat` file with:
    x  : [N, 1]   spatial grid
    tt : [T, 1]   time grid
    uu : [N, T]   exact solution u(x, t)
"""
import numpy as np
import scipy.io


def load_data(data_path, n_train=8000, seed=None):
    """Load AC.mat and build the training set (same points are reused for
    both the data-fidelity and the PDE-residual loss terms).
    """
    data = scipy.io.loadmat(data_path)
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['uu']).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    if seed is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X_star.shape[0], n_train, replace=False)
    else:
        idx = np.random.choice(X_star.shape[0], n_train, replace=False)

    X_train = X_star[idx, :]
    u_train = u_star[idx, :]

    return {
        'x': x, 't': t, 'Exact': Exact, 'X': X, 'T': T,
        'X_star': X_star, 'u_star': u_star,
        'lb': lb, 'ub': ub,
        'X_train': X_train, 'u_train': u_train, 'N_u': n_train,
    }
