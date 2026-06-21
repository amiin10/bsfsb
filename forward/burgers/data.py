"""
Data loading for the forward Burgers benchmark.

Expects the classic Raissi et al. `burgers_shock.mat` file, which contains:
    x    : [256, 1]   spatial grid
    t    : [100, 1]   time grid
    usol : [256, 100] exact solution u(x, t)

Download (one of the standard mirrors), e.g.:
    https://github.com/maziarraissi/PINNs (appendix/Data/burgers_shock.mat)
"""
import numpy as np
import scipy.io


def load_data(data_path, n_train=2000, seed=None):
    """Load burgers_shock.mat and build the training / collocation sets.

    Returns a dict with:
        x, t, Exact          raw grid + exact solution, shapes [256,1],[100,1],[100,256]
        X_star, u_star       flattened (x,t) grid and exact u,  [N*T, 2] / [N*T, 1]
        X_f                  collocation points (here: the FULL grid, X_star)
        lb, ub                domain bounds
        X_train, u_train      randomly-sampled supervised points (n_train of them)
    """
    data = scipy.io.loadmat(data_path)
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    # Collocation points: the full space-time grid (matches the original setup)
    X_f = X_star

    if seed is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X_star.shape[0], n_train, replace=False)
    else:
        idx = np.random.choice(X_star.shape[0], n_train, replace=False)

    X_train = X_star[idx, :]
    u_train = u_star[idx, :]

    return {
        'x': x, 't': t, 'Exact': Exact, 'X': X, 'T': T,
        'X_star': X_star, 'u_star': u_star, 'X_f': X_f,
        'lb': lb, 'ub': ub,
        'X_train': X_train, 'u_train': u_train, 'N_u': n_train,
    }
