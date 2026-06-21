"""
Data loading for the inverse Burgers benchmark (parameter identification):
same `burgers_shock.mat` file as the forward problem, with added Gaussian
measurement noise. Collocation points (`X_f`) reuse the full space-time
grid, same as the forward setup.
"""
import numpy as np
import scipy.io

LAMBDA_1_TRUE = 1.0
LAMBDA_2_TRUE = 0.01 / np.pi


def load_data(data_path):
    data = scipy.io.loadmat(data_path)
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    return {'x': x, 't': t, 'Exact': Exact, 'X_star': X_star, 'u_star': u_star,
            'lb': lb, 'ub': ub}


def sample_training_set(X_star, u_star, n_train, noise_pct, seed):
    """Draw a random training subset, add Gaussian noise (std = noise_pct/100
    * std(u_star)), and use the FULL grid as the collocation set X_f.
    """
    rng = np.random.RandomState(seed)
    u_std = float(np.std(u_star))
    sigma_noise = (noise_pct / 100.0) * u_std

    idx = rng.choice(X_star.shape[0], n_train, replace=False)
    X_train = X_star[idx, :]
    u_clean = u_star[idx, :]
    noise = sigma_noise * rng.randn(*u_clean.shape)
    u_train = (u_clean + noise).astype(np.float32)

    X_f = X_star
    return X_train, u_train, X_f
