"""
Data loading for the inverse Allen-Cahn benchmark (parameter
identification): same `AC.mat` file as the forward problem, but here we
also inject Gaussian measurement noise to test robustness of the
recovered lambda_1, lambda_2.
"""
import numpy as np
import scipy.io

LAMBDA_1_TRUE = 0.0001
LAMBDA_2_TRUE = 5.0


def load_data(data_path):
    data = scipy.io.loadmat(data_path)
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['uu']).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    return {'x': x, 't': t, 'Exact': Exact, 'X_star': X_star, 'u_star': u_star,
            'lb': lb, 'ub': ub}


def add_gaussian_noise(u_clean, noise_pct, rng):
    """Add zero-mean Gaussian noise with std = (noise_pct/100) * std(u_clean)."""
    if noise_pct <= 0:
        return u_clean.copy()
    sigma = (noise_pct / 100.0) * np.std(u_clean)
    return u_clean + rng.normal(0.0, sigma, size=u_clean.shape).astype(u_clean.dtype)


def sample_training_set(X_star, u_star, n_train, noise_pct, seed):
    """Draw a random training subset and add measurement noise."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_star.shape[0], n_train, replace=False)
    X_train = X_star[idx, :]
    u_train_clean = u_star[idx, :]
    u_train_noisy = add_gaussian_noise(u_train_clean, noise_pct, rng)
    return X_train, u_train_noisy
