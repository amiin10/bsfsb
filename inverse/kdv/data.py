"""
Data loading for the inverse KdV benchmark (parameter identification):
same `KdV.mat` file as the forward problem, with added Gaussian
measurement noise. Like forward KdV, this setup reuses the same points
for both the data-fidelity and the PDE-residual loss terms.
"""
import numpy as np
import scipy.io

LAMBDA_1_TRUE = 1.0
LAMBDA_2_TRUE = 0.0025


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


def sample_training_set(X_star, u_star, n_train, noise_pct, seed):
    """Draw a random training subset and add Gaussian noise
    (std = noise_pct/100 * std(u_star))."""
    rng = np.random.RandomState(seed)
    u_std = float(np.std(u_star))
    sigma_noise = (noise_pct / 100.0) * u_std

    idx = rng.choice(X_star.shape[0], n_train, replace=False)
    X_train = X_star[idx, :]
    u_clean = u_star[idx, :]
    noise = sigma_noise * rng.randn(*u_clean.shape)
    u_train = (u_clean + noise).astype(np.float32)
    return X_train, u_train
