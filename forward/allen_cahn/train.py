#!/usr/bin/env python3
"""
Train ePINN-AF on the forward Allen-Cahn equation and save results.

Usage:
    python train.py --data /path/to/AC.mat
    python train.py --data /path/to/AC.mat --n-iter 10000
"""
import argparse
import pickle
import time

import numpy as np

from data import load_data
from pinn import PhysicsInformedNN
from utils import seed_torch

# Architecture (Table 1)
FUZZY_NODES = 4
ATTN_HIDDEN = 32
BACKBONE_HIDDEN = [200, 200, 200, 200]

N_TRAIN = 8000
N_ITER = 10000
SEED = 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True, help='Path to AC.mat')
    ap.add_argument('--n-iter', type=int, default=N_ITER, help='Adam iterations before L-BFGS')
    ap.add_argument('--n-train', type=int, default=N_TRAIN, help='Number of supervised points')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default='allen_cahn_epinn_af_results.pkl')
    args = ap.parse_args()

    seed_torch(args.seed)

    d = load_data(args.data, n_train=args.n_train)
    x, t, Exact = d['x'], d['t'], d['Exact']
    X_star, u_star = d['X_star'], d['u_star']
    lb, ub = d['lb'], d['ub']
    X_train, u_train = d['X_train'], d['u_train']

    print(f"Shape of x: {x.shape}, t: {t.shape}, Exact: {Exact.shape}")
    print(f"Total points: {X_star.shape[0]}, training points: {args.n_train}")

    pinn = PhysicsInformedNN(
        X_train, u_train,
        backbone_layers=BACKBONE_HIDDEN,
        n_rules=FUZZY_NODES,
        attn_hidden=ATTN_HIDDEN,
        lb=lb, ub=ub,
    )

    t0 = time.time()
    pinn.train(args.n_iter)
    training_time = time.time() - t0
    print(f'ePINN-AF training completed in {training_time:.2f} seconds')

    u_pred, f_pred = pinn.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    mse_f = float(np.mean(f_pred ** 2))
    print(f'Relative L2 error (u): {error_u:.6e}')
    print(f'MSE of PDE residual:   {mse_f:.6e}')

    n_t, n_x = t.shape[0], x.shape[0]
    u_pred_reshaped = u_pred.reshape((n_t, n_x))
    error_array = np.abs(Exact - u_pred_reshaped)
    max_error = float(np.max(error_array))
    print(f'Max pointwise error:   {max_error:.6e}')

    results = {
        'training_time': training_time,
        'error_u': error_u,
        'mse_f': mse_f,
        'max_error': max_error,
        'u_pred': u_pred, 'f_pred': f_pred,
        'u_pred_reshaped': u_pred_reshaped, 'error_array': error_array,
        'loss_history': pinn.loss_history,
        'grad_norms': pinn.grad_norms,
        'param_norms': pinn.param_norms,
        'X_star': X_star, 'u_star': u_star, 'Exact': Exact,
        'x': x, 't': t, 'lb': lb, 'ub': ub,
        'X_train': X_train, 'u_train': u_train, 'N_u': args.n_train,
        'backbone_hidden': BACKBONE_HIDDEN, 'fuzzy_nodes': FUZZY_NODES,
        'attn_hidden': ATTN_HIDDEN,
    }

    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved results to {args.out}')


if __name__ == "__main__":
    main()
