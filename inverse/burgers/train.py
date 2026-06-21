#!/usr/bin/env python3
"""
Recover the Burgers PDE parameters (lambda_1, lambda_2) with ePINN-AF from
noisy measurements of u, and save results.

Usage:
    python train.py --data /path/to/burgers_shock.mat
    python train.py --data /path/to/burgers_shock.mat --noise 3 --n-iter 10000
"""
import argparse
import pickle
import time

import numpy as np

from data import load_data, sample_training_set, LAMBDA_1_TRUE, LAMBDA_2_TRUE
from pinn import PhysicsInformedNN
from utils import seed_torch

FUZZY_NODES = 4
ATTN_HIDDEN = 32
BACKBONE_HIDDEN = [100, 100, 100, 100]

N_TRAIN = 2000
N_ITER = 10000
SEED = 113


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True, help='Path to burgers_shock.mat')
    ap.add_argument('--noise', type=float, default=0.0,
                    help='Measurement noise, in percent of std(u) (default: 0)')
    ap.add_argument('--n-iter', type=int, default=N_ITER)
    ap.add_argument('--n-train', type=int, default=N_TRAIN)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default='burgers_inverse_epinn_af_results.pkl')
    args = ap.parse_args()

    seed_torch(args.seed)

    d = load_data(args.data)
    X_star, u_star = d['X_star'], d['u_star']
    lb, ub = d['lb'], d['ub']
    print(f"Total points: {X_star.shape[0]}")
    print(f"True parameters: lambda_1={LAMBDA_1_TRUE}, lambda_2={LAMBDA_2_TRUE:.6f}")
    print(f"Noise level: {args.noise}%")

    X_train, u_train, X_f = sample_training_set(
        X_star, u_star, args.n_train, args.noise, seed=args.seed)


    pinn = PhysicsInformedNN(
        X_train, u_train, X_f,
        input_dim=2, backbone_layers=BACKBONE_HIDDEN,
        n_rules=FUZZY_NODES, attn_hidden=ATTN_HIDDEN,
        lb=lb, ub=ub,
    )

    t0 = time.time()
    pinn.train(args.n_iter)
    training_time = time.time() - t0
    print(f'ePINN-AF training completed in {training_time:.2f} seconds')

    u_pred, f_pred = pinn.predict(X_star)
    error_u = float(np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2))
    mse_f = float(np.mean(f_pred ** 2))

    l1_hat = float(pinn.lambda_1.item())
    l2_hat = float(pinn.lambda_2.item())
    err_l1 = abs(l1_hat - LAMBDA_1_TRUE) / abs(LAMBDA_1_TRUE) * 100.0
    err_l2 = abs(l2_hat - LAMBDA_2_TRUE) / abs(LAMBDA_2_TRUE) * 100.0

    print(f'Relative L2 error (u): {error_u:.6e}')
    print(f'Recovered lambda_1 = {l1_hat:.5f}  (true {LAMBDA_1_TRUE}, error {err_l1:.3f}%)')
    print(f'Recovered lambda_2 = {l2_hat:.6f}  (true {LAMBDA_2_TRUE:.6f}, error {err_l2:.3f}%)')

    results = {
        'noise_pct': args.noise,
        'training_time': training_time,
        'error_u': error_u, 'mse_f': mse_f,
        'lambda_1_hat': l1_hat, 'lambda_2_hat': l2_hat,
        'lambda_1_true': LAMBDA_1_TRUE, 'lambda_2_true': LAMBDA_2_TRUE,
        'lambda_1_err_pct': err_l1, 'lambda_2_err_pct': err_l2,
        'u_pred': u_pred, 'f_pred': f_pred,
        'loss_history': pinn.loss_history,
        'lambda_history': pinn.lambda_history,
        'grad_norms': pinn.grad_norms,
        'param_norms': pinn.param_norms,
        'X_star': X_star, 'u_star': u_star,
        'X_train': X_train, 'u_train': u_train, 'X_f': X_f,
        'lb': lb, 'ub': ub,
        'backbone_hidden': BACKBONE_HIDDEN, 'fuzzy_nodes': FUZZY_NODES,
        'attn_hidden': ATTN_HIDDEN,
    }


    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved results to {args.out}')


if __name__ == "__main__":
    main()
