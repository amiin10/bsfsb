#!/usr/bin/env python3
"""
Train ePINN-AF on the 2D or 3D Poisson BVP and save results.

Usage:
    python train.py --dim 2
    python train.py --dim 3 --n-iter 10000
"""
import argparse
import pickle
import time

import numpy as np
import torch

from data import sample_points, make_test_grid, analytical_solution
from pinn import PhysicsInformedNN
from utils import seed_torch

FUZZY_NODES = 8
ATTN_HIDDEN = 64
BACKBONE_HIDDEN = [5, 5, 5, 5]

N_ITER = 10000
SEED = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dim', type=int, default=2, choices=[2, 3], help='Problem dimension')
    ap.add_argument('--n-iter', type=int, default=N_ITER)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    seed_torch(args.seed)
    DIM = args.dim
    out_path = args.out or f'poisson_{DIM}d_epinn_af_results.pkl'

    # Domain Omega = [-5, 5]^d
    lb = np.array([-5.0] * DIM, dtype=np.float32)
    ub = np.array([5.0] * DIM, dtype=np.float32)

    if DIM == 2:
        N_b, N_r = 2000, 4000
        n_test_per_axis = 10  # 10^2 = 100 test points
    else:
        N_b, N_r = 1500, 3000
        n_test_per_axis = 41  # 41^3 = 68921 test points

    print(f"=== Poisson {DIM}D ===")
    print(f"Domain: {lb.tolist()} -> {ub.tolist()}")
    print(f"Boundary points N_b = {N_b}, interior points N_r = {N_r}")

    Z_b, u_b, Z_r = sample_points(N_b, N_r, DIM, lb, ub, seed=0)
    Z_test, mesh = make_test_grid(DIM, lb, ub, n_test_per_axis)
    u_star = analytical_solution(torch.from_numpy(Z_test), DIM).numpy()
    print(f"Test grid: {Z_test.shape[0]} points")

    def rel_l2(u_pred):
        return float(np.linalg.norm(u_star - u_pred) / np.linalg.norm(u_star))

    pinn = PhysicsInformedNN(
        Z_b, u_b, Z_r, dim=DIM, lb=lb, ub=ub,
        backbone_layers=BACKBONE_HIDDEN,
        n_rules=FUZZY_NODES, attn_hidden=ATTN_HIDDEN,
    )

    t0 = time.time()
    pinn.train(args.n_iter)
    training_time = time.time() - t0
    print(f'ePINN-AF training: {training_time:.2f} s')

    u_pred, f_pred = pinn.predict(Z_test)
    error_u = rel_l2(u_pred)
    mse_f = float(np.mean(f_pred ** 2))
    print(f'Relative L2 error: {error_u:.4e}')
    print(f'MSE of PDE residual: {mse_f:.4e}')

    results = {
        'dim': DIM,
        'training_time': training_time,
        'error_u': error_u,
        'mse_f': mse_f,
        'u_pred': u_pred, 'f_pred': f_pred,
        'u_star': u_star, 'Z_test': Z_test, 'mesh': mesh,
        'Z_b': Z_b, 'u_b': u_b, 'Z_r': Z_r,
        'lb': lb, 'ub': ub,
        'loss_history': pinn.loss_history,
        'grad_norms': pinn.grad_norms,
        'param_norms': pinn.param_norms,
        'backbone_hidden': BACKBONE_HIDDEN, 'fuzzy_nodes': FUZZY_NODES,
        'attn_hidden': ATTN_HIDDEN,
    }

    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved results to {out_path}')


if __name__ == "__main__":
    main()
