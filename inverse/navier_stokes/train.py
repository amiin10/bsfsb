#!/usr/bin/env python3
"""
Recover the Navier-Stokes PDE parameters (lambda_1, lambda_2) with
ePINN-AF from noisy velocity measurements (pressure is not supervised),
and save results.

Usage:
    python train.py --data /path/to/cylinder_nektar_wake.mat
    python train.py --data /path/to/cylinder_nektar_wake.mat --noise 0.01 --n-iter 20000
"""
import argparse
import pickle
import time

import numpy as np

from data import load_data
from pinn import PhysicsInformedNN
from utils import seed_torch, get_device

TRUE_LAMBDA_1 = 1.0
TRUE_LAMBDA_2 = 0.01

FUZZY_NODES = 8
ATTN_HIDDEN = 64
BACKBONE_HIDDEN = [200, 200, 200, 200]

# partition_dims=[2] => time-only fuzzy partitioning (see forward/navier_stokes)
PARTITION_DIMS = [2]
USE_DIRECT_HEAD = True

N_TRAIN = 5000
N_ITER = 20000
SEED = 42


def relL2(ref, pred):
    return np.linalg.norm(ref - pred, 2) / np.linalg.norm(ref, 2)


def velL2(u_ref, v_ref, u_p, v_p):
    num = np.sqrt(np.linalg.norm(u_ref - u_p, 2) ** 2
                  + np.linalg.norm(v_ref - v_p, 2) ** 2)
    den = np.sqrt(np.linalg.norm(u_ref, 2) ** 2 + np.linalg.norm(v_ref, 2) ** 2)
    return num / den


def shift_p(p_pred, p_true):
    """Pressure is defined up to a constant — match means before comparison."""
    return p_pred - np.mean(p_pred) + np.mean(p_true)


def lambda_error_pct(pred, true):
    return 100.0 * abs(pred - true) / abs(true)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True, help='Path to cylinder_nektar_wake.mat')
    ap.add_argument('--noise', type=float, default=0.0,
                    help='Measurement noise, as a FRACTION of std(u)/std(v), '
                         'e.g. 0.01 = 1%% (default: 0)')
    ap.add_argument('--n-iter', type=int, default=N_ITER)
    ap.add_argument('--n-train', type=int, default=N_TRAIN)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default='ns_inverse_epinn_af_results.pkl')
    args = ap.parse_args()

    seed_torch(args.seed)

    print(f"True parameters : lambda_1 = {TRUE_LAMBDA_1}, lambda_2 = {TRUE_LAMBDA_2}")
    print(f"Initial guesses : lambda_1 = 0.0, lambda_2 = 0.0")
    print(f"Noise level     : {args.noise * 100:.1f}% Gaussian on (u, v)")
    print(f"Device          : {get_device()}")

    d = load_data(args.data, n_train=args.n_train, noise_level=args.noise, seed=args.seed)
    X_star_full, u, v, p_true = d['X_star_full'], d['u'], d['v'], d['p_true']
    UU, VV, PP = d['UU'], d['VV'], d['PP']
    N, T = d['N'], d['T']
    lb, ub = d['lb'], d['ub']
    X_train, u_train, v_train = d['X_train'], d['u_train'], d['v_train']
    X_f = d['X_f']

    print(f"Dataset: N={N} spatial x T={T} time, total {N * T} points")
    print(f"Training set: {args.n_train} (u, v) points — pressure NOT supervised")
    print(f"Collocation set: {X_f.shape[0]} points")

    wrapper = PhysicsInformedNN(
        X_train, u_train, v_train, X_f,
        input_dim=3, backbone_layers=BACKBONE_HIDDEN,
        n_rules=FUZZY_NODES, attn_hidden=ATTN_HIDDEN,
        lb=lb, ub=ub,
        partition_dims=PARTITION_DIMS, use_direct_head=USE_DIRECT_HEAD,
    )

    print("\n=== Training ePINN-AF (inverse) ===")
    t0 = time.time()
    wrapper.train(args.n_iter)
    training_time = time.time() - t0
    print(f'ePINN-AF training completed in {training_time:.2f} seconds')

    print("\n=== Predicting on full dataset ===")
    u_pred, v_pred, p_pred, fu_pred, fv_pred = wrapper.predict(X_star_full)
    p_pred = shift_p(p_pred, p_true)

    u_NT = u_pred.reshape((N, T))
    v_NT = v_pred.reshape((N, T))
    p_NT = p_pred.reshape((N, T))

    error_u = relL2(u, u_pred)
    error_v = relL2(v, v_pred)
    error_p = relL2(p_true, p_pred)
    error_vel = velL2(u, v, u_pred, v_pred)

    err_u_arr = np.abs(UU - u_NT)
    err_v_arr = np.abs(VV - v_NT)
    err_p_arr = np.abs(PP - p_NT)
    err_vel_arr = np.sqrt(err_u_arr ** 2 + err_v_arr ** 2)

    mse_fu = float(np.mean(fu_pred ** 2))
    mse_fv = float(np.mean(fv_pred ** 2))

    lambda_1_pred = float(wrapper.lambda_1.detach().cpu().item())
    lambda_2_pred = float(wrapper.lambda_2.detach().cpu().item())
    err_lambda_1 = lambda_error_pct(lambda_1_pred, TRUE_LAMBDA_1)
    err_lambda_2 = lambda_error_pct(lambda_2_pred, TRUE_LAMBDA_2)

    print(f'\nDiscovered lambda_1 = {lambda_1_pred:+.6f}  '
          f'(true {TRUE_LAMBDA_1}, error {err_lambda_1:.4f}%)')
    print(f'Discovered lambda_2 = {lambda_2_pred:+.6f}  '
          f'(true {TRUE_LAMBDA_2}, error {err_lambda_2:.4f}%)')
    print(f'Rel L2: u={error_u:.4e}, v={error_v:.4e}, p={error_p:.4e}, vel={error_vel:.4e}')
    print(f'MSE PDE: fu={mse_fu:.4e}, fv={mse_fv:.4e}, total={mse_fu + mse_fv:.4e}')
    print(f'Time:   {training_time:.2f} s')

    results = {
        'noise_level': args.noise,
        'training_time': training_time,
        'lambda_1_pred': lambda_1_pred, 'lambda_2_pred': lambda_2_pred,
        'lambda_1_true': TRUE_LAMBDA_1, 'lambda_2_true': TRUE_LAMBDA_2,
        'err_lambda_1_pct': err_lambda_1, 'err_lambda_2_pct': err_lambda_2,
        'lambda_1_history': list(wrapper.lambda_1_history),
        'lambda_2_history': list(wrapper.lambda_2_history),
        'error_u': error_u, 'error_v': error_v, 'error_p': error_p, 'error_vel': error_vel,
        'u_pred': u_pred, 'v_pred': v_pred, 'p_pred': p_pred,
        'fu_pred': fu_pred, 'fv_pred': fv_pred,
        'mse_fu': mse_fu, 'mse_fv': mse_fv, 'mse_f': mse_fu + mse_fv,
        'u_pred_reshaped': u_NT, 'v_pred_reshaped': v_NT, 'p_pred_reshaped': p_NT,
        'error_u_array': err_u_arr, 'error_v_array': err_v_arr,
        'error_p_array': err_p_arr, 'error_vel_array': err_vel_arr,
        'loss_history': wrapper.loss_history,
        'grad_norms': wrapper.grad_norms,
        'param_norms': wrapper.param_norms,
        'X_star': X_star_full, 'u_star': u, 'v_star': v, 'p_star': p_true,
        'Exact_u': UU, 'Exact_v': VV, 'Exact_p': PP,
        'lb': lb, 'ub': ub, 'N': N, 'T': T,
        'X_train': X_train, 'u_train': u_train, 'v_train': v_train, 'X_f': X_f,
        'N_train': args.n_train,
        'backbone_hidden': BACKBONE_HIDDEN, 'fuzzy_nodes': FUZZY_NODES,
        'attn_hidden': ATTN_HIDDEN,
        'partition_dims': PARTITION_DIMS, 'use_direct_head': USE_DIRECT_HEAD,
    }

    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved results to {args.out}')


if __name__ == "__main__":
    main()
