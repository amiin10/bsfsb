"""
ePINN-AF training wrapper for the INVERSE 2D Navier-Stokes (cylinder
wake) problem: given noisy measurements of (u, v) only (pressure is NOT
supervised), simultaneously learn the network AND the two PDE
coefficients lambda_1, lambda_2.

PDE (with TRAINABLE lambda_1, lambda_2):
    u_t + lambda_1*(u u_x + v u_y) + p_x - lambda_2*(u_xx + u_yy) = 0
    v_t + lambda_1*(u v_x + v v_y) + p_y - lambda_2*(v_xx + v_yy) = 0

The network predicts a stream function psi and pressure p; velocities
are recovered as u = psi_y, v = -psi_x (continuity satisfied exactly).

Loss:
    L = MSE_u + MSE_v + MSE_fu + MSE_fv
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EPINNAFNetwork
from utils import get_device

INIT_LAMBDA_1 = 0.0
INIT_LAMBDA_2 = 0.0
LR_NET = 1e-3
LR_LAMBDA = 1e-3      # learning rate for the discovered PDE coefficients
LOG_EVERY = 100        # iterations between history entries


def grad(y, x, create_graph=True):
    """First derivative dy/dx (assumes y, x require_grad)."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=create_graph, retain_graph=True,
    )[0]



def inverse_ns_residual(net, X, lambda_1, lambda_2):
    """Return u, v, p, f_u, f_v at the given collocation points X.
    X is a tensor of shape (N, 3) with columns (x, y, t).
    """
    x = X[:, 0:1].requires_grad_(True)
    y = X[:, 1:2].requires_grad_(True)
    t = X[:, 2:3].requires_grad_(True)
    Xin = torch.cat([x, y, t], dim=1)

    psip = net(Xin)
    psi = psip[:, 0:1]
    p   = psip[:, 1:2]

    # Velocity from streamfunction (incompressibility automatic)
    u =  grad(psi, y)
    v = -grad(psi, x)

    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    v_t = grad(v, t)
    v_x = grad(v, x)
    v_y = grad(v, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    p_x = grad(p, x)
    p_y = grad(p, y)

    f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
    f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
    return u, v, p, f_u, f_v


class _InverseBase:
    def __init__(self, net, X_data, u_data, v_data, X_f,
                 init_lambda_1=INIT_LAMBDA_1, init_lambda_2=INIT_LAMBDA_2):
        self.device = get_device()
        self.net = net.to(self.device)

        # Learnable PDE coefficients (registered as nn.Parameter so they
        # are visible to the optimiser and have ordinary .grad tracking)
        self.lambda_1 = nn.Parameter(
            torch.tensor(float(init_lambda_1), device=self.device))
        self.lambda_2 = nn.Parameter(
            torch.tensor(float(init_lambda_2), device=self.device))

        # Data tensors
        self.X_data = torch.as_tensor(X_data, dtype=torch.float32, device=self.device)
        self.u_data = torch.as_tensor(u_data, dtype=torch.float32, device=self.device)
        self.v_data = torch.as_tensor(v_data, dtype=torch.float32, device=self.device)
        self.X_f    = torch.as_tensor(X_f,    dtype=torch.float32, device=self.device)

        # History buffers
        self.loss_history     = []
        self.grad_norms       = []
        self.param_norms      = []
        self.lambda_1_history = []
        self.lambda_2_history = []

        # Default optimiser: all net params + the two lambdas
        self._build_optimizer()

    # ---- subclasses can override to attach extra params (e.g. SA weights) ----
    def _extra_params(self):
        return []

    def _build_optimizer(self):
        net_params = list(self.net.parameters())
        extra      = self._extra_params()
        # Lambdas get their own learning rate (Raissi-style)
        self.opt = torch.optim.Adam(
            [
                {'params': net_params, 'lr': LR_NET},
                {'params': extra,      'lr': LR_NET},
                {'params': [self.lambda_1, self.lambda_2], 'lr': LR_LAMBDA},
            ]
        )

    # ---- N-S forward at collocation points ----
    def _residuals(self, X):
        return inverse_ns_residual(self.net, X, self.lambda_1, self.lambda_2)

    # ---- variant-specific loss; subclasses override ----
    def compute_loss(self):
        # Data: predict u, v at data points; only u_data and v_data are supervised.
        u_p, v_p, _, _, _ = self._residuals(self.X_data)
        loss_u = F.mse_loss(u_p, self.u_data)
        loss_v = F.mse_loss(v_p, self.v_data)
        # PDE residual at collocation points
        _, _, _, f_u, f_v = self._residuals(self.X_f)
        loss_fu = (f_u ** 2).mean()
        loss_fv = (f_v ** 2).mean()
        loss = loss_u + loss_v + loss_fu + loss_fv
        parts = dict(u=loss_u.item(), v=loss_v.item(),
                     fu=loss_fu.item(), fv=loss_fv.item())
        return loss, parts

    # ---- one training step ----
    def step(self):
        self.opt.zero_grad()
        loss, parts = self.compute_loss()
        loss.backward()
        self.opt.step()
        return loss, parts

    # ---- main training loop ----
    def train(self, n_iter):
        t0 = time.time()
        for it in range(n_iter):
            loss, parts = self.step()

            if it % LOG_EVERY == 0 or it == n_iter - 1:
                # Track histories
                self.loss_history.append(
                    {'iter': it, 'total': loss.item(), **parts})
                self.lambda_1_history.append(float(self.lambda_1.detach().cpu()))
                self.lambda_2_history.append(float(self.lambda_2.detach().cpu()))
                # Gradient + parameter norms (over net params)
                gnorm = 0.0
                pnorm = 0.0
                for p in self.net.parameters():
                    if p.grad is not None:
                        gnorm += p.grad.detach().pow(2).sum().item()
                    pnorm += p.detach().pow(2).sum().item()
                self.grad_norms.append({'iter': it, 'g_l2': math.sqrt(gnorm)})
                self.param_norms.append({'iter': it, 'p_l2': math.sqrt(pnorm)})

                if it % (LOG_EVERY * 10) == 0:
                    print(f'[iter {it:6d}] loss={loss.item():.4e}  '
                          f'lam1={self.lambda_1.item():+.4f}  '
                          f'lam2={self.lambda_2.item():+.6f}  '
                          f'(u={parts["u"]:.2e} v={parts["v"]:.2e} '
                          f'fu={parts["fu"]:.2e} fv={parts["fv"]:.2e})')

        print(f'  ... training done in {time.time() - t0:.1f}s')

    # ---- prediction on (possibly large) batch ----
    def predict(self, X_np, batch_size=4096):
        self.net.eval()
        X = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
        us, vs, ps, fus, fvs = [], [], [], [], []
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size]
            # We do need grad for residuals — autograd-based derivatives.
            u, v, p, f_u, f_v = inverse_ns_residual(
                self.net, xb, self.lambda_1, self.lambda_2)
            us.append(u.detach().cpu().numpy())
            vs.append(v.detach().cpu().numpy())
            ps.append(p.detach().cpu().numpy())
            fus.append(f_u.detach().cpu().numpy())
            fvs.append(f_v.detach().cpu().numpy())
        self.net.train()
        return (np.concatenate(us, axis=0),
                np.concatenate(vs, axis=0),
                np.concatenate(ps, axis=0),
                np.concatenate(fus, axis=0),
                np.concatenate(fvs, axis=0))


class PhysicsInformedNN(_InverseBase):
    """ePINN-AF wrapper for the inverse Navier-Stokes problem."""
    def __init__(self, X_train, u_train, v_train, X_f,
                 input_dim, backbone_layers, n_rules, attn_hidden,
                 lb, ub, partition_dims=None, use_direct_head=True,
                 init_lambda_1=INIT_LAMBDA_1, init_lambda_2=INIT_LAMBDA_2):
        net = EPINNAFNetwork(input_dim, backbone_layers, n_rules, attn_hidden,
                             lb, ub, partition_dims=partition_dims,
                             use_direct_head=use_direct_head)
        super().__init__(net, X_train, u_train, v_train, X_f,
                         init_lambda_1=init_lambda_1, init_lambda_2=init_lambda_2)

