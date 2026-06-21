"""
ePINN-AF training wrapper for the Poisson boundary-value problem.

Loss (different structure from the time-dependent PDEs -- this is a BVP):
    L(theta) = MSE_b + MSE_r
    MSE_b = (1/N_b) sum |u_theta(x_b) - g(x_b)|^2     boundary fidelity
    MSE_r = (1/N_r) sum |Delta u_theta(x_r) - f(x_r)|^2   interior residual
"""
import numpy as np
import torch

from model import AFPINN
from utils import get_device
from data import laplacian, forcing


def _common_optim_init(self, lr_adam=1e-3, tol_grad=1e-7):
    self.optimizer_Adam = torch.optim.Adam(self.model.parameters(), lr=lr_adam)
    self.optimizer_LBFGS = torch.optim.LBFGS(
        self.model.parameters(),
        lr=1.0, max_iter=100000, max_eval=100000, history_size=100,
        tolerance_grad=tol_grad,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe",
    )
    self.loss_history = {'Adam': [], 'LBFGS': [], 'data': [], 'pde': []}
    self.grad_norms = {'Adam': [], 'LBFGS': []}
    self.param_norms = {'Adam': [], 'LBFGS': []}
    self.iter = 0


def _compute_grad_norms(model):
    g = 0.0
    p = 0.0
    for param in model.parameters():
        if param.grad is not None:
            g += param.grad.data.norm(2).item() ** 2
        p += param.data.norm(2).item() ** 2
    return g ** 0.5, p ** 0.5


class PhysicsInformedNN:
    """ePINN-AF wrapper for the Poisson BVP. Loss = MSE_b + MSE_r."""

    def __init__(self, Z_b, u_b, Z_r, dim, lb, ub,
                 backbone_layers, n_rules, attn_hidden):
        self.device = get_device()
        self.dim = dim
        self.lb = torch.tensor(lb).float().to(self.device)
        self.ub = torch.tensor(ub).float().to(self.device)
        self.z_b = torch.tensor(Z_b, requires_grad=False).float().to(self.device)
        self.u_b = torch.tensor(u_b).float().to(self.device)
        self.z_r = torch.tensor(Z_r, requires_grad=True).float().to(self.device)

        self.model = AFPINN(
            backbone_layers=backbone_layers, n_rules=n_rules,
            attn_hidden=attn_hidden, output_dim=1, input_dim=dim,
            sigma_lo=0.15, sigma_hi=0.40, lb=lb, ub=ub,
        ).to(self.device)
        _common_optim_init(self)

    def net_u(self, z):
        return self.model(z)

    def net_f(self, z):
        u = self.net_u(z)
        lap = laplacian(u, z, self.dim)
        return lap - forcing(z, self.dim)  # 0 at the solution

    def loss_fn(self):
        u_pred_b = self.net_u(self.z_b)
        z_r = self.z_r.requires_grad_(True)
        f_pred = self.net_f(z_r)
        mse_u = torch.mean((self.u_b - u_pred_b) ** 2)
        mse_f = torch.mean(f_pred ** 2)
        loss = mse_u + mse_f
        self.loss_history['data'].append(mse_u.item())
        self.loss_history['pde'].append(mse_f.item())
        return loss

    def train(self, n_iter):
        self.model.train()
        for epoch in range(n_iter):
            loss = self.loss_fn()
            self.optimizer_Adam.zero_grad()
            loss.backward()
            g, p = _compute_grad_norms(self.model)
            self.grad_norms['Adam'].append(g)
            self.param_norms['Adam'].append(p)
            self.optimizer_Adam.step()
            self.loss_history['Adam'].append(loss.item())
            if epoch % 1000 == 0:
                print(f'[ePINN-AF] Adam {epoch:6d}  L={loss.item():.4e}  '
                      f'L_b={self.loss_history["data"][-1]:.4e}  '
                      f'L_r={self.loss_history["pde"][-1]:.4e}')

        def closure():
            loss = self.loss_fn()
            self.optimizer_LBFGS.zero_grad()
            loss.backward()
            g, p = _compute_grad_norms(self.model)
            self.grad_norms['LBFGS'].append(g)
            self.param_norms['LBFGS'].append(p)
            self.iter += 1
            self.loss_history['LBFGS'].append(loss.item())
            if self.iter % 1000 == 0:
                print(f'[ePINN-AF] LBFGS {self.iter:6d}  L={loss.item():.4e}')
            return loss

        self.optimizer_LBFGS.step(closure)

    def predict(self, Z):
        z = torch.tensor(Z, requires_grad=True).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            u_pred = self.net_u(z)
        f_pred = self.net_f(z)
        return u_pred.cpu().numpy(), f_pred.detach().cpu().numpy()
