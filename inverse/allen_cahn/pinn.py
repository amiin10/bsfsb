"""
ePINN-AF training wrapper for the INVERSE Allen-Cahn problem: given noisy
supervised data on `u`, simultaneously learn the network AND the two PDE
coefficients lambda_1 (diffusion) and lambda_2 (reaction), which are
trainable parameters alongside the network weights.

PDE (with TRAINABLE lambda_1, lambda_2):
    u_t - lambda_1 * u_xx + lambda_2 * u^3 - lambda_2 * u = 0

True values used to generate data: lambda_1 = 0.0001, lambda_2 = 5.0
"""
import numpy as np
import torch

from model import AFPINN
from utils import get_device


class _InverseMixin:
    """Mixin holding the learnable PDE parameters and history bookkeeping.
    Subclasses must call _init_inverse() after registering self.model and
    self.optimizer_*.
    """

    def _init_inverse(self, lambda_1_init=0.0, lambda_2_init=0.0):
        self.lambda_1 = torch.nn.Parameter(
            torch.tensor([lambda_1_init], dtype=torch.float32, device=self.device))
        self.lambda_2 = torch.nn.Parameter(
            torch.tensor([lambda_2_init], dtype=torch.float32, device=self.device))

        self.optimizer_Adam.add_param_group({'params': [self.lambda_1, self.lambda_2]})
        # L-BFGS doesn't support add_param_group cleanly; rebuild with the
        # full parameter list (network weights + lambdas).
        all_params = list(self.model.parameters()) + [self.lambda_1, self.lambda_2]
        self.optimizer_LBFGS = torch.optim.LBFGS(
            all_params,
            lr=1.0, max_iter=50000, max_eval=50000, history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        self.lambda_history = {'Adam_l1': [], 'Adam_l2': [],
                               'LBFGS_l1': [], 'LBFGS_l2': []}


def _ac_residual(u, u_t, u_xx, lambda_1, lambda_2):
    """Allen-Cahn residual with TRAINABLE lambda_1, lambda_2."""
    return u_t - lambda_1 * u_xx + lambda_2 * (u ** 3) - lambda_2 * u


class PhysicsInformedNN(_InverseMixin):
    """ePINN-AF, inverse Allen-Cahn problem."""

    def __init__(self, X, u, backbone_layers, n_rules, attn_hidden, lb, ub):
        self.device = get_device()
        self.lb = torch.tensor(lb).float().to(self.device)
        self.ub = torch.tensor(ub).float().to(self.device)
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        self.u = torch.tensor(u).float().to(self.device)

        self.model = AFPINN(input_dim=2, backbone_layers=backbone_layers, n_rules=n_rules,
                            attn_hidden=attn_hidden, output_dim=1).to(self.device)
        self.optimizer_Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # Placeholder L-BFGS — rebuilt by _init_inverse() to include lambdas
        self.optimizer_LBFGS = torch.optim.LBFGS(self.model.parameters(), lr=1.0)

        self.loss_history = {'Adam': [], 'LBFGS': [], 'data': [], 'pde': []}
        self.grad_norms = {'Adam': [], 'LBFGS': []}
        self.param_norms = {'Adam': [], 'LBFGS': []}
        self.iter = 0

        self._init_inverse()

    def net_u(self, x, t):
        return self.model(torch.cat([x, t], dim=1))

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]
        return _ac_residual(u, u_t, u_xx, self.lambda_1, self.lambda_2)

    def loss_fn(self):
        u_pred = self.net_u(self.x, self.t)
        f_pred = self.net_f(self.x, self.t)
        mse_u = torch.mean((self.u - u_pred) ** 2)
        mse_f = torch.mean(f_pred ** 2)
        loss = mse_u + mse_f
        self.loss_history['data'].append(mse_u.item())
        self.loss_history['pde'].append(mse_f.item())
        return loss

    def compute_grad_norms(self):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_param_norm += param.data.norm(2).item() ** 2
        return total_grad_norm ** 0.5, total_param_norm ** 0.5

    def train(self, n_iter):
        self.model.train()
        for epoch in range(n_iter):
            loss = self.loss_fn()

            self.optimizer_Adam.zero_grad()
            loss.backward()
            grad_norm, param_norm = self.compute_grad_norms()
            self.grad_norms['Adam'].append(grad_norm)
            self.param_norms['Adam'].append(param_norm)

            self.optimizer_Adam.step()
            self.loss_history['Adam'].append(loss.item())
            self.lambda_history['Adam_l1'].append(self.lambda_1.item())
            self.lambda_history['Adam_l2'].append(self.lambda_2.item())

            if epoch % 1000 == 0:
                print(f'ePINN-AF Adam Epoch: {epoch}, Loss: {loss.item():.4e}, '
                      f'lam1: {self.lambda_1.item():.6e}, lam2: {self.lambda_2.item():.4f}')

        def closure():
            loss = self.loss_fn()
            self.optimizer_LBFGS.zero_grad()
            loss.backward()
            grad_norm, param_norm = self.compute_grad_norms()
            self.grad_norms['LBFGS'].append(grad_norm)
            self.param_norms['LBFGS'].append(param_norm)
            self.iter += 1
            self.loss_history['LBFGS'].append(loss.item())
            self.lambda_history['LBFGS_l1'].append(self.lambda_1.item())
            self.lambda_history['LBFGS_l2'].append(self.lambda_2.item())
            if self.iter % 1000 == 0:
                print(f'ePINN-AF LBFGS Iter: {self.iter}, Loss: {loss.item():.4e}, '
                      f'lam1: {self.lambda_1.item():.6e}, lam2: {self.lambda_2.item():.4f}')
            return loss

        self.optimizer_LBFGS.step(closure)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            u_pred = self.net_u(x, t)
        f_pred = self.net_f(x, t)
        return u_pred.cpu().numpy(), f_pred.detach().cpu().numpy()
