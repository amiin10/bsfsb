"""
ePINN-AF training wrapper for the (forward) Burgers equation.

PDE (Eq 4.6):
    u_t + u * u_x - (0.01/pi) * u_xx = 0

Loss (Eq 3.22):
    L = MSE_u + MSE_f
"""
import numpy as np
import torch

from model import AFPINN
from utils import get_device


class PhysicsInformedNN:
    """ePINN-AF wrapper for Burgers equation."""

    def __init__(self, X, u, X_f, input_dim, backbone_layers, n_rules, attn_hidden, lb, ub):
        self.device = get_device()
        self.lb = torch.tensor(lb).float().to(self.device)
        self.ub = torch.tensor(ub).float().to(self.device)

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        self.u = torch.tensor(u).float().to(self.device)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(self.device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(self.device)

        self.model = AFPINN(
            input_dim=input_dim,
            backbone_layers=backbone_layers,
            n_rules=n_rules,
            attn_hidden=attn_hidden,
            output_dim=1
        ).to(self.device)

        self.optimizer_Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=100000,
            max_eval=100000,
            history_size=100,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.loss_history = {'Adam': [], 'LBFGS': [], 'data': [], 'pde': []}
        self.grad_norms = {'Adam': [], 'LBFGS': []}
        self.param_norms = {'Adam': [], 'LBFGS': []}
        self.iter = 0

    def compute_grad_norms(self):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_param_norm += p.data.norm(2).item() ** 2
        return total_grad_norm ** 0.5, total_param_norm ** 0.5

    def net_u(self, x, t):
        return self.model(torch.cat([x, t], dim=1))

    def net_f(self, x, t):
        """Burgers residual: u_t + u*u_x - (0.01/pi)*u_xx = 0"""
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]
        return u_t + u * u_x - (0.01 / np.pi) * u_xx

    def loss_fn(self):
        u_pred = self.net_u(self.x, self.t)
        f_pred = self.net_f(self.x_f, self.t_f)
        mse_u = torch.mean((self.u - u_pred) ** 2)
        mse_f = torch.mean(f_pred ** 2)
        loss = mse_u + mse_f
        self.loss_history['data'].append(mse_u.item())
        self.loss_history['pde'].append(mse_f.item())
        return loss

    def train(self, n_iter):
        self.model.train()

        # ---- Phase 1: Adam ----
        for epoch in range(n_iter):
            loss = self.loss_fn()

            self.optimizer_Adam.zero_grad()
            loss.backward()
            grad_norm, param_norm = self.compute_grad_norms()
            self.grad_norms['Adam'].append(grad_norm)
            self.param_norms['Adam'].append(param_norm)

            self.optimizer_Adam.step()
            self.loss_history['Adam'].append(loss.item())

            if epoch % 1000 == 0:
                print(f'Adam Epoch: {epoch}, Loss: {loss.item():.4e}, '
                      f'L_data: {self.loss_history["data"][-1]:.4e}, '
                      f'L_pde: {self.loss_history["pde"][-1]:.4e}')

        # ---- Phase 2: L-BFGS ----
        def closure():
            loss = self.loss_fn()
            self.optimizer_LBFGS.zero_grad()
            loss.backward()
            grad_norm, param_norm = self.compute_grad_norms()
            self.grad_norms['LBFGS'].append(grad_norm)
            self.param_norms['LBFGS'].append(param_norm)
            self.iter += 1
            self.loss_history['LBFGS'].append(loss.item())
            if self.iter % 1000 == 0:
                print(f'LBFGS Iter: {self.iter}, Loss: {loss.item():.4e}, '
                      f'L_data: {self.loss_history["data"][-1]:.4e}, '
                      f'L_pde: {self.loss_history["pde"][-1]:.4e}')
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
