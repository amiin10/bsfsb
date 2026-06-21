"""
ePINN-AF training wrapper for the 2D Navier-Stokes (cylinder wake) forward
problem.

PDE (incompressible momentum, with known LAMBDA_1, LAMBDA_2):
    u_t + LAMBDA_1*(u u_x + v u_y) + p_x - LAMBDA_2*(u_xx + u_yy) = 0
    v_t + LAMBDA_1*(u v_x + v v_y) + p_y - LAMBDA_2*(v_xx + v_yy) = 0

The network predicts a stream function psi and pressure p; velocities are
u = psi_y, v = -psi_x, which satisfies the continuity equation u_x + v_y = 0
exactly, by construction (see `ns_forward` below).

Loss:
    L = MSE_u + MSE_v + MSE_p + MSE_fu + MSE_fv
"""
import numpy as np
import torch

from model import AFPINN
from utils import get_device

LAMBDA_1 = 1.0
LAMBDA_2 = 0.01


def ns_forward(model, x, y, t):
    """
    Compute (u, v, p, f_u, f_v) given a network that outputs (ψ, p).
    u = ∂ψ/∂y,  v = -∂ψ/∂x  ⇒  u_x + v_y ≡ 0  (incompressibility built in).
    """
    z = torch.cat([x, y, t], dim=1)
    out = model(z)                                # [B, 2]
    psi = out[:, 0:1]
    p = out[:, 1:2]

    psi_x = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi),
                                retain_graph=True, create_graph=True)[0]
    psi_y = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi),
                                retain_graph=True, create_graph=True)[0]
    u = psi_y
    v = -psi_x

    # Velocity derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                              retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                              retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                              retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                               retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y),
                               retain_graph=True, create_graph=True)[0]

    # Pressure derivatives
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                              retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p),
                              retain_graph=True, create_graph=True)[0]

    # Momentum residuals (Eq 4.9, forward setting with known λ₁, λ₂)
    f_u = u_t + LAMBDA_1 * (u * u_x + v * u_y) + p_x - LAMBDA_2 * (u_xx + u_yy)
    f_v = v_t + LAMBDA_1 * (u * v_x + v * v_y) + p_y - LAMBDA_2 * (v_xx + v_yy)

    return u, v, p, f_u, f_v


class NSBase:
    """
    Base class for all N-S training wrappers.
    Stores (x, y, t) data points + (u, v, p) targets + (x_f, y_f, t_f) collocation
    points on the device.  Subclasses create self.model and implement loss_fn.

    Provides shared helpers:
      - net_u(x, y, t)        : returns network output (ψ, p)
      - net_u_scalar(x, y, t) : returns u-velocity [N, 1]
      - net_f_scalar(x, y, t) : returns f_u residual [N, 1]
      - net_f(x, y, t)        : returns (f_u, f_v) tuple
      - predict(X)            : batched (u, v, p, f_u, f_v) on full grid
    """
    def __init__(self, X, u, v, p, X_f, lb, ub):
        self.device = get_device()
        self.lb = torch.tensor(lb).float().to(self.device)
        self.ub = torch.tensor(ub).float().to(self.device)

        # Observation points (with supervision on u, v, p)
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        self.y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        self.t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(self.device)
        self.u = torch.tensor(u).float().to(self.device)
        self.v = torch.tensor(v).float().to(self.device)
        self.p = torch.tensor(p).float().to(self.device)

        # Collocation points (PDE residual only)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(self.device)
        self.y_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(self.device)
        self.t_f = torch.tensor(X_f[:, 2:3], requires_grad=True).float().to(self.device)

        self.loss_history = {
            'Adam': [], 'LBFGS': [],
            'data_uv': [], 'data_p': [], 'data': [], 'pde': [], 'total': []
        }
        self.grad_norms = {'Adam': [], 'LBFGS': []}
        self.param_norms = {'Adam': [], 'LBFGS': []}
        self.iter = 0

    # --------------- shared autograd-aware helpers ---------------------------
    def net_u(self, x, y, t):
        """Return raw (ψ, p) network output, [N, 2]."""
        return self.model(torch.cat([x, y, t], dim=1))

    def net_f(self, x, y, t):
        """Return (f_u, f_v), the two momentum-equation residuals."""
        _, _, _, fu, fv = ns_forward(self.model, x, y, t)
        return fu, fv

    def net_u_scalar(self, x, y, t):
        """Return u-velocity [N, 1]."""
        u, _, _, _, _ = ns_forward(self.model, x, y, t)
        return u

    def net_f_scalar(self, x, y, t):
        """Return f_u residual [N, 1]."""
        _, _, _, fu, _ = ns_forward(self.model, x, y, t)
        return fu

    # --------------- gradient-norm helper ------------------------------------
    def compute_grad_norms(self):
        tg, tp = 0.0, 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                tg += p.grad.data.norm(2).item() ** 2
            tp += p.data.norm(2).item() ** 2
        return tg ** 0.5, tp ** 0.5

    # --------------- batched prediction --------------------------------------
    def predict(self, X, batch_size=1024):
        """Batched prediction for (u, v, p, f_u, f_v) on full N×T grid."""
        self.model.eval()
        u_preds, v_preds, p_preds, fu_preds, fv_preds = [], [], [], [], []
        for i in range(0, len(X), batch_size):
            Xb = X[i:i + batch_size]
            xb = torch.tensor(Xb[:, 0:1], requires_grad=True).float().to(self.device)
            yb = torch.tensor(Xb[:, 1:2], requires_grad=True).float().to(self.device)
            tb = torch.tensor(Xb[:, 2:3], requires_grad=True).float().to(self.device)
            u, v, p, fu, fv = ns_forward(self.model, xb, yb, tb)
            u_preds.append(u.detach().cpu().numpy())
            v_preds.append(v.detach().cpu().numpy())
            p_preds.append(p.detach().cpu().numpy())
            fu_preds.append(fu.detach().cpu().numpy())
            fv_preds.append(fv.detach().cpu().numpy())
        return (np.concatenate(u_preds), np.concatenate(v_preds),
                np.concatenate(p_preds), np.concatenate(fu_preds),
                np.concatenate(fv_preds))



class PhysicsInformedNN(NSBase):
    """
    ePINN-AF training wrapper for 2-D Navier-Stokes.

    Loss (Eq 3.22 extended to two momentum residuals + p data):
        L = MSE_u + MSE_v + MSE_p + MSE_fu + MSE_fv

    No adaptive weighting — the fuzzy-attention mechanism inside the network
    itself handles adaptive focus (Eq 3.10).

    NS-SPECIFIC SWITCHES (forwarded to AFPINN):
      partition_dims:    which input axes drive fuzzy memberships.
                         Default [2] = time-only partitioning (recommended
                         for cylinder wake — see AFPINN docstring).
                         Pass None to partition along all axes (the
                         original ePINN-AF behaviour from the manuscript).
      use_direct_head:   add a parallel backbone-only output path so the
                         network always has a clean MLP route (default True).
    """
    def __init__(self, X, u, v, p, X_f, input_dim, backbone_layers,
                 n_rules, attn_hidden, lb, ub,
                 partition_dims=(2,), use_direct_head=True):
        super().__init__(X, u, v, p, X_f, lb, ub)

        self.model = AFPINN(
            input_dim=input_dim,
            backbone_layers=backbone_layers,
            n_rules=n_rules,
            attn_hidden=attn_hidden,
            output_dim=2,                 # (ψ, p)
            sigma_lo=0.15,
            sigma_hi=0.40,
            lb=lb, ub=ub,
            partition_dims=(list(partition_dims)
                            if partition_dims is not None else None),
            use_direct_head=use_direct_head,
        ).to(self.device)

        self.optimizer_Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0, max_iter=100000, max_eval=100000,
            history_size=100, tolerance_grad=1e-12,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

    def loss_fn(self):
        # Data fidelity at observation points
        u_pred, v_pred, p_pred, _, _ = ns_forward(self.model, self.x, self.y, self.t)
        MSE_u = torch.mean((self.u - u_pred) ** 2)
        MSE_v = torch.mean((self.v - v_pred) ** 2)
        MSE_p = torch.mean((self.p - p_pred) ** 2)

        # PDE residual at collocation points
        _, _, _, fu, fv = ns_forward(self.model, self.x_f, self.y_f, self.t_f)
        MSE_fu = torch.mean(fu ** 2)
        MSE_fv = torch.mean(fv ** 2)

        mse_data = MSE_u + MSE_v + MSE_p
        mse_pde = MSE_fu + MSE_fv
        loss = mse_data + mse_pde

        self.loss_history['data_uv'].append((MSE_u + MSE_v).item())
        self.loss_history['data_p'].append(MSE_p.item())
        self.loss_history['data'].append(mse_data.item())
        self.loss_history['pde'].append(mse_pde.item())
        self.loss_history['total'].append(loss.item())
        return loss

    def train(self, n_iter):
        self.model.train()

        # Phase 1: Adam
        for epoch in range(n_iter):
            loss = self.loss_fn()

            self.optimizer_Adam.zero_grad()
            loss.backward()
            g, p = self.compute_grad_norms()
            self.grad_norms['Adam'].append(g)
            self.param_norms['Adam'].append(p)

            self.optimizer_Adam.step()
            self.loss_history['Adam'].append(loss.item())

            if epoch % 1000 == 0:
                print(f'Adam Epoch: {epoch}, Loss: {loss.item():.4e}, '
                      f'L_uv: {self.loss_history["data_uv"][-1]:.4e}, '
                      f'L_p: {self.loss_history["data_p"][-1]:.4e}, '
                      f'L_pde: {self.loss_history["pde"][-1]:.4e}')

        # Phase 2: L-BFGS
        def closure():
            loss = self.loss_fn()
            self.optimizer_LBFGS.zero_grad()
            loss.backward()
            g, p = self.compute_grad_norms()
            self.grad_norms['LBFGS'].append(g)
            self.param_norms['LBFGS'].append(p)
            self.iter += 1
            self.loss_history['LBFGS'].append(loss.item())
            if self.iter % 1000 == 0:
                print(f'LBFGS Iter: {self.iter}, Loss: {loss.item():.4e}, '
                      f'L_uv: {self.loss_history["data_uv"][-1]:.4e}, '
                      f'L_p: {self.loss_history["data_p"][-1]:.4e}, '
                      f'L_pde: {self.loss_history["pde"][-1]:.4e}')
            return loss

        self.optimizer_LBFGS.step(closure)

