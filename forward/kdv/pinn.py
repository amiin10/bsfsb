"""
ePINN-AF training wrapper for the (forward) Korteweg-de Vries (KdV) equation.

PDE:
    u_t + u * u_x + 0.0025 * u_xxx = 0

Loss (Eq 3.22):
    L = MSE_u + MSE_f

No adaptive weighting -- the fuzzy-attention mechanism inside the network
itself handles adaptive focus (Eq 3.10).
"""
import numpy as np
import torch

from model import AFPINN
from utils import get_device


class PhysicsInformedNN:
    """ePINN-AF training wrapper for the KdV equation.

    Args:
        X: training coordinates [N, 2] with columns [x, t]
        u: training solution values [N, 1]
        backbone_layers: list of hidden layer widths, e.g. [200, 200, 200, 200]
        n_rules: number of fuzzy rules M
        attn_hidden: hidden neurons in the attention subnetwork
        lb, ub: lower/upper bounds of the domain
    """
    def __init__(self, X, u, backbone_layers, n_rules, attn_hidden, lb, ub):
        self.device = get_device()
        self.lb = torch.tensor(lb).float().to(self.device)
        self.ub = torch.tensor(ub).float().to(self.device)

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        self.u = torch.tensor(u).float().to(self.device)

        self.model = AFPINN(
            backbone_layers=backbone_layers,
            n_rules=n_rules,
            attn_hidden=attn_hidden,
            output_dim=1,
            input_dim=2,
            sigma_lo=0.1,
            sigma_hi=1,
            lb=lb, ub=ub,
        ).to(self.device)

        self.optimizer_Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=100000,
            max_eval=100000,
            history_size=100,
            tolerance_grad=1e-12,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        self.loss_history = {'Adam': [], 'LBFGS': [], 'data': [], 'pde': []}
        self.grad_norms = {'Adam': [], 'LBFGS': []}
        self.param_norms = {'Adam': [], 'LBFGS': []}
        self.iter = 0

    def net_u(self, x, t):
        """Compute the predicted solution u_hat(x, t)."""
        return self.model(torch.cat([x, t], dim=1))

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx),
                                    retain_graph=True, create_graph=True)[0]
        return u_t + u * u_x + 0.0025 * u_xxx

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
        """Train with Adam pre-training followed by L-BFGS refinement (Table 1)."""
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
            if epoch % 1000 == 0:
                print(f'Adam Epoch: {epoch}, Loss: {loss.item():.4e}, '
                      f'L_data: {self.loss_history["data"][-1]:.4e}, '
                      f'L_pde: {self.loss_history["pde"][-1]:.4e}')

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

    def predict(self, X, batch_size=8192):
        """Batched prediction to avoid OOM on the full grid."""
        self.model.eval()
        n = X.shape[0]
        u_pred = np.zeros((n, 1), dtype=np.float32)
        f_pred = np.zeros((n, 1), dtype=np.float32)

        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            X_batch = X[i:end]

            x = torch.tensor(X_batch[:, 0:1], requires_grad=True, dtype=torch.float32, device=self.device)
            t = torch.tensor(X_batch[:, 1:2], requires_grad=True, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                u_pred_batch = self.net_u(x, t)

            f_pred_batch = self.net_f(x, t)

            u_pred[i:end] = u_pred_batch.detach().cpu().numpy()
            f_pred[i:end] = f_pred_batch.detach().cpu().numpy()

            del x, t, u_pred_batch, f_pred_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return u_pred, f_pred
