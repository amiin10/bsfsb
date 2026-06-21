"""
ePINN-AF network architecture — inverse Allen-Cahn equation (parameter identification).

Two pieces:
    AttentionFuzzyLayer  Eqs 3.5-3.8  (Gaussian fuzzy membership x softmax
                                       attention -> per-rule gate gamma_j)
    AFPINN               Eq 3.10      (gate-weighted backbone feature map
                                       -> linear output head)
"""
import torch


class AttentionFuzzyLayer(torch.nn.Module):
    """
    Attention-enhanced fuzzy layer (Eqs 3.5 - 3.8).

    Computes:
      - mu_j(z):    multivariate diagonal Gaussian membership   (Eq 3.5)
      - alpha_j(z): softmax-normalized attention weights        (Eq 3.8)
      - gamma_j(z): mu_j * alpha_j  (combined gate per rule)
    """
    def __init__(self, input_dim, n_rules, attn_hidden=32):
        super().__init__()
        self.input_dim = input_dim
        self.n_rules = n_rules

        # Fuzzy rule centers/widths: [n_rules, input_dim]
        self.centers = torch.nn.Parameter(torch.empty(n_rules, input_dim))
        self.log_sigma = torch.nn.Parameter(torch.empty(n_rules, input_dim))

        # Eq 3.7: a(z) = W2 * tanh(W1*z + b1) + b2
        self.attention_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, attn_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(attn_hidden, n_rules)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Sec. 3.3.1: c_ji ~ U(-1,1), sigma_ji ~ U(0.1,1)
        with torch.no_grad():
            self.centers.uniform_(-1.0, 1.0)
            sigma_init = torch.empty_like(self.log_sigma).uniform_(0.1, 1.0)
            self.log_sigma.copy_(torch.log(sigma_init))
        for m in self.attention_net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, z):
        # Eq 3.5: multivariate diagonal Gaussian
        sigma = torch.exp(self.log_sigma).clamp(min=1e-6)
        diff = z.unsqueeze(1) - self.centers.unsqueeze(0)
        exponent = -0.5 * torch.sum((diff / sigma.unsqueeze(0)) ** 2, dim=2)
        mu = torch.exp(exponent)

        # Eq 3.8: softmax attention
        raw_scores = self.attention_net(z)
        alpha = torch.softmax(raw_scores, dim=1)

        gamma = mu * alpha
        return mu, alpha, gamma


class AFPINN(torch.nn.Module):
    """
    ePINN-AF (Eq 3.10):
        u_hat(z) = [sum_j alpha_j(z) * mu_j(z)] * h(z; theta_h) + b

    - h(z) is a SHARED latent vector in R^D from the deep backbone.
    - All hidden layers use tanh (Eq 3.9, Table 1).
    - Backbone outputs R^D (e.g. 200), not M.
    """
    def __init__(self, input_dim, backbone_layers, n_rules, attn_hidden=32, output_dim=1):
        super().__init__()
        self.n_rules = n_rules

        # Deep backbone h(z; theta_h) -- tanh on EVERY hidden layer
        backbone_sizes = [input_dim] + backbone_layers
        layers = []
        for i in range(len(backbone_sizes) - 1):
            layers.append(torch.nn.Linear(backbone_sizes[i], backbone_sizes[i + 1]))
            layers.append(torch.nn.Tanh())
        self.backbone = torch.nn.Sequential(*layers)
        self.backbone_out_dim = backbone_layers[-1]

        # Attention-Fuzzy layer
        self.fuzzy_attention = AttentionFuzzyLayer(
            input_dim=input_dim, n_rules=n_rules, attn_hidden=attn_hidden
        )

        # Output: R^D -> R^output_dim
        self.output_linear = torch.nn.Linear(self.backbone_out_dim, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, z, return_components=False):
        h = self.backbone(z)
        mu, alpha, gamma = self.fuzzy_attention(z)

        # Eq 3.10: u_hat = [sum_j alpha_j * mu_j] * h(z) + b
        gate = gamma.sum(dim=1, keepdim=True)
        gated_h = gate * h
        u_hat = self.output_linear(gated_h) + self.bias

        if return_components:
            return u_hat, {"h": h, "mu": mu, "alpha": alpha, "gamma": gamma, "gate": gate}
        return u_hat
