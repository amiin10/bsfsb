"""
ePINN-AF network architecture — forward Korteweg-de Vries (KdV) equation.

This version uses the "per-rule output head" formulation:
    u_hat(z) = sum_j alpha_j(z) * mu_j(z) * h_j(z; theta_h) + b
where h_j(z) = W_j . h(z) is a rule-specific linear projection of a SHARED
backbone feature vector h(z) (instead of one scalar gate times the whole
backbone, as in the simpler Burgers/Allen-Cahn variant). Inputs are also
normalised to [-1, 1] via the domain bounds (lb, ub) before being passed to
both the backbone and the fuzzy-attention layer.
"""
import torch


class AttentionFuzzyLayer(torch.nn.Module):
    """Eqs 3.5-3.8: mu_j (Gaussian membership), alpha_j (softmax attention),
    gamma_j = mu_j * alpha_j. Inputs are assumed normalised to [-1, 1]^d.
    """
    def __init__(self, input_dim, n_rules, attn_hidden, sigma_lo=0.1, sigma_hi=1):
        super().__init__()
        self.input_dim = input_dim
        self.n_rules = n_rules
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi
        self.centers = torch.nn.Parameter(torch.empty(n_rules, input_dim))
        self.log_sigma = torch.nn.Parameter(torch.empty(n_rules, input_dim))
        self.attention_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, attn_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(attn_hidden, n_rules),
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.centers.uniform_(-1.0, 1.0)
            sigma_init = torch.empty_like(self.log_sigma).uniform_(self.sigma_lo, self.sigma_hi)
            self.log_sigma.copy_(torch.log(sigma_init))
        for m in self.attention_net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, z):
        sigma = torch.exp(self.log_sigma).clamp(min=1e-6)            # [M, d]
        diff = z.unsqueeze(1) - self.centers.unsqueeze(0)            # [B, M, d]
        exponent = -0.5 * torch.sum((diff / sigma.unsqueeze(0)) ** 2, dim=2)
        mu = torch.exp(exponent)                                      # [B, M]
        raw_scores = self.attention_net(z)                           # [B, M]
        alpha = torch.softmax(raw_scores, dim=1)                     # [B, M]
        gamma = mu * alpha                                            # [B, M]
        return mu, alpha, gamma


class AFPINN(torch.nn.Module):
    """ePINN-AF (Eq 3.10) with per-rule output heads:
        u_hat(z) = sum_j alpha_j(z) mu_j(z) h_j(z) + b,
        h_j(z) = W_j . h(z)   (rule-indexed projection of shared backbone features)

    Domain bounds (lb, ub) are used to map inputs into [-1, 1]^d before
    both the backbone and the fuzzy-attention layer see them.

    Architecture (Table 1, 1D problems):
        Backbone: [2, 200, 200, 200, 200]  (4 hidden layers, tanh everywhere)
        Fuzzy rules: M = 8 (KdV default)
        Attention hidden: N = 64
    """
    def __init__(self, backbone_layers, n_rules, attn_hidden,
                 output_dim=1, input_dim=2,
                 sigma_lo=0.1, sigma_hi=1,
                 lb=None, ub=None):
        super().__init__()
        self.n_rules = n_rules
        self.input_dim = input_dim
        self.output_dim = output_dim

        if lb is None:
            lb = [-1.0] * input_dim
        if ub is None:
            ub = [1.0] * input_dim
        self.register_buffer('lb', torch.tensor(lb, dtype=torch.float32))
        self.register_buffer('ub', torch.tensor(ub, dtype=torch.float32))

        backbone_sizes = [input_dim] + backbone_layers
        layers = []
        for i in range(len(backbone_sizes) - 1):
            layers.append(torch.nn.Linear(backbone_sizes[i], backbone_sizes[i + 1]))
            layers.append(torch.nn.Tanh())
        self.backbone = torch.nn.Sequential(*layers)
        self.backbone_out_dim = backbone_layers[-1]

        self.fuzzy_attention = AttentionFuzzyLayer(
            input_dim=input_dim, n_rules=n_rules, attn_hidden=attn_hidden,
            sigma_lo=sigma_lo, sigma_hi=sigma_hi
        )

        # Per-rule output heads h_j(z) = W_j . h(z)
        self.rule_heads = torch.nn.Linear(
            self.backbone_out_dim, n_rules * output_dim, bias=False
        )

        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.backbone:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        torch.nn.init.xavier_uniform_(self.rule_heads.weight)

    def _normalize(self, z):
        return 2.0 * (z - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, z, return_components=False):
        z_n = self._normalize(z)
        h = self.backbone(z_n)                                       # [B, D]
        mu, alpha, gamma = self.fuzzy_attention(z_n)                 # each [B, M]
        per_rule = self.rule_heads(h).view(-1, self.n_rules, self.output_dim)
        u_hat = (gamma.unsqueeze(-1) * per_rule).sum(dim=1) + self.bias

        if return_components:
            return u_hat, {"h": h, "mu": mu, "alpha": alpha,
                           "gamma": gamma, "per_rule": per_rule}
        return u_hat
