"""
ePINN-AF network architecture for the inverse 2D Navier-Stokes (cylinder
wake) problem: a softmax-attention-gated mixture of Gaussian fuzzy rules
over a shared MLP backbone, predicting a stream function `psi` and
pressure `p`. This is a from-scratch re-implementation (not shared with
the forward-NS `AFPINN`) written directly in terms of `nn.ModuleList`
rule heads plus an optional sigmoid-gated direct head.
"""
import torch
import torch.nn as nn


def xavier_linear(in_f, out_f):
    layer = nn.Linear(in_f, out_f)
    nn.init.xavier_normal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _normalise(X, lb, ub):
    """Map X in [lb, ub] to [-1, 1]."""
    return 2.0 * (X - lb) / (ub - lb) - 1.0


class EPINNAFNetwork(nn.Module):
    """Attention-Fuzzy PINN backbone.
    backbone: shared MLP
    rules:    n_rules Gaussian rules over the chosen partition dims
    attn:     small MLP turns backbone features into attention logits
    head:     linear map from (rule-attended features) -> (psi, p)
    """
    def __init__(self, input_dim, backbone_layers, n_rules, attn_hidden,
                 lb, ub, partition_dims=None, use_direct_head=True):
        super().__init__()
        self.register_buffer('lb', torch.as_tensor(lb, dtype=torch.float32))
        self.register_buffer('ub', torch.as_tensor(ub, dtype=torch.float32))
        self.n_rules = n_rules
        self.use_direct_head = use_direct_head
        self.partition_dims = (list(partition_dims)
                               if partition_dims is not None
                               else list(range(input_dim)))
        # Shared backbone
        self.backbone = nn.ModuleList()
        prev = input_dim
        for h in backbone_layers:
            self.backbone.append(xavier_linear(prev, h))
            prev = h
        self.feat_dim = prev
        # Fuzzy rule centers/widths along the partition dims
        pdim = len(self.partition_dims)
        self.rule_centers = nn.Parameter(torch.linspace(-1, 1, n_rules)
                                         .unsqueeze(1).expand(n_rules, pdim).clone())
        self.rule_log_sigmas = nn.Parameter(torch.zeros(n_rules, pdim) - 0.5)
        # Attention head: backbone features -> n_rules attention logits
        self.attn = nn.Sequential(
            xavier_linear(self.feat_dim, attn_hidden), nn.Tanh(),
            xavier_linear(attn_hidden, n_rules),
        )
        # Rule consequents: each rule has its own (psi, p) prediction from features
        self.rule_heads = nn.ModuleList(
            [xavier_linear(self.feat_dim, 2) for _ in range(n_rules)])
        # Optional direct head: predicts (psi, p) from features straight
        if use_direct_head:
            self.direct_head = xavier_linear(self.feat_dim, 2)
            self.gate_direct = nn.Parameter(torch.tensor(0.0))

    def forward(self, X):
        H = _normalise(X, self.lb, self.ub)
        # backbone
        feat = H
        for lin in self.backbone:
            feat = torch.tanh(lin(feat))
        # rule firing strengths from partition coords
        Xn = _normalise(X, self.lb, self.ub)
        Xp = Xn[:, self.partition_dims].unsqueeze(1)              # (N, 1, P)
        c = self.rule_centers.unsqueeze(0)                        # (1, K, P)
        sig = torch.exp(self.rule_log_sigmas).unsqueeze(0)        # (1, K, P)
        mu = torch.exp(-((Xp - c) / sig).pow(2)).prod(dim=-1)     # (N, K)
        mu = mu / (mu.sum(dim=-1, keepdim=True) + 1e-8)
        # attention logits scaled by fuzzy weights
        attn_logits = self.attn(feat)                             # (N, K)
        attn = torch.softmax(attn_logits, dim=-1) * mu
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        # rule outputs
        rule_outs = torch.stack(
            [head(feat) for head in self.rule_heads], dim=-1)     # (N, 2, K)
        out = (rule_outs * attn.unsqueeze(1)).sum(dim=-1)         # (N, 2)
        if self.use_direct_head:
            g = torch.sigmoid(self.gate_direct)
            out = (1.0 - g) * out + g * self.direct_head(feat)
        return out
