"""
ePINN-AF network architecture for the 2D Navier-Stokes (cylinder wake)
forward problem. Outputs a stream function psi and pressure p; velocities
are recovered via u = d(psi)/dy, v = -d(psi)/dx (see pinn.ns_forward),
which makes the divergence-free constraint exact by construction.
"""
import torch


class AttentionFuzzyLayer(torch.nn.Module):
    """
    Attention-enhanced fuzzy layer (Eqs 3.5 – 3.8 of the manuscript).

    Computes:
      - mu_j(z):    Multivariate diagonal Gaussian membership (Eq 3.5)
      - alpha_j(z): Softmax-normalized attention weights      (Eq 3.8)
      - gamma_j(z): mu_j * alpha_j  (combined gate per rule)

    PATCH (carried over from KdV experiments): sigma initialization narrowed
    to U(sigma_lo, sigma_hi) so each rule actually localizes on normalized
    [-1,1] inputs. Default U(0.15, 0.4) — still within the manuscript's
    U(0.1, 1.0) envelope.

    NS-SPECIFIC ENHANCEMENT — partition_dims:
      The fuzzy MEMBERSHIP μ_j is computed only on a SUBSET of input axes
      (e.g. partition_dims=[2] ⇒ partition only on time t).  The
      ATTENTION network still sees the full input z, so spatial information
      is not lost — it just doesn't drive the rule centers.

      Rationale for the cylinder wake (and any flow with strong temporal
      coherence): vortex shedding is periodic in t with one dominant
      frequency.  Partitioning rules along t alone matches the physics;
      adding x, y to the partition forces rules to specialize on spatial
      regions where the underlying flow physics is identical, which is
      counter-productive.

      partition_dims=None (default) reproduces the original behaviour
      (rules partition the full input space).
    """
    def __init__(self, input_dim, n_rules, attn_hidden,
                 sigma_lo=0.15, sigma_hi=0.40,
                 partition_dims=None):
        super().__init__()
        self.input_dim = input_dim
        self.n_rules = n_rules
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi

        # ---- partition_dims: which input axes drive μ ---------------------
        if partition_dims is None:
            partition_dims = list(range(input_dim))
        self.partition_dims = list(partition_dims)
        self.n_partition = len(self.partition_dims)
        # Register as a non-trainable buffer so it serializes & moves to device
        self.register_buffer(
            'partition_idx',
            torch.tensor(self.partition_dims, dtype=torch.long)
        )

        # Fuzzy centers c_j ∈ R^p and widths σ_j ∈ R^p (p = n_partition)
        self.centers = torch.nn.Parameter(torch.empty(n_rules, self.n_partition))
        self.log_sigma = torch.nn.Parameter(torch.empty(n_rules, self.n_partition))

        # Attention A(z; θ_A): R^d → R^M still sees FULL input
        self.attention_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, attn_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(attn_hidden, n_rules)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Section 3.3.1: c_ji ~ U(-1,1), σ_ji ~ U(0.1,1)
        with torch.no_grad():
            self.centers.uniform_(-1.0, 1.0)
            sigma_init = torch.empty_like(self.log_sigma).uniform_(
                self.sigma_lo, self.sigma_hi
            )
            self.log_sigma.copy_(torch.log(sigma_init))
        for m in self.attention_net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, z):
        # Membership μ uses only partition_dims of z   [B, p]
        z_part = z.index_select(1, self.partition_idx)
        sigma = torch.exp(self.log_sigma).clamp(min=1e-6)        # [M, p]
        diff = z_part.unsqueeze(1) - self.centers.unsqueeze(0)   # [B, M, p]
        exponent = -0.5 * torch.sum((diff / sigma.unsqueeze(0)) ** 2, dim=2)
        mu = torch.exp(exponent)                                 # [B, M]

        # Attention sees full z
        raw_scores = self.attention_net(z)                       # [B, M]
        alpha = torch.softmax(raw_scores, dim=1)                 # [B, M]
        gamma = mu * alpha                                       # [B, M]
        return mu, alpha, gamma


class AFPINN(torch.nn.Module):
    """
    ePINN-AF (Eq 3.10) for 2-D Navier-Stokes:
        û(z) = Σ_{j=1}^{M} α_j(z) · μ_j(z) · h_j(z; θ_h) + h_direct(z) + b
    Output: (ψ, p)  →  u = ∂ψ/∂y, v = -∂ψ/∂x  (computed in ns_forward).

    PATCHES carried over from the KdV experiments:
      (1) Per-rule output heads h_j(z) = W_j · h(z) — each rule contributes
          an independent projection of the shared backbone features instead
          of multiplying a scalar gate by one MLP.
      (2) Narrowed fuzzy sigma init (handled inside AttentionFuzzyLayer).
      (3) Input normalization to [-1,1] via (lb, ub), so fuzzy centers
          initialized on U(-1,1) align with the actual NS domain.

    NS-SPECIFIC ENHANCEMENTS (added because the cylinder wake at Re=100
    is a single-regime, smooth, temporally-coherent flow where the
    original 3-D fuzzy partition + pure-fuzzy output were a poor match):

      (A) DIRECT HEAD — h_direct = W_d · h(z) is added in PARALLEL to the
          fuzzy-gated sum:
              û(z) = Σ_j γ_j(z) · h_j(z) + W_d · h(z) + b
          Guarantees a clean MLP path so the network never has to "find
          its way out" of degenerate fuzzy gating to produce any output.
          Toggleable via use_direct_head=True/False.

      (B) TIME-ONLY PARTITIONING — the fuzzy MEMBERSHIP μ_j is partitioned
          only along the time axis (partition_dims=[2]) so rules
          specialize on temporal phases of vortex shedding rather than on
          spatial regions where the underlying physics is identical.
          Configurable via partition_dims (None ⇒ partition all dims).

    Architecture (Table 1 for 2-D N-S):
        Backbone: 4 hidden layers × 200 neurons, tanh on every layer
        Fuzzy rules: M = 8
        Attention hidden: N = 64
        Input dim: 3 (x, y, t)
        Output dim: 2 (ψ, p)
    """
    def __init__(self, input_dim, backbone_layers, n_rules, attn_hidden,
                 output_dim=2, sigma_lo=0.15, sigma_hi=0.40,
                 lb=None, ub=None,
                 partition_dims=None, use_direct_head=True):
        super().__init__()
        self.n_rules = n_rules
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_direct_head = use_direct_head
        self.partition_dims = (list(partition_dims)
                               if partition_dims is not None
                               else list(range(input_dim)))

        # PATCH (3): register domain bounds for input normalization
        if lb is None:
            lb = [-1.0] * input_dim
        if ub is None:
            ub = [1.0] * input_dim
        self.register_buffer('lb_buf', torch.tensor(lb, dtype=torch.float32))
        self.register_buffer('ub_buf', torch.tensor(ub, dtype=torch.float32))

        # Deep backbone h(z; θ_h) — tanh on every hidden layer
        backbone_sizes = [input_dim] + backbone_layers
        layers = []
        for i in range(len(backbone_sizes) - 1):
            layers.append(torch.nn.Linear(backbone_sizes[i], backbone_sizes[i + 1]))
            layers.append(torch.nn.Tanh())
        self.backbone = torch.nn.Sequential(*layers)
        self.backbone_out_dim = backbone_layers[-1]

        # Attention-Fuzzy layer (with optional partition_dims)
        self.fuzzy_attention = AttentionFuzzyLayer(
            input_dim=input_dim, n_rules=n_rules, attn_hidden=attn_hidden,
            sigma_lo=sigma_lo, sigma_hi=sigma_hi,
            partition_dims=self.partition_dims,
        )

        # PATCH (1): per-rule heads h_j(z) = W_j · h(z)
        self.rule_heads = torch.nn.Linear(
            self.backbone_out_dim, n_rules * output_dim, bias=False
        )

        # NS ENHANCEMENT (A): DIRECT HEAD — backbone-only path that bypasses
        # the fuzzy gate entirely.  Output is added to the fuzzy sum.
        if self.use_direct_head:
            self.direct_head = torch.nn.Linear(
                self.backbone_out_dim, output_dim, bias=False
            )
        else:
            self.direct_head = None

        # Trainable bias b (per-output)
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.backbone:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        torch.nn.init.xavier_uniform_(self.rule_heads.weight)
        if self.direct_head is not None:
            torch.nn.init.xavier_uniform_(self.direct_head.weight)

    def _normalize(self, z):
        """PATCH (3): map input from [lb, ub] to [-1, 1]."""
        return 2.0 * (z - self.lb_buf) / (self.ub_buf - self.lb_buf) - 1.0

    def forward(self, z, return_components=False):
        z_n = self._normalize(z)
        h = self.backbone(z_n)                                     # [B, D]
        mu, alpha, gamma = self.fuzzy_attention(z_n)               # each [B, M]
        per_rule = self.rule_heads(h).view(-1, self.n_rules, self.output_dim)
        # Eq 3.10:  fuzzy_out = Σ_j γ_j(z) · h_j(z)
        fuzzy_out = (gamma.unsqueeze(-1) * per_rule).sum(dim=1)    # [B, output_dim]

        # NS ENHANCEMENT (A): direct backbone path
        if self.direct_head is not None:
            direct_out = self.direct_head(h)                       # [B, output_dim]
            u_hat = fuzzy_out + direct_out + self.bias
        else:
            direct_out = None
            u_hat = fuzzy_out + self.bias

        if return_components:
            comps = {
                "h": h, "mu": mu, "alpha": alpha, "gamma": gamma,
                "per_rule": per_rule, "fuzzy_out": fuzzy_out,
                "direct_out": direct_out,
            }
            return u_hat, comps
        return u_hat

