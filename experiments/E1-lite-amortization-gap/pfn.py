"""Small PFN and the floored 1-D logistic prior used by E1-lite.

Prior (label-DP setting, features public and bounded):
    w ~ N(0, 2^2),  b ~ N(0, 1),  x ~ U[-1, 1]
    y | x, w, b ~ Bernoulli((1 - eta) * sigmoid(w x + b) + eta / 2)
eta is the label floor of Definition A3 (K = 2); it is drawn per dataset and
given to the network as an input token, so one network covers every eta.
B_y(eta) = log(1 + 2 (1 - eta) / eta) (Lemma A4).

The exact PPD is computed on a (w, b) grid; it is differentiable in torch so
adversarial searches can target the exact mechanism as well as the network.
"""

from __future__ import annotations

import math

import torch
from torch import nn

W_SD, B_SD = 2.0, 1.0


# --------------------------------------------------------------------- prior
def sample_batch(batch: int, n_ctx: int, n_qry: int, eta_range=(0.05, 0.9), gen=None):
    """Draw `batch` datasets. Returns context (x, y), query (x, y), eta."""
    w = torch.randn(batch, 1, generator=gen) * W_SD
    b = torch.randn(batch, 1, generator=gen) * B_SD
    eta = torch.empty(batch, 1).uniform_(*eta_range, generator=gen)
    x = torch.rand(batch, n_ctx + n_qry, generator=gen) * 2 - 1
    p1 = (1 - eta) * torch.sigmoid(w * x + b) + eta / 2
    y = (torch.rand(batch, n_ctx + n_qry, generator=gen) < p1).long()
    return x[:, :n_ctx], y[:, :n_ctx], x[:, n_ctx:], y[:, n_ctx:], eta.squeeze(1)


# ----------------------------------------------------------------- the model
class PFN(nn.Module):
    """Standard PFN masking: context attends to context; queries attend to context only.

    No positional encoding, so the output is invariant to the order of context rows.
    """

    def __init__(self, d: int = 128, heads: int = 4, layers: int = 4, ff: int = 256):
        super().__init__()
        self.x_enc = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.eta_enc = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.y_emb = nn.Embedding(3, d)  # 0, 1, 2 = "query, label unknown"
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, xc, yc, xq, eta):
        """xc, yc: (B, n); xq: (B, m); eta: (B,). Returns log-probs (B, m, 2)."""
        B, n = xc.shape
        m = xq.shape[1]
        e = self.eta_enc(eta.view(B, 1, 1))
        ctx = self.x_enc(xc.unsqueeze(-1)) + self.y_emb(yc) + e
        qry = self.x_enc(xq.unsqueeze(-1)) + self.y_emb(torch.full_like(xq, 2, dtype=torch.long)) + e
        tokens = torch.cat([ctx, qry], 1)
        # True = blocked. Every token may attend only to context positions.
        if n < 1:
            raise ValueError("E1-lite PFN expects at least one context row")
        mask = torch.zeros(n + m, n + m, dtype=torch.bool, device=xc.device)
        mask[:, n:] = True
        h = self.body(tokens, mask=mask)
        return torch.log_softmax(self.head(h[:, n:]), -1)


# ------------------------------------------------------------- exact PPD
class ExactPPD:
    """Grid posterior over (w, b) for the floored logistic prior (differentiable)."""

    def __init__(self, nw: int = 241, nb: int = 81, wmax: float = 12.0, bmax: float = 4.5):
        w = torch.linspace(-wmax, wmax, nw, dtype=torch.float64)
        b = torch.linspace(-bmax, bmax, nb, dtype=torch.float64)
        W, Bm = torch.meshgrid(w, b, indexing="ij")
        self.w = W.reshape(-1)
        self.b = Bm.reshape(-1)
        self.log_prior = -0.5 * (self.w / W_SD) ** 2 - 0.5 * (self.b / B_SD) ** 2
        self.log_prior = self.log_prior - torch.logsumexp(self.log_prior, 0)

    def _logp1(self, x, eta):
        # x: (..., k) -> (..., k, G)
        s = torch.sigmoid(x.unsqueeze(-1).double() * self.w + self.b)
        return torch.log((1 - eta) * s + eta / 2), torch.log((1 - eta) * (1 - s) + eta / 2)

    def log_posterior(self, xc, yc, eta):
        """xc, yc: (n,) -> (G,) normalized log posterior."""
        if xc.numel() == 0:
            return self.log_prior
        l1, l0 = self._logp1(xc, eta)
        ll = torch.where(yc.unsqueeze(-1).bool(), l1, l0).sum(0)
        lp = self.log_prior + ll
        return lp - torch.logsumexp(lp, 0)

    def log_ppd(self, xc, yc, xq, eta):
        """Returns (m, 2) log P(y | xq, D) under the floored likelihood."""
        post = self.log_posterior(xc, yc, eta)
        l1, l0 = self._logp1(xq, eta)
        lp1 = torch.logsumexp(post + l1, -1)
        lp0 = torch.logsumexp(post + l0, -1)
        return torch.stack([lp0, lp1], -1)


def two_B_y(eta: float) -> float:
    return 2 * math.log1p(2 * (1 - eta) / eta)
