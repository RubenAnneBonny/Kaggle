"""model.py (v2) — Graph-attention policy + value network.

Changes from v1:
  * NODE_F 10 -> 11: added an explicit signed NET-ATTACK feature per planet
    (incoming-own minus incoming-enemy ships). The two halves existed before,
    but handing the net directly is cheap and slightly easier to use.
  * The fraction head is now CONTINUOUS. Instead of 4 discrete buckets it
    outputs the (alpha, beta) of a Beta distribution over [0, 1] — "what
    fraction of this planet's available ships to send". Beta is the right family
    for a bounded action: proper PPO log-prob + entropy, no boundary blow-ups.
    At decode time we use the Beta MEAN (greedy) or a sample (exploration);
    a value below FRAC_FLOOR (see ow_env) means "send nothing".
  * Target head (N targets + HOLD) is unchanged — HOLD stays the clean discrete
    "act or not / where"; the fraction only says "how much, given we act".

Heads:
  target head : per source i, score over N targets (+HOLD). Categorical.
  frac head   : per source i, (alpha, beta) > 1 for a Beta over [0,1].
  value head  : mean/max pool over real nodes -> scalar V(s).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

N_MAX = 40
NODE_F = 11           # v2: +1 for net-attack
FRAC_EPS = 1e-3       # clamp Beta samples/targets away from {0,1} for stable logp


class DistBiasEncoderLayer(nn.Module):
    """Transformer encoder layer with a learned distance bias on attention."""
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads; self.d = d; self.dk = d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.dist_mlp = nn.Sequential(nn.Linear(1, heads), nn.Tanh(), nn.Linear(heads, heads))
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)

    def forward(self, x, edge, node_mask):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        dbias = self.dist_mlp(edge[..., :1]).permute(0, 3, 1, 2)
        att = att + dbias
        keymask = (node_mask < 0.5).view(B, 1, 1, N)
        att = att.masked_fill(keymask, float("-inf"))
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, N, self.d)
        x = self.n1(x + self.o(out))
        x = self.n2(x + self.ff(x))
        return x


class OrbitNet(nn.Module):
    def __init__(self, d=64, heads=4, layers=3):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(NODE_F, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([DistBiasEncoderLayer(d, heads) for _ in range(layers)])
        self.tgt_mlp = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1))
        self.hold = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # frac head now outputs 2 raw params -> softplus(+1) -> Beta(alpha, beta)
        self.frac_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.value = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.d = d

    def forward(self, node_feats, edge, node_mask, attack_mask):
        """Returns:
            tgt_logits : (B, N, N+1)  last col = HOLD
            frac_ab    : (B, N, 2)    Beta (alpha, beta), both > 1
            value      : (B,)
        """
        B, N, _ = node_feats.shape
        x = self.embed(node_feats)
        for L in self.layers:
            x = L(x, edge, node_mask)

        Hi = x.unsqueeze(2).expand(B, N, N, self.d)
        Hj = x.unsqueeze(1).expand(B, N, N, self.d)
        pair = torch.cat([Hi, Hj, edge[..., :1]], dim=-1)
        tgt = self.tgt_mlp(pair).squeeze(-1)                 # B,N,N
        tgt = tgt.masked_fill(attack_mask < 0.5, float("-inf"))
        hold = self.hold(x)                                  # B,N,1
        tgt_logits = torch.cat([tgt, hold], dim=-1)          # B,N,N+1

        frac_ab = F.softplus(self.frac_mlp(x)) + 1.0         # B,N,2  (alpha,beta > 1)

        nm = node_mask.unsqueeze(-1)
        mean = (x * nm).sum(1) / nm.sum(1).clamp(min=1)
        mx = x.masked_fill(nm < 0.5, float("-inf")).max(1).values
        mx = torch.nan_to_num(mx, neginf=0.0)
        value = self.value(torch.cat([mean, mx], dim=-1)).squeeze(-1)
        return tgt_logits, frac_ab, value


def frac_dist(frac_ab):
    """Build a Beta distribution from the head output (B,N,2)."""
    a = frac_ab[..., 0].clamp(min=1.0 + 1e-3)
    b = frac_ab[..., 1].clamp(min=1.0 + 1e-3)
    return torch.distributions.Beta(a, b)


def build_edge(node_feats):
    """Pairwise normalized distance from the (x,y) features (indices 6,7)."""
    xy = node_feats[..., 6:8]
    diff = xy.unsqueeze(2) - xy.unsqueeze(1)
    dist = diff.norm(dim=-1, keepdim=True)
    return dist


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 5, N_MAX
    nf = torch.randn(B, N, NODE_F)
    nm = (torch.rand(B, N) > 0.2).float()
    am = (torch.rand(B, N, N) > 0.5).float()
    edge = build_edge(nf)
    net = OrbitNet()
    tl, fab, v = net(nf, edge, nm, am)
    print("tgt_logits", tl.shape, "frac_ab", fab.shape, "value", v.shape)
    fd = frac_dist(fab)
    print("frac mean range", float(fd.mean.min()), float(fd.mean.max()))
    loss = tl.float().nan_to_num(neginf=0).sum() + fab.sum() + v.sum()
    loss.backward()
    nparams = sum(p.numel() for p in net.parameters())
    print(f"backward OK; params={nparams:,}")
