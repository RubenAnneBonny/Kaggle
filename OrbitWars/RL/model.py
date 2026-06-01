"""model.py — Graph-attention policy + value network (PyTorch).

Architecture (small; the graph has <=40 nodes):
  node MLP embed -> L transformer-encoder layers with a DISTANCE BIAS added to
  attention logits (so geometry is explicit) -> node embeddings H (B, N, d).

Heads (factored per-owned-planet policy):
  target head : for each source node i, a score over all N target nodes
                (logit_ij = MLP([H_i, H_j, edge_ij])), masked to legal targets,
                plus a HOLD logit. -> categorical over (N targets + hold).
  frac head   : for each source node i, logits over N_FRAC send-fraction buckets.
  value head  : mean/max pool over real nodes -> scalar V(s).

Run `python model.py` first to confirm shapes/backward on your machine.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

N_MAX = 40
NODE_F = 10
N_FRAC = 4


class DistBiasEncoderLayer(nn.Module):
    """Transformer encoder layer whose attention logits get a learned function
    of pairwise distance added in (distance bias). edge[...,0] is normalized dist."""
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
        q = self.q(x).view(B, N, self.h, self.dk).transpose(1, 2)   # B,h,N,dk
        k = self.k(x).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)         # B,h,N,N
        dbias = self.dist_mlp(edge[..., :1]).permute(0, 3, 1, 2)     # B,h,N,N
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
        self.frac_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_FRAC))
        self.value = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.d = d

    def forward(self, node_feats, edge, node_mask, attack_mask):
        """All tensors batched (B, ...). Returns:
            tgt_logits : (B, N, N+1)  last col = HOLD
            frac_logits: (B, N, N_FRAC)
            value      : (B,)
        """
        B, N, _ = node_feats.shape
        x = self.embed(node_feats)
        for L in self.layers:
            x = L(x, edge, node_mask)

        # target logits: pair MLP over (H_i, H_j, dist_ij)
        Hi = x.unsqueeze(2).expand(B, N, N, self.d)
        Hj = x.unsqueeze(1).expand(B, N, N, self.d)
        pair = torch.cat([Hi, Hj, edge[..., :1]], dim=-1)
        tgt = self.tgt_mlp(pair).squeeze(-1)                 # B,N,N
        tgt = tgt.masked_fill(attack_mask < 0.5, float("-inf"))
        hold = self.hold(x)                                  # B,N,1
        tgt_logits = torch.cat([tgt, hold], dim=-1)          # B,N,N+1

        frac_logits = self.frac_mlp(x)                       # B,N,N_FRAC

        nm = node_mask.unsqueeze(-1)
        mean = (x * nm).sum(1) / nm.sum(1).clamp(min=1)
        mx = x.masked_fill(nm < 0.5, float("-inf")).max(1).values
        mx = torch.nan_to_num(mx, neginf=0.0)
        value = self.value(torch.cat([mean, mx], dim=-1)).squeeze(-1)
        return tgt_logits, frac_logits, value


def build_edge(node_feats):
    """Pairwise normalized distance from the (x,y) features (indices 6,7).
    Returns (B, N, N, 1)."""
    xy = node_feats[..., 6:8]                                 # B,N,2 in [-1,1]
    diff = xy.unsqueeze(2) - xy.unsqueeze(1)                  # B,N,N,2
    dist = diff.norm(dim=-1, keepdim=True)                    # B,N,N,1
    return dist


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 5, N_MAX
    nf = torch.randn(B, N, NODE_F)
    nm = (torch.rand(B, N) > 0.2).float()
    am = (torch.rand(B, N, N) > 0.5).float()
    edge = build_edge(nf)
    net = OrbitNet()
    tl, fl, v = net(nf, edge, nm, am)
    print("tgt_logits", tl.shape, "frac_logits", fl.shape, "value", v.shape)
    # backward sanity
    loss = tl.float().nan_to_num(neginf=0).sum() + fl.sum() + v.sum()
    loss.backward()
    nparams = sum(p.numel() for p in net.parameters())
    print(f"backward OK; params={nparams:,}")
