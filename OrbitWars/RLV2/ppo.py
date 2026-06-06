"""ppo.py (v2) — Stage 2: PPO self-play fine-tuning, warm-started from a BC ckpt.

What changed from v1 (and why):
  * CONTINUOUS FRACTION. The fraction head is a Beta over [0,1]. Its log-prob now
    counts for EVERY non-HOLD source, not just friendly ferries: v1 ignored the
    fraction for captures (capture_size sized them with how_many_send), so the
    sampled frac was a no-op there. In v2 the network's fraction sizes every
    launch, so it must receive PPO credit everywhere it acts.
  * N-PLAYER. Trains a FOCAL player (slot 0); the other 1 or 3 slots are filled
    by the opponent. Pick --players 2 or 4 — you train one model for each, since
    the strategies differ, and submit_agent dispatches on player count.
  * FLEXIBLE OPPONENTS, incl. frozen checkpoints (current/future AI):
      self                 self-play vs a periodically-refreshed frozen copy
      teacher              net_roi_support        (scripted)
      aggressive           net_roi_aggressive     (scripted, the BC teacher)
      <name>               any agent in ow_base.py
      ckpt:PATH.pt         a frozen policy loaded from PATH (e.g. a past best)
      pool:A.pt,B.pt,...   sample one of these frozen policies per slot/episode
  * 500-STEP CAP (engine truth) and the env's faithful comets.
  * SEED DISCIPLINE. Training draws from seeds.train_seed; the benchmark draws
    from seeds.eval_seed. They live in disjoint pools, so eval games can never be
    trained on — the v1 leak is impossible by construction.

The stability fixes from v1 are retained verbatim: minibatched updates (step
once per minibatch, not per state), per-planet ratio clipping, approx-KL early
stop, one-shot value warmup.

Usage:
  python ppo.py --players 2 --init bc2_best.pt --opponent self --iters 2000 --out ppo2.pt
  python ppo.py --players 4 --init bc4_best.pt --opponent self --iters 2000 --out ppo4.pt
  # train against a frozen earlier champion:
  python ppo.py --players 2 --init bc2_best.pt --opponent ckpt:ppo2_best.pt --out ppo2b.pt
"""
import argparse, contextlib, importlib.util, io, glob, os, re, numpy as np, torch, torch.nn.functional as F
import parallel
from ow_env import OrbitEnv, encode_state, decode_action, N_MAX, FRAC_FLOOR
from model import OrbitNet, build_edge, frac_dist, FRAC_EPS
from seeds import train_seed, eval_seed
import ow_base

HOLD = N_MAX

# Decode knobs, set once in main(); read by every decode call site.
_DECODE = {"coord_cap": 2.0, "hard_skip": False}


def _dec(enc, obs, player, tgt_np, frac_np):
    return decode_action(enc, obs, player, tgt_np, frac_np,
                         frac_floor=FRAC_FLOOR,
                         coord_cap=_DECODE["coord_cap"],
                         hard_skip=_DECODE["hard_skip"])


def _frac_active_mask(enc, tgt_np):
    """Per owned source: 1.0 iff its chosen target is non-HOLD and legal — i.e.
    the fraction head actually sized a send there. In v2 that's EVERY attack and
    ferry (no capture_size shortcut), so the fraction gets PPO credit wherever it
    drove the move; HOLD and illegal picks are excluded."""
    own = enc["own_mask"]; am = enc["attack_mask"]
    out = np.zeros(N_MAX, np.float32)
    for i in range(N_MAX):
        if own[i] == 0.0:
            continue
        j = int(tgt_np[i])
        if j == HOLD or j < 0 or j >= N_MAX:
            continue
        if am[i, j] == 1.0:
            out[i] = 1.0
    return out


def _fwd(net, enc, dev):
    nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(dev)
    nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(dev)
    am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(dev)
    edge = build_edge(nf)
    return nf, nm, am, edge


def policy_act(net, obs, player, dev, greedy=False):
    """Sample (or argmax/mean) an action for one observation. Returns engine
    moves + (logp, value, training cache). Collection runs under no_grad."""
    enc = encode_state(obs, player)
    nf, nm, am, edge = _fwd(net, enc, dev)
    with torch.no_grad():
        tgt_logits, frac_ab, value = net(nf, edge, nm, am)
    tgt_logits = tgt_logits[0]; frac_ab = frac_ab[0]          # (N,N+1),(N,2)

    own = torch.tensor(enc["own_mask"], dtype=torch.bool, device=dev)
    td = torch.distributions.Categorical(logits=tgt_logits)
    fd = frac_dist(frac_ab)
    if greedy:
        tgt = tgt_logits.argmax(-1)
        frac = fd.mean
    else:
        tgt = td.sample()
        frac = fd.sample()
    frac = frac.clamp(FRAC_EPS, 1 - FRAC_EPS)

    tgt_np = tgt.detach().cpu().numpy()
    frac_np = frac.detach().cpu().numpy()
    frac_active = torch.tensor(_frac_active_mask(enc, tgt_np), dtype=torch.bool, device=dev)

    logp_t = td.log_prob(tgt)
    logp_f = fd.log_prob(frac)
    logp_per = logp_t * own.float() + logp_f * frac_active.float()
    logp = logp_per.sum()

    moves = _dec(enc, obs, player, tgt_np, frac_np)
    # collapse diagnostics (per owned source): how often it picks HOLD, the target
    # head's entropy (mode sharpness), and the Beta concentration alpha+beta.
    with torch.no_grad():
        own_f = own.float(); n_own = own_f.sum().clamp(min=1)
        diag = dict(n_own=float(own.float().sum()),
                    hold_frac=float(((tgt == HOLD).float() * own_f).sum() / n_own),
                    ent_cat=float((td.entropy() * own_f).sum() / n_own),
                    conc=float((frac_ab.sum(-1) * own_f).sum() / n_own),
                    n_moves=len(moves))
    cache = dict(nf=nf, nm=nm, am=am, edge=edge, tgt=tgt.detach(),
                 frac=frac.detach(), own=own, frac_active=frac_active,
                 old_logp_per=logp_per.detach(), diag=diag)
    return moves, logp, value[0], cache


def recompute_logp_value(net, cache, dev):
    tgt_logits, frac_ab, value = net(cache["nf"], cache["edge"], cache["nm"], cache["am"])
    tgt_logits = tgt_logits[0]; frac_ab = frac_ab[0]
    own = cache["own"]; frac_active = cache["frac_active"]
    td = torch.distributions.Categorical(logits=tgt_logits)
    fd = frac_dist(frac_ab)
    logp_t = td.log_prob(cache["tgt"])
    logp_f = fd.log_prob(cache["frac"])
    logp_per = logp_t * own.float() + logp_f * frac_active.float()
    active = own.float()
    # Per-owned-node MEAN entropy (not a sum): the policy loss is averaged over
    # owned nodes (sum / denom below), so the entropy bonus must be normalized the
    # same way. A raw sum makes the effective entropy weight scale with how many
    # planets the focal player owns (~ent*n_own), which silently turns a nominal
    # 0.05 into ~0.5+ and randomizes the warm-started policy. Dividing by the owned
    # count keeps --ent a true per-node coefficient, independent of board size.
    ent = ((td.entropy() * own.float()).sum() + (fd.entropy() * frac_active.float()).sum()) \
        / active.sum().clamp(min=1)
    return logp_per, active, value[0], ent


# --------------------------------------------------------------------------- #
# Batched collection + update (the GPU-efficient path used by main()).
#
# The per-state path above (policy_act / recompute_logp_value) issues ONE batch-1
# forward per state. Profiling (prof.py) put ~53% of collection wall-clock in that
# kernel-launch latency on a tiny net. These helpers batch the focal forward across
# all episodes_per_iter games (collection) and across a whole minibatch (update),
# turning B batch-1 calls into one B-sized call. The per-node math is identical to
# the per-state path — each board is padded to N_MAX and attention is masked
# per-sample (node_mask), so a batched forward is bit-equivalent to looping B
# single forwards (verified by test_batched.py). Caches here are plain NumPy (cheap
# to keep thousands of in RAM, only minibatches are moved to the GPU).
# --------------------------------------------------------------------------- #
def _stack(caches, key, dev, dtype):
    return torch.as_tensor(np.stack([c[key] for c in caches]), dtype=dtype, device=dev)


def _batched_policy(net, caches, dev, grad):
    """Re-eval stored (NumPy) caches in one forward. Returns per-node logp (B,N),
    owned mask (B,N), value (B,), per-sample mean entropy (B,), owned counts (B)."""
    nf = _stack(caches, "nf", dev, torch.float32)
    nm = _stack(caches, "nm", dev, torch.float32)
    am = _stack(caches, "am", dev, torch.float32)
    own = _stack(caches, "own", dev, torch.float32)
    fa = _stack(caches, "frac_active", dev, torch.float32)
    tgt = _stack(caches, "tgt", dev, torch.long)
    frac = _stack(caches, "frac", dev, torch.float32).clamp(FRAC_EPS, 1 - FRAC_EPS)
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx:
        edge = build_edge(nf)
        tgt_logits, frac_ab, value = net(nf, edge, nm, am)
        td = torch.distributions.Categorical(logits=tgt_logits)
        fd = frac_dist(frac_ab)
        logp_per = td.log_prob(tgt) * own + fd.log_prob(frac) * fa     # (B,N)
        denom = own.sum(1).clamp(min=1)                                # (B,)
        ent = ((td.entropy() * own).sum(1) + (fd.entropy() * fa).sum(1)) / denom
    return logp_per, own, value, ent, denom


def _batched_value(net, caches, dev):
    """Value-only batched forward (warmup); grad flows, caller zeroes non-value grads."""
    nf = _stack(caches, "nf", dev, torch.float32)
    nm = _stack(caches, "nm", dev, torch.float32)
    am = _stack(caches, "am", dev, torch.float32)
    _, _, value = net(nf, build_edge(nf), nm, am)
    return value


def collect_batched(net, opp_draw, envs, seeds, dev, gamma=0.99, lam=0.95):
    """Run len(envs) games in lockstep, batching the focal forward across all
    still-active games each step. Returns a list of (traj, adv, ret, dbg) — one per
    game, same shape collect_episode returns — and leaves each env in its terminal
    state (so the caller can read env._terminal_reward(0) for the win flag)."""
    B = len(envs)
    for env, seed in zip(envs, seeds):
        env.reset(seed=seed)
    P = envs[0].num_players
    slot_opps = [[opp_draw() for _ in range(P - 1)] for _ in range(B)]
    trajs = [[] for _ in range(B)]
    active = [True] * B
    while any(active):
        idxs = [b for b in range(B) if active[b]]
        obs0 = [envs[b].obs_for(0) for b in idxs]
        encs = [encode_state(o, 0) for o in obs0]
        nf = torch.as_tensor(np.stack([e["node_feats"] for e in encs]), device=dev)
        nm = torch.as_tensor(np.stack([e["node_mask"] for e in encs]), device=dev)
        am = torch.as_tensor(np.stack([e["attack_mask"] for e in encs]), device=dev)
        with torch.no_grad():
            tgt_logits, frac_ab, value = net(nf, build_edge(nf), nm, am)
        for k, b in enumerate(idxs):
            enc = encs[k]; obs = obs0[k]
            own = torch.as_tensor(enc["own_mask"], dtype=torch.bool, device=dev)
            td = torch.distributions.Categorical(logits=tgt_logits[k])
            fd = frac_dist(frac_ab[k])
            tgt = td.sample(); frac = fd.sample().clamp(FRAC_EPS, 1 - FRAC_EPS)
            tgt_np = tgt.detach().cpu().numpy(); frac_np = frac.detach().cpu().numpy()
            fa_np = _frac_active_mask(enc, tgt_np)
            fa = torch.as_tensor(fa_np, dtype=torch.float32, device=dev)
            ownf = own.float()
            logp_per = td.log_prob(tgt) * ownf + fd.log_prob(frac) * fa
            n_own = ownf.sum().clamp(min=1)
            moves0 = _dec(enc, obs, 0, tgt_np, frac_np)
            diag = dict(n_own=float(ownf.sum()),
                        hold_frac=float(((tgt == HOLD).float() * ownf).sum() / n_own),
                        ent_cat=float((td.entropy() * ownf).sum() / n_own),
                        conc=float((frac_ab[k].sum(-1) * ownf).sum() / n_own),
                        n_moves=len(moves0))
            moves = [moves0]
            for slot in range(1, P):
                moves.append(slot_opps[b][slot - 1].moves(envs[b].obs_for(slot), slot))
            r, done = envs[b].step(moves)
            cache = dict(nf=enc["node_feats"], nm=enc["node_mask"], am=enc["attack_mask"],
                         own=enc["own_mask"].astype(np.float32),
                         frac_active=fa_np.astype(np.float32),
                         tgt=tgt_np.astype(np.int64), frac=frac_np.astype(np.float32),
                         old_logp_per=logp_per.detach().cpu().numpy().astype(np.float32),
                         diag=diag)
            trajs[b].append([float(logp_per.sum()), float(value[k]), r, cache])
            if done:
                active[b] = False
    out = []
    for b in range(B):
        traj = trajs[b]
        adv, gae, nextv = [0.0] * len(traj), 0.0, 0.0
        for t in reversed(range(len(traj))):
            v = traj[t][1]
            delta = traj[t][2] + gamma * nextv - v
            gae = delta + gamma * lam * gae
            adv[t] = gae; nextv = v
        ret = [adv[t] + traj[t][1] for t in range(len(traj))]
        dg = [t[3]["diag"] for t in traj]
        dbg = dict(ep_len=len(traj), ret_sum=sum(t[2] for t in traj),
                   adv_min=min(adv) if adv else 0.0, adv_max=max(adv) if adv else 0.0,
                   val_min=min(t[1] for t in traj) if traj else 0.0,
                   val_max=max(t[1] for t in traj) if traj else 0.0,
                   ret_min=min(ret) if ret else 0.0, ret_max=max(ret) if ret else 0.0,
                   hold_frac=np.mean([d["hold_frac"] for d in dg]) if dg else 0.0,
                   ent_cat=np.mean([d["ent_cat"] for d in dg]) if dg else 0.0,
                   conc=np.mean([d["conc"] for d in dg]) if dg else 0.0,
                   n_moves=np.mean([d["n_moves"] for d in dg]) if dg else 0.0)
        out.append((traj, adv, ret, dbg))
    return out


def net_greedy_agent(net, dev):
    """A kaggle-style single-arg agent that plays `net` greedily (comet-safe)."""
    def agent(obs):
        try:
            enc = encode_state(obs, obs["player"])
            if enc["own_mask"].sum() == 0:
                return []
            nf, nm, am, edge = _fwd(net, enc, dev)
            with torch.no_grad():
                tl, fab, _ = net(nf, edge, nm, am)
            tgt = tl[0].argmax(-1).cpu().numpy()
            frac = frac_dist(fab[0]).mean.cpu().numpy()
            return _dec(enc, obs, obs["player"], tgt, frac)
        except Exception:
            return []
    return agent


# --------------------------------------------------------------------------- #
# Opponent providers (scripted fn, frozen net, or pool of frozen nets)
# --------------------------------------------------------------------------- #
class Opponent:
    """Produces moves for a non-focal slot. Backed by an ow_base function, a
    frozen OrbitNet, or a pool sampled per call."""
    def __init__(self, kind, fn=None, nets=None, dev="cpu"):
        self.kind = kind; self.fn = fn; self.nets = nets or []; self.dev = dev

    def moves(self, obs, player):
        if self.kind == "fn":
            try:
                o = dict(obs); o["player"] = player
                return self.fn(o) or []
            except Exception:
                return []
        net = self.nets[np.random.randint(len(self.nets))] if len(self.nets) > 1 else self.nets[0]
        return net_greedy_agent(net, self.dev)(dict(obs, player=player))


def _load_frozen(path, dev):
    n = OrbitNet().to(dev)
    n.load_state_dict(torch.load(path, map_location=dev)["model"])
    n.eval()
    for p in n.parameters():
        p.requires_grad_(False)
    return n


def _snapshot(net, dev):
    """Mint a fresh frozen OrbitNet copying `net`'s current weights. Used to add a
    detached snapshot to the rolling self-play league (no shared parameters)."""
    n = OrbitNet().to(dev)
    n.load_state_dict(net.state_dict())
    n.eval()
    for p in n.parameters():
        p.requires_grad_(False)
    return n


def _load_external_agent(path):
    """Load a .py file's `agent(obs)` callable. Used to wire a previously-built
    Kaggle submission (e.g. RL/submission_orbitnet.py) in as a sparring partner."""
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(
        f"_ext_{os.path.splitext(os.path.basename(path))[0]}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import external agent at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "agent", None)
    if not callable(fn):
        raise RuntimeError(f"{path} does not expose a callable `agent`")
    return fn


def build_opponent(spec, self_pool, dev):
    """Map an --opponent spec string to an Opponent. `self_pool` is the rolling
    list of frozen self-play snapshots (mutated in place by the trainer); for
    `--league-size 1` it stays length 1 and behaves like the old single copy."""
    if spec == "self":
        return Opponent("net", nets=self_pool, dev=dev), True   # refreshable
    if spec == "teacher":
        return Opponent("fn", fn=ow_base.net_roi_support, dev=dev), False
    if spec == "aggressive":
        return Opponent("fn", fn=ow_base.net_roi_aggressive, dev=dev), False
    if spec.startswith("ckpt:"):
        return Opponent("net", nets=[_load_frozen(spec[5:], dev)], dev=dev), False
    if spec.startswith("pool:"):
        paths = []
        for tok in spec[5:].split(","):
            paths.extend(glob.glob(tok))
        assert paths, f"pool spec matched no files: {spec}"
        return Opponent("net", nets=[_load_frozen(p, dev) for p in paths], dev=dev), False
    return Opponent("fn", fn=getattr(ow_base, spec), dev=dev), False


def make_opp_draw(mix_weights, self_pool, externals, dev, gauntlet_pool=None):
    """Build the per-episode opponent sampler from a mix-weight dict, the live
    self-play league, and a name->callable externals map. Module-level (not a
    closure) so collection workers can reconstruct the exact same draw from a
    league of frozen nets they rebuilt from state-dicts (see parallel.py).
    gauntlet_pool: promoted frozen snapshots drawn (uniformly) by the 'gauntlet'
    key — past strong selves that the agent keeps sparring against."""
    gauntlet_pool = gauntlet_pool or []
    kinds = list(mix_weights.keys())
    probs = np.array([mix_weights[k] for k in kinds], dtype=np.float64)
    probs = probs / probs.sum()

    def draw():
        k = kinds[np.random.choice(len(kinds), p=probs)]
        if k == "teacher":    return Opponent("fn", fn=ow_base.net_roi_support, dev=dev)
        if k == "aggressive": return Opponent("fn", fn=ow_base.net_roi_aggressive, dev=dev)
        if k == "weak":       return Opponent("fn", fn=ow_base.nearest_planet, dev=dev)
        if k.startswith("script:"):
            return Opponent("fn", fn=getattr(ow_base, k[7:]), dev=dev)
        if k.startswith("ext:"):
            return Opponent("fn", fn=externals[k[4:]], dev=dev)
        if k == "gauntlet":
            pick = gauntlet_pool[np.random.randint(len(gauntlet_pool))]
            return Opponent("net", nets=[pick], dev=dev)
        net_pick = self_pool[np.random.randint(len(self_pool))]
        return Opponent("net", nets=[net_pick], dev=dev)
    return draw


# --------------------------------------------------------------------------- #
# Benchmark (real comet-bearing env, eval seed pool, n-player aware)
# --------------------------------------------------------------------------- #
def benchmark_winrate(net, dev, eval_opp_fn, games, num_players, seed_offset=0):
    """GREEDY eval of `net` (focal) vs a fixed scripted opponent filling the
    other slots, over real-env games. Rotates the focal slot for fairness. Uses
    the EVAL seed pool (disjoint from training). Win = focal reward == +1
    (engine: highest positive ship total; ties win)."""
    from quiet_kaggle import make
    was_training = net.training
    net.eval()
    agent = net_greedy_agent(net, dev)
    wins = 0
    for i in range(games):
        slot = i % num_players
        order = [eval_opp_fn] * num_players
        order[slot] = agent
        e = make("orbit_wars", configuration={"seed": eval_seed(seed_offset + i)}, debug=False)
        with contextlib.redirect_stderr(io.StringIO()):
            e.run(order)
        r = [s.reward if s.reward is not None else -1 for s in e.steps[-1]]
        wins += 1 if r[slot] == max(r) and r[slot] > 0 else 0
    if was_training:
        net.train()
    return wins / max(games, 1)


# --------------------------------------------------------------------------- #
# Rollout (focal = slot 0; opponents fill the rest)
# --------------------------------------------------------------------------- #
def collect_episode(net, opp_draw, env, seed, dev, gamma=0.99, lam=0.95):
    """opp_draw() -> Opponent. Called once per non-focal slot at episode start,
    so each slot's identity is sampled independently but stays fixed within the
    episode (coherent gameplay per slot)."""
    env.reset(seed=seed)
    P = env.num_players
    slot_opps = [opp_draw() for _ in range(P - 1)]
    traj = []
    done = False
    while not done:
        mv0, logp, val, cache = policy_act(net, env.obs_for(0), 0, dev)
        moves = [mv0]
        for slot in range(1, P):
            moves.append(slot_opps[slot - 1].moves(env.obs_for(slot), slot))
        r, done = env.step(moves)
        traj.append([float(logp), float(val), r, cache])
    adv, gae, nextv = [0.0] * len(traj), 0.0, 0.0
    for t in reversed(range(len(traj))):
        v = traj[t][1]
        delta = traj[t][2] + gamma * nextv - v
        gae = delta + gamma * lam * gae
        adv[t] = gae; nextv = v
    ret = [adv[t] + traj[t][1] for t in range(len(traj))]
    vals = [traj[t][1] for t in range(len(traj))]
    dg = [t[3]["diag"] for t in traj]
    dbg = dict(ep_len=len(traj), ret_sum=sum(t[2] for t in traj),
               adv_min=min(adv), adv_max=max(adv),
               val_min=min(vals), val_max=max(vals), ret_min=min(ret), ret_max=max(ret),
               hold_frac=np.mean([d["hold_frac"] for d in dg]),
               ent_cat=np.mean([d["ent_cat"] for d in dg]),
               conc=np.mean([d["conc"] for d in dg]),
               n_moves=np.mean([d["n_moves"] for d in dg]))
    return traj, adv, ret, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=2, choices=[2, 4],
                    help="train the 2-player or the 4-player model (separate models)")
    ap.add_argument("--init", default="bc.pt")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--episodes_per_iter", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5, dest="vf_coef",
                    help="value-loss weight; lower (~0.25) if value grads through the "
                         "shared trunk destabilize a peaked warm-started policy")
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--target-kl", type=float, default=0.03, dest="target_kl")
    ap.add_argument("--max-grad-norm", type=float, default=0.5, dest="max_grad_norm")
    ap.add_argument("--train-seed-offset", type=int, default=0, dest="train_off",
                    help="offset into the TRAIN seed pool (cannot reach eval seeds)")
    ap.add_argument("--refresh", type=int, default=20, help="iters between self-play opponent refresh")
    ap.add_argument("--opponent", default="self",
                    help="self | teacher | aggressive | <ow_base name> | ckpt:PATH | pool:GLOB[,GLOB]")
    ap.add_argument("--league-size", type=int, default=1, dest="league_size",
                    help="max in-memory self-snapshots (1 = back-compat: one frozen copy)")
    ap.add_argument("--league-add-every", type=int, default=0, dest="league_add_every",
                    help="iters between snapshot append to the league (0 = use --refresh)")
    ap.add_argument("--mix-teacher", type=float, default=0.0, dest="mix_teacher",
                    help="per-slot probability of using scripted teacher (net_roi_support)")
    ap.add_argument("--mix-aggressive", type=float, default=0.0, dest="mix_aggressive",
                    help="per-slot probability of using scripted aggressive teacher")
    ap.add_argument("--mix-weak", type=float, default=0.0, dest="mix_weak",
                    help="per-slot probability of using a weak scripted agent (nearest_planet)")
    ap.add_argument("--mix-scripted", action="append", default=[], dest="mix_scripted",
                    help="NAME:WEIGHT (repeatable); per-slot weight for any callable in ow_base.py")
    ap.add_argument("--external", action="append", default=[], dest="external",
                    help="NAME=PATH (repeatable); PATH is a .py file exposing agent(obs)")
    ap.add_argument("--mix-external", action="append", default=[], dest="mix_external",
                    help="NAME:WEIGHT (repeatable); per-slot weight for a registered --external")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--coord-cap", type=float, default=2.0, dest="coord_cap",
                    help="decode coordination cap (set <0 to disable -> no overkill trim)")
    ap.add_argument("--hard-skip", action="store_true", dest="hard_skip",
                    help="restore v1 'solo-capture-or-skip' (forbids coordinated attacks)")
    ap.add_argument("--out", default="ppo.pt")
    ap.add_argument("--value-warmup", type=int, default=100, dest="value_warmup")
    ap.add_argument("--warmup-lr", type=float, default=1e-3, dest="warmup_lr",
                    help="LR for the value-head warmup ONLY (it zeroes non-value grads, so "
                         "this can't destabilize the policy). Decoupled from --lr so a low "
                         "policy lr doesn't leave the value head under-calibrated.")
    ap.add_argument("--warmup-games", type=int, default=6, dest="warmup_games")
    ap.add_argument("--warmup-cache", default=None, dest="warmup_cache",
                    help="path to cache the post-warmup net. If the file exists it is "
                         "loaded and warmup is SKIPPED; otherwise warmup runs and is saved "
                         "there. Valid only for the same --init and opponent mix — lr/vf-coef "
                         "do NOT affect warmup, so one cache is reusable across hp sweeps.")
    ap.add_argument("--eval-every", type=int, default=10, dest="eval_every")
    ap.add_argument("--eval-games", type=int, default=40, dest="eval_games")
    ap.add_argument("--eval-opponent", default="teacher", dest="eval_opponent")
    ap.add_argument("--bench-also", action="append", default=[], dest="bench_also",
                    help="extra benchmark opponent (same syntax as --eval-opponent); "
                         "repeatable; logged but does not drive _best")
    ap.add_argument("--eval-seed-offset", type=int, default=0, dest="eval_off")
    ap.add_argument("--early-stop-patience", type=int, default=0, dest="early_stop_patience",
                    help="stop when the PRIMARY benchmark (--eval-opponent) hasn't set a new best "
                         "for this many EVAL TICKS (0 = never). A tick is --eval-every iters apart, "
                         "so the no-improvement window is patience*eval-every iters. Used to "
                         "auto-end phase-1 once it has plateaued — it only needs to be 'good "
                         "enough' to seed phase-2.")
    # --- self-improving gauntlet ladder ----------------------------------- #
    # When the agent beats the PRIMARY benchmark at/above --promote-threshold, it
    # snapshots itself to <out>_gauntlet_<k>.pt and (a) adds that snapshot as a
    # fixed benchmark it must keep beating (own _best_gauntlet_<k>.pt) and (b) folds
    # it into the training mix (--mix-gauntlet weight, taken from the league share)
    # so it keeps sparring past strong selves. This is what lets a long run keep
    # finding harder targets instead of saturating once the scripted teacher falls.
    ap.add_argument("--promote-threshold", type=float, default=0.0, dest="promote_threshold",
                    help="primary-benchmark win-rate that triggers a gauntlet promotion "
                         "(0 disables the ladder; e.g. 0.92)")
    ap.add_argument("--promote-every", type=int, default=3, dest="promote_every",
                    help="min EVAL TICKS between promotions (rate-limit so rungs aren't near-dupes)")
    ap.add_argument("--mix-gauntlet", type=float, default=0.0, dest="mix_gauntlet",
                    help="training-mix weight spread (uniformly) over promoted rungs; "
                         "drawn from the league share, applied once any rung exists")
    ap.add_argument("--gauntlet-max", type=int, default=8, dest="gauntlet_max",
                    help="cap on rungs kept in the training mix AND benchmarked (most-recent "
                         "kept; older .pt files persist on disk). 0 = unlimited")
    # --- collapse guard (auto-stop on entropy/passivity runaway) ----------- #
    ap.add_argument("--collapse-entH", type=float, default=0.0, dest="collapse_entH",
                    help="stop if mean target entropy stays >= this for --collapse-patience iters "
                         "(0 disables; healthy ~0.3-0.5, collapse >1.0)")
    ap.add_argument("--collapse-hold", type=float, default=-1.0, dest="collapse_hold",
                    help="stop if mean hold-fraction stays <= this for --collapse-patience iters "
                         "(<0 disables; healthy ~0.8, collapse <0.4)")
    ap.add_argument("--collapse-patience", type=int, default=0, dest="collapse_patience",
                    help="consecutive iters past a collapse threshold before stopping (0 disables)")
    ap.add_argument("--save-every", type=int, default=25, dest="save_every")
    ap.add_argument("--log-every", type=int, default=1, dest="log_every")
    ap.add_argument("--workers", type=int, default=1,
                    help="CPU worker processes for collection AND eval (both are game-sim "
                         "bound after the forward is batched). 0 = all cores, 1 = in-process "
                         "(deterministic). Big speedup for the CPU-bound rollouts/eval.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    coord_cap = None if args.coord_cap < 0 else args.coord_cap
    _DECODE["coord_cap"] = coord_cap
    _DECODE["hard_skip"] = args.hard_skip

    net = OrbitNet().to(dev)
    if args.init:
        ck = torch.load(args.init, map_location=dev)
        net.load_state_dict(ck["model"]); print("warm-started from", args.init)
        if ck.get("players") not in (None, args.players):
            print(f"  WARNING: init ckpt was trained for {ck.get('players')}p, now training {args.players}p")
    self_pool = [_snapshot(net, dev)]
    externals = {}
    ext_paths = {}        # name -> abspath, for workers to (re)load the external agent
    for spec in args.external:
        if "=" not in spec:
            raise SystemExit(f"--external must be NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        externals[name] = _load_external_agent(path)
        ext_paths[name] = os.path.abspath(path)
        print(f"loaded external agent '{name}' from {path}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # CPU worker pool for collection + eval (both game-sim bound). 1 = in-process.
    nworkers = parallel.init_pool(args.workers)
    use_mp = nworkers > 1
    if use_mp:
        print(f"[parallel] {nworkers} CPU workers for collection + eval")

    # A persistent pool of envs stepped in lockstep by collect_batched (one per
    # game in a batch). Reused across iters; reset() rebuilds each game's state.
    train_envs = [OrbitEnv(max_steps=args.max_steps, num_players=args.players, focal=0)
                  for _ in range(args.episodes_per_iter)]
    opp, refreshable = build_opponent(args.opponent, self_pool, dev)
    def _resolve_eval_opp(name):
        if name.startswith("external:"):
            return externals[name[len("external:"):]]
        if name.startswith("ckpt:"):
            return net_greedy_agent(_load_frozen(name[len("ckpt:"):], dev), dev)
        if name == "teacher":     return ow_base.net_roi_support
        if name == "aggressive":  return ow_base.net_roi_aggressive
        return getattr(ow_base, name)

    eval_opp_fn = _resolve_eval_opp(args.eval_opponent)
    bench_also = [(spec, _resolve_eval_opp(spec)) for spec in args.bench_also]

    mix_weights = {"teacher": args.mix_teacher,
                   "aggressive": args.mix_aggressive,
                   "weak": args.mix_weak}
    for spec in args.mix_scripted:
        if ":" not in spec:
            raise SystemExit(f"--mix-scripted must be NAME:WEIGHT, got: {spec}")
        name, w = spec.rsplit(":", 1)
        if not callable(getattr(ow_base, name, None)):
            raise SystemExit(f"--mix-scripted '{name}' is not a callable in ow_base.py")
        mix_weights[f"script:{name}"] = float(w)
    for spec in args.mix_external:
        if ":" not in spec:
            raise SystemExit(f"--mix-external must be NAME:WEIGHT, got: {spec}")
        name, w = spec.rsplit(":", 1)
        if name not in externals:
            raise SystemExit(f"--mix-external '{name}' has no matching --external NAME=PATH")
        mix_weights[f"ext:{name}"] = float(w)
    explicit_sum = sum(mix_weights.values())
    if explicit_sum > 1.0 + 1e-9:
        raise SystemExit(f"--mix-* weights sum to {explicit_sum:.3f} > 1.0")
    mix_weights["league"] = max(0.0, 1.0 - explicit_sum)
    # Promote to per-episode draw path whenever the league has >1 snapshot so the
    # opponent identity stays coherent within a game (back-compat single-opp would
    # resample every step).
    mix_active = explicit_sum > 0.0 or args.mix_gauntlet > 0.0 \
        or (args.league_size > 1 and refreshable)

    # Promoted-rung state for the gauntlet ladder (populated at eval ticks).
    gauntlet_pool = []        # frozen snapshots fed to the training mix (most-recent kept)
    gauntlet_rungs = []       # [{"name","path"}] for every promotion (benchmarks use recent)
    n_promoted = 0

    def _effective_mix():
        """Base mix, plus a 'gauntlet' share carved out of the league once any rung
        exists (so the reserved weight isn't wasted while the pool is empty)."""
        if args.mix_gauntlet > 0 and gauntlet_pool:
            m = dict(mix_weights)
            g = min(args.mix_gauntlet, m["league"])
            m["league"] = m["league"] - g
            m["gauntlet"] = g
            return m
        return mix_weights

    def _collect(seeds):
        """Collect len(seeds) games -> (results, win_flags) in seed order. Parallel
        across CPU workers when --workers>1, else in-process on the GPU via
        collect_batched. Both yield identical result/cache structure."""
        eff = _effective_mix()
        if use_mp:
            return parallel.collect(net, self_pool, eff, ext_paths, seeds,
                                    args.players, args.max_steps, gauntlet_pool=gauntlet_pool)
        draw = (make_opp_draw(eff, self_pool, externals, dev, gauntlet_pool)
                if mix_active else (lambda: opp))
        envs = train_envs[:len(seeds)]
        results = collect_batched(net, draw, envs, seeds, dev)
        wins = [1 if envs[j]._terminal_reward(0) > 0 else 0 for j in range(len(envs))]
        return results, wins

    print(f"[{args.players}p] opponent={args.opponent} | lr={args.lr} ent={args.ent} clip={args.clip} vf={args.vf_coef}"
          f" | eps/iter={args.episodes_per_iter} mb={args.minibatch} kl={args.target_kl}"
          f" | coord_cap={coord_cap} hard_skip={args.hard_skip} max_steps={args.max_steps}")
    if mix_active:
        nz = {k: round(v, 4) for k, v in mix_weights.items() if v > 0}
        print(f"  mix (per non-focal slot, per episode): {nz}"
              f" | league_size={args.league_size}"
              f" add_every={args.league_add_every or args.refresh}")
    else:
        print(f"  league_size={args.league_size} add_every={args.league_add_every or args.refresh}"
              f" (back-compat single-opponent path)")
    print(f"  selection: GREEDY benchmark vs {args.eval_opponent} every {args.eval_every} iters"
          f" ({args.eval_games} eval-pool games) -> _best (submit target)")
    if args.bench_also:
        print(f"  also benchmarking vs: {', '.join(args.bench_also)} "
              f"({args.eval_games} games each per eval tick — each keeps its own _best_<name>)")
    if args.promote_threshold > 0:
        print(f"  gauntlet: promote a snapshot when {args.eval_opponent} bench >= "
              f"{args.promote_threshold:.2f} (>= {args.promote_every} ticks apart); "
              f"mix_gauntlet={args.mix_gauntlet} keep={args.gauntlet_max or 'all'} "
              f"-> _gauntlet_<k>.pt + _best_gauntlet_<k>.pt")
    if args.collapse_patience > 0:
        print(f"  collapse-guard: stop after {args.collapse_patience} iters with "
              f"entH>={args.collapse_entH or '-'} or hold<={args.collapse_hold if args.collapse_hold>=0 else '-'}")

    def _suffix(tag):
        return (args.out.replace(".pt", f"_{tag}.pt") if args.out.endswith(".pt")
                else args.out + f"_{tag}")

    # Every benchmarked opponent keeps its OWN rolling best checkpoint: the primary
    # (--eval-opponent) saves to the canonical _best (the submit target); each
    # --bench-also opponent saves to _best_<name>. Each file is updated
    # independently per eval tick, so beating several opponents at once saves
    # several bests in the same tick.
    # Each target is (display_name, eval_spec, eval_fn, is_primary): eval_spec is
    # the string handed to the parallel worker (incl. "ckpt:<path>" for gauntlet
    # rungs); eval_fn is the in-process callable; display_name keys best_wr / files.
    def _rebuild_bench_targets():
        bt = [(args.eval_opponent, args.eval_opponent, eval_opp_fn, True)]
        bt += [(spec, spec, fn, False) for spec, fn in bench_also]
        recent = gauntlet_rungs[-args.gauntlet_max:] if args.gauntlet_max > 0 else gauntlet_rungs
        for r in recent:
            cspec = "ckpt:" + r["path"]
            bt.append((r["name"], cspec, _resolve_eval_opp(cspec), False))
        return bt

    bench_targets = _rebuild_bench_targets()
    best_wr = {}        # opponent display name -> best benchmark win-rate so far

    def _best_path(name, primary):
        if primary:
            return _suffix("best")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
        return _suffix(f"best_{safe}")

    def _eval_and_save_bests():
        """Benchmark net vs every target; save that opponent's own _best whenever
        it improves. Returns (primary_wr, [(name, wr) extras], [saved filenames])."""
        primary_wr, extras, saved = None, [], []
        for disp, spec, fn, primary in bench_targets:
            if use_mp:
                wr = parallel.eval_winrate(net, spec, ext_paths, args.eval_games,
                                           args.players, args.eval_off)
            else:
                wr = benchmark_winrate(net, dev, fn, args.eval_games, args.players, args.eval_off)
            if wr > best_wr.get(disp, -1.0):
                best_wr[disp] = wr
                path = _best_path(disp, primary)
                torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                            "players": args.players}, path)
                saved.append(os.path.basename(path))
            (extras.append((disp, wr)) if not primary else None)
            if primary:
                primary_wr = wr
        return primary_wr, extras, saved

    seed_i = args.train_off
    no_improve_ticks = 0          # eval ticks since the primary bench last set a new best
    eval_tick = 0                 # count of eval ticks (gates gauntlet promotion cadence)
    last_promote_tick = -10**9    # eval_tick of the most recent promotion
    collapse_run = 0              # consecutive iters past a collapse threshold

    # ===== one-shot value warmup (optionally cached to disk for reuse) =====
    # The warmup-calibrated value head depends ONLY on the init policy and the
    # opponent mix (NOT on lr/vf-coef/clip), so one cache is valid across an hp
    # sweep. We store init+mix to warn on a mismatch.
    cache_path = args.warmup_cache
    if cache_path and os.path.exists(cache_path):
        ck = torch.load(cache_path, map_location=dev)
        net.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:    # opt state in the cache carries the warmup lr; the
            g["lr"] = args.lr         # training phase must use THIS run's --lr (sweep-safe)
        seed_i += args.warmup_games          # keep training seeds aligned with a fresh warmup
        if ck.get("init") not in (None, args.init):
            print(f"  WARNING: warmup cache built from init={ck.get('init')}, now init={args.init}")
        if ck.get("mix") is not None and ck["mix"] != mix_weights:
            print(f"  WARNING: warmup cache built under a DIFFERENT opponent mix — value "
                  f"calibration may be stale (cached {ck['mix']} vs now {mix_weights})")
        print(f"[warmup] reused cache {cache_path} — skipped {args.warmup_games}-game collect "
              f"+ {args.value_warmup}-epoch value train")
    elif args.value_warmup > 0:
        print(f"[warmup] collecting {args.warmup_games} games for value calibration...")
        wbatch, wwins = [], 0
        collected = 0
        while collected < args.warmup_games:
            b = min(args.episodes_per_iter, args.warmup_games - collected)
            seeds_w = [train_seed(seed_i + j) for j in range(b)]
            results, wins_w = _collect(seeds_w)
            seed_i += b; collected += b
            for j, (traj, adv, ret, _dbg) in enumerate(results):
                wwins += wins_w[j]
                for t in range(len(traj)):
                    wbatch.append([traj[t][3], ret[t]])
        print(f"[warmup] {len(wbatch)} states (collect win_rate {wwins/args.warmup_games:.2f}); "
              f"training value head {args.value_warmup} epochs @ lr {args.warmup_lr:g}...")
        for g in opt.param_groups:    # value-only phase: use the dedicated warmup lr
            g["lr"] = args.warmup_lr
        for ep in range(args.value_warmup):
            np.random.shuffle(wbatch); vloss_sum = 0.0
            for s in range(0, len(wbatch), args.minibatch):
                chunk = wbatch[s:s + args.minibatch]; opt.zero_grad()
                val = _batched_value(net, [c for c, _ in chunk], dev)
                rett = torch.as_tensor([r for _, r in chunk], device=dev, dtype=torch.float32)
                vloss = F.mse_loss(val, rett)
                vloss.backward(); vloss_sum += float(vloss.detach()) * len(chunk)
                for name, p in net.named_parameters():
                    if not name.startswith("value") and p.grad is not None:
                        p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if ep % max(1, args.value_warmup // 10) == 0 or ep == args.value_warmup - 1:
                print(f"[warmup] epoch {ep:3d}  value_loss {vloss_sum/len(wbatch):.3f}")
        for g in opt.param_groups:    # restore the policy lr before the cache save / training
            g["lr"] = args.lr
        print("[warmup] done.")
        if cache_path:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "players": args.players, "init": args.init, "mix": mix_weights},
                       cache_path)
            print(f"[warmup] cached to {cache_path} (reuse later with --warmup-cache {cache_path})")

    # ===== baseline benchmark (where the warm-start stands BEFORE any PPO step) =====
    # Establishes the starting win-rate and seeds EVERY opponent's _best with the
    # warm-start, so a diverging run can't overwrite any _best with something worse
    # than where we began.
    if args.eval_every > 0 and args.eval_games > 0:
        p_wr, extras, saved = _eval_and_save_bests()
        es = (" [" + "  ".join(f"{n} {w:.2f}" for n, w in extras) + "]") if extras else ""
        sv = ("  -> saved " + ", ".join(saved)) if saved else ""
        print(f"[baseline] bench vs {args.eval_opponent} {p_wr:.2f}{es}{sv}")

    for it in range(args.iters):
        batch, ep_rewards, wins, dbgs = [], [], 0, []
        iter_ret, iter_val = [], []      # for value explained variance (PPO health)
        # Collect all episodes_per_iter games (parallel CPU workers or in-process GPU).
        seeds = [train_seed(seed_i + j) for j in range(args.episodes_per_iter)]
        seed_i += args.episodes_per_iter
        results, wins_list = _collect(seeds)
        for j, (traj, adv, ret, dbg) in enumerate(results):
            ep_rewards.append(sum(t[2] for t in traj))
            wins += wins_list[j]
            dbgs.append(dbg)
            for t in range(len(traj)):
                batch.append([traj[t][3], adv[t], ret[t]])     # cache holds old_logp_per
                iter_ret.append(ret[t]); iter_val.append(traj[t][1])
        # Explained variance of the value head on THIS iter's collected states. This
        # is the #1 PPO health metric: <=0 means V explains none of the return
        # variance -> GAE advantages are noise -> the policy can't learn. It should
        # climb off ~0 within the first few dozen iters once the value head warms in;
        # if it stays pinned at <=0 the reward/return is the problem, not the hp.
        _r = np.array(iter_ret); _v = np.array(iter_val)
        ev = float(1.0 - np.var(_r - _v) / (np.var(_r) + 1e-9))
        # Collapse diagnostics (computed every iter so the guard can watch them):
        # hold% falling -> passive play; entH (target entropy) rising -> the policy
        # is dispersing; conc (Beta alpha+beta); mv = decoded moves/turn.
        hold = np.mean([d["hold_frac"] for d in dbgs])
        entc = np.mean([d["ent_cat"] for d in dbgs])
        conc = np.mean([d["conc"] for d in dbgs])
        mv = np.mean([d["n_moves"] for d in dbgs])
        all_adv = np.array([b[1] for b in batch], dtype=np.float64)
        amean, astd = all_adv.mean(), all_adv.std() + 1e-6
        for b in batch:
            b[1] = (b[1] - amean) / astd

        stopped_epoch = args.epochs
        mean_kl = 0.0
        kl_hist = []          # one entry per applied minibatch (drift from collection policy)
        stop = False
        for ep in range(args.epochs):
            np.random.shuffle(batch)
            for s in range(0, len(batch), args.minibatch):
                chunk = batch[s:s + args.minibatch]
                caches = [c[0] for c in chunk]
                # ONE batched forward for the whole minibatch (was batch-1 per state)
                logp_per, own, val, ent, denom = _batched_policy(net, caches, dev, grad=True)
                old_logp = _stack(caches, "old_logp_per", dev, torch.float32)         # (B,N)
                a_t = torch.as_tensor([c[1] for c in chunk], device=dev,
                                      dtype=torch.float32).unsqueeze(1)               # (B,1)
                rett = torch.as_tensor([c[2] for c in chunk], device=dev, dtype=torch.float32)
                ratio = torch.exp(logp_per - old_logp)                               # (B,N)
                s1 = ratio * a_t
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_t
                pol = ((-torch.min(s1, s2) * own).sum(1) / denom).mean()
                vloss = F.mse_loss(val, rett)
                loss = pol + args.vf_coef * vloss - args.ent * ent.mean()
                with torch.no_grad():
                    mb_kl = float((((old_logp - logp_per) * own).sum(1) / denom).mean())
                # PER-MINIBATCH KL GATE. mb_kl is total drift from the collection
                # policy measured at this minibatch's weights (i.e. BEFORE this step).
                # If we're already past target, stop without applying another step —
                # this bounds drift to ~target_kl instead of letting a whole epoch
                # (~batch/minibatch steps) overshoot before the old per-epoch check.
                if args.target_kl > 0 and mb_kl > args.target_kl:
                    stop = True; break
                kl_hist.append(mb_kl)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()
            if stop:
                stopped_epoch = ep + 1; break
        mean_kl = float(np.mean(kl_hist)) if kl_hist else 0.0

        if refreshable or mix_active:
            add_every = args.league_add_every or args.refresh
            if (it + 1) % add_every == 0:
                self_pool.append(_snapshot(net, dev))
                while len(self_pool) > args.league_size:
                    self_pool.pop(0)

        if args.save_every > 0 and (it + 1) % args.save_every == 0:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "players": args.players}, _suffix(f"it{it+1}"))

        bench = None
        bench_extras = []        # list of (display_name, win_rate)
        saved_bests = []         # filenames written this tick (one per opponent improved)
        early_stop_now = False
        if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            prev_primary_best = best_wr.get(args.eval_opponent, -1.0)
            bench, bench_extras, saved_bests = _eval_and_save_bests()
            if best_wr.get(args.eval_opponent, -1.0) > prev_primary_best:
                no_improve_ticks = 0
            else:
                no_improve_ticks += 1
            if args.early_stop_patience > 0 and no_improve_ticks >= args.early_stop_patience:
                early_stop_now = True

            # ---- gauntlet promotion: snapshot a new rung once we clear the bar ----
            eval_tick += 1
            if (args.promote_threshold > 0 and bench is not None
                    and bench >= args.promote_threshold
                    and (eval_tick - last_promote_tick) >= args.promote_every):
                n_promoted += 1
                gp = _suffix(f"gauntlet_{n_promoted}")
                torch.save({"model": net.state_dict(), "players": args.players}, gp)
                gauntlet_pool.append(_snapshot(net, dev))
                if args.gauntlet_max > 0:
                    while len(gauntlet_pool) > args.gauntlet_max:
                        gauntlet_pool.pop(0)        # keep most-recent in the training mix
                gauntlet_rungs.append({"name": f"gauntlet_{n_promoted}", "path": gp})
                bench_targets = _rebuild_bench_targets()   # benchmark the new rung from now on
                last_promote_tick = eval_tick
                print(f"[promote] {args.eval_opponent} bench {bench:.2f} >= "
                      f"{args.promote_threshold:.2f} -> rung {n_promoted} saved "
                      f"{os.path.basename(gp)}; {len(gauntlet_pool)} rung(s) in mix")

        if it % args.log_every == 0:
            wr = wins / args.episodes_per_iter
            bs = ""
            if bench is not None:
                bs = f"  bench {bench:.2f}"
                if bench_extras:
                    bs += " [" + "  ".join(f"{n} {w:.2f}" for n, w in bench_extras) + "]"
            sel_wr = best_wr.get(args.eval_opponent, -1.0)
            sel = f"best_bench {sel_wr:.2f}" if sel_wr >= 0 else "best_bench --"
            line = (f"iter {it:4d}  mean_ep_reward {np.mean(ep_rewards):+.3f}  train_wr {wr:.2f}"
                    f"{bs}  {sel}  kl {mean_kl:.4f}  ev {ev:+.2f}"
                    f"{'' if stopped_epoch == args.epochs else f' (early-stop @ep{stopped_epoch})'}"
                    f"  | hold {hold:.0%} entH {entc:.2f} conc {conc:.1f} mv {mv:.1f}"
                    f"{('  *saved ' + ', '.join(saved_bests)) if saved_bests else ''}")
            if args.debug:
                line += (f"\n        raw_adv [{min(d['adv_min'] for d in dbgs):+.1f},"
                         f"{max(d['adv_max'] for d in dbgs):+.1f}] val "
                         f"[{min(d['val_min'] for d in dbgs):+.1f},{max(d['val_max'] for d in dbgs):+.1f}]"
                         f" ep_len {np.mean([d['ep_len'] for d in dbgs]):.0f}")
            print(line)

        if early_stop_now:
            print(f"[early-stop] '{args.eval_opponent}' bench set no new best for "
                  f"{no_improve_ticks} eval ticks (~{no_improve_ticks * args.eval_every} iters); "
                  f"best {best_wr.get(args.eval_opponent, -1.0):.2f}; stopping at iter {it} "
                  f"-> phase-1 done, seeding phase-2")
            break

        # ---- collapse guard: abort a runaway entropy/passivity spiral early ----
        if args.collapse_patience > 0:
            bad = ((args.collapse_entH > 0 and entc >= args.collapse_entH)
                   or (args.collapse_hold >= 0 and hold <= args.collapse_hold))
            collapse_run = collapse_run + 1 if bad else 0
            if collapse_run >= args.collapse_patience:
                print(f"[collapse-guard] entH {entc:.2f} / hold {hold:.0%} past thresholds for "
                      f"{collapse_run} iters -> stopping at iter {it} "
                      f"(best checkpoints preserved: {os.path.basename(_suffix('best'))} etc.)")
                break

    torch.save({"model": net.state_dict(), "players": args.players}, args.out)
    print("saved", args.out)
    parallel.close_pool()


if __name__ == "__main__":
    main()
