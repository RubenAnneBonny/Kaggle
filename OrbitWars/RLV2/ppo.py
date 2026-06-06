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
    ent = (td.entropy() * own.float()).sum() + (fd.entropy() * frac_active.float()).sum()
    return logp_per, active, value[0], ent


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
    ap.add_argument("--save-every", type=int, default=25, dest="save_every")
    ap.add_argument("--log-every", type=int, default=1, dest="log_every")
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
    for spec in args.external:
        if "=" not in spec:
            raise SystemExit(f"--external must be NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        externals[name] = _load_external_agent(path)
        print(f"loaded external agent '{name}' from {path}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = OrbitEnv(max_steps=args.max_steps, num_players=args.players, focal=0)
    opp, refreshable = build_opponent(args.opponent, self_pool, dev)
    def _resolve_eval_opp(name):
        if name.startswith("external:"):
            return externals[name[len("external:"):]]
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
    mix_active = explicit_sum > 0.0 or (args.league_size > 1 and refreshable)

    def _make_draw():
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
            net_pick = self_pool[np.random.randint(len(self_pool))]
            return Opponent("net", nets=[net_pick], dev=dev)
        return draw

    opp_draw = _make_draw() if mix_active else (lambda: opp)

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

    def _suffix(tag):
        return (args.out.replace(".pt", f"_{tag}.pt") if args.out.endswith(".pt")
                else args.out + f"_{tag}")

    # Every benchmarked opponent keeps its OWN rolling best checkpoint: the primary
    # (--eval-opponent) saves to the canonical _best (the submit target); each
    # --bench-also opponent saves to _best_<name>. Each file is updated
    # independently per eval tick, so beating several opponents at once saves
    # several bests in the same tick.
    bench_targets = [(args.eval_opponent, eval_opp_fn, True)]
    bench_targets += [(spec, fn, False) for spec, fn in bench_also]
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
        for name, fn, primary in bench_targets:
            wr = benchmark_winrate(net, dev, fn, args.eval_games, args.players, args.eval_off)
            if wr > best_wr.get(name, -1.0):
                best_wr[name] = wr
                path = _best_path(name, primary)
                torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                            "players": args.players}, path)
                saved.append(os.path.basename(path))
            (extras.append((name, wr)) if not primary else None)
            if primary:
                primary_wr = wr
        return primary_wr, extras, saved

    seed_i = args.train_off
    no_improve_ticks = 0          # eval ticks since the primary bench last set a new best

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
        for _ in range(args.warmup_games):
            traj, adv, ret, _ = collect_episode(net, opp_draw, env, train_seed(seed_i), dev)
            seed_i += 1
            wwins += 1 if env._terminal_reward(0) > 0 else 0
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
                for cache, rett in chunk:
                    _, _, val, _ = recompute_logp_value(net, cache, dev)
                    vloss = F.mse_loss(val, torch.tensor(rett, device=dev, dtype=torch.float32))
                    (vloss / len(chunk)).backward(); vloss_sum += vloss.detach().item()
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
        for _ in range(args.episodes_per_iter):
            traj, adv, ret, dbg = collect_episode(net, opp_draw, env, train_seed(seed_i), dev)
            seed_i += 1
            ep_rewards.append(sum(t[2] for t in traj))
            wins += 1 if env._terminal_reward(0) > 0 else 0
            dbgs.append(dbg)
            for t in range(len(traj)):
                batch.append([traj[t][3], traj[t][3]["old_logp_per"], adv[t], ret[t]])
        all_adv = np.array([b[2] for b in batch], dtype=np.float64)
        amean, astd = all_adv.mean(), all_adv.std() + 1e-6
        for b in batch:
            b[2] = (b[2] - amean) / astd

        stopped_epoch = args.epochs
        mean_kl = 0.0
        kl_hist = []          # one entry per applied minibatch (drift from collection policy)
        stop = False
        for ep in range(args.epochs):
            np.random.shuffle(batch)
            for s in range(0, len(batch), args.minibatch):
                chunk = batch[s:s + args.minibatch]; opt.zero_grad()
                mb_kl_sum, mb_n = 0.0, 0
                for cache, old_logp_per, advt, rett in chunk:
                    logp_per, active, val, ent = recompute_logp_value(net, cache, dev)
                    a_t = torch.tensor(advt, device=dev, dtype=torch.float32)
                    ratio = torch.exp(logp_per - old_logp_per)
                    s1 = ratio * a_t
                    s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_t
                    denom = active.sum().clamp(min=1)
                    pol = (-torch.min(s1, s2) * active).sum() / denom
                    vloss = F.mse_loss(val, torch.tensor(rett, device=dev, dtype=torch.float32))
                    ((pol + args.vf_coef * vloss - args.ent * ent) / len(chunk)).backward()
                    with torch.no_grad():
                        mb_kl_sum += float(((old_logp_per - logp_per) * active).sum() / denom); mb_n += 1
                mb_kl = mb_kl_sum / max(mb_n, 1)
                # PER-MINIBATCH KL GATE. mb_kl is total drift from the collection
                # policy measured at this minibatch's weights (i.e. BEFORE this step).
                # If we're already past target, stop without applying another step —
                # this bounds drift to ~target_kl instead of letting a whole epoch
                # (~batch/minibatch steps) overshoot before the old per-epoch check.
                if args.target_kl > 0 and mb_kl > args.target_kl:
                    stop = True; break
                kl_hist.append(mb_kl)
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

        if it % args.log_every == 0:
            wr = wins / args.episodes_per_iter
            bs = ""
            if bench is not None:
                bs = f"  bench {bench:.2f}"
                if bench_extras:
                    bs += " [" + "  ".join(f"{n} {w:.2f}" for n, w in bench_extras) + "]"
            sel_wr = best_wr.get(args.eval_opponent, -1.0)
            sel = f"best_bench {sel_wr:.2f}" if sel_wr >= 0 else "best_bench --"
            # collapse diagnostics: hold% rising -> passivity; ent_cat -> 0 means the
            # target argmax sharpened (mode collapse); conc (Beta alpha+beta) rising
            # means the fraction head is going deterministic; mv = decoded moves/turn.
            hold = np.mean([d["hold_frac"] for d in dbgs])
            entc = np.mean([d["ent_cat"] for d in dbgs])
            conc = np.mean([d["conc"] for d in dbgs])
            mv = np.mean([d["n_moves"] for d in dbgs])
            line = (f"iter {it:4d}  mean_ep_reward {np.mean(ep_rewards):+.3f}  train_wr {wr:.2f}"
                    f"{bs}  {sel}  kl {mean_kl:.4f}"
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

    torch.save({"model": net.state_dict(), "players": args.players}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
