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
import argparse, contextlib, importlib.util, io, glob, os, numpy as np, torch, torch.nn.functional as F
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
    cache = dict(nf=nf, nm=nm, am=am, edge=edge, tgt=tgt.detach(),
                 frac=frac.detach(), own=own, frac_active=frac_active,
                 old_logp_per=logp_per.detach())
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
    from kaggle_environments import make
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
    dbg = dict(ep_len=len(traj), ret_sum=sum(t[2] for t in traj),
               adv_min=min(adv), adv_max=max(adv),
               val_min=min(vals), val_max=max(vals), ret_min=min(ret), ret_max=max(ret))
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
    ap.add_argument("--warmup-games", type=int, default=6, dest="warmup_games")
    ap.add_argument("--eval-every", type=int, default=10, dest="eval_every")
    ap.add_argument("--eval-games", type=int, default=40, dest="eval_games")
    ap.add_argument("--eval-opponent", default="teacher", dest="eval_opponent")
    ap.add_argument("--bench-also", action="append", default=[], dest="bench_also",
                    help="extra benchmark opponent (same syntax as --eval-opponent); "
                         "repeatable; logged but does not drive _best")
    ap.add_argument("--eval-seed-offset", type=int, default=0, dest="eval_off")
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

    print(f"[{args.players}p] opponent={args.opponent} | lr={args.lr} ent={args.ent} clip={args.clip}"
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
          f" ({args.eval_games} eval-pool games) -> _best")
    if args.bench_also:
        print(f"  also benchmarking vs: {', '.join(args.bench_also)} "
              f"({args.eval_games} games each per eval tick — logged, not selected on)")

    def _suffix(tag):
        return (args.out.replace(".pt", f"_{tag}.pt") if args.out.endswith(".pt")
                else args.out + f"_{tag}")

    seed_i = args.train_off
    best_bench = -1.0

    # ===== one-shot value warmup =====
    if args.value_warmup > 0:
        print(f"[warmup] collecting {args.warmup_games} games for value calibration...")
        wbatch, wwins = [], 0
        for _ in range(args.warmup_games):
            traj, adv, ret, _ = collect_episode(net, opp_draw, env, train_seed(seed_i), dev)
            seed_i += 1
            wwins += 1 if env._terminal_reward(0) > 0 else 0
            for t in range(len(traj)):
                wbatch.append([traj[t][3], ret[t]])
        print(f"[warmup] {len(wbatch)} states (collect win_rate {wwins/args.warmup_games:.2f}); "
              f"training value head {args.value_warmup} epochs...")
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
        print("[warmup] done.")

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
        for ep in range(args.epochs):
            np.random.shuffle(batch); kl_sum, kl_n = 0.0, 0
            for s in range(0, len(batch), args.minibatch):
                chunk = batch[s:s + args.minibatch]; opt.zero_grad()
                for cache, old_logp_per, advt, rett in chunk:
                    logp_per, active, val, ent = recompute_logp_value(net, cache, dev)
                    a_t = torch.tensor(advt, device=dev, dtype=torch.float32)
                    ratio = torch.exp(logp_per - old_logp_per)
                    s1 = ratio * a_t
                    s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_t
                    denom = active.sum().clamp(min=1)
                    pol = (-torch.min(s1, s2) * active).sum() / denom
                    vloss = F.mse_loss(val, torch.tensor(rett, device=dev, dtype=torch.float32))
                    ((pol + 0.5 * vloss - args.ent * ent) / len(chunk)).backward()
                    with torch.no_grad():
                        kl_sum += float(((old_logp_per - logp_per) * active).sum() / denom); kl_n += 1
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()
            mean_kl = kl_sum / max(kl_n, 1)
            if args.target_kl > 0 and mean_kl > args.target_kl:
                stopped_epoch = ep + 1; break

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
        bench_extras = []   # list of (display_name, win_rate)
        if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            bench = benchmark_winrate(net, dev, eval_opp_fn, args.eval_games,
                                      args.players, args.eval_off)
            if bench > best_bench:
                best_bench = bench
                torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                            "players": args.players}, _suffix("best"))
            for spec, fn in bench_also:
                wr = benchmark_winrate(net, dev, fn, args.eval_games,
                                       args.players, args.eval_off)
                bench_extras.append((spec, wr))

        if it % args.log_every == 0:
            wr = wins / args.episodes_per_iter
            bs = ""
            if bench is not None:
                bs = f"  bench {bench:.2f}"
                if bench_extras:
                    bs += " [" + "  ".join(f"{n} {wr:.2f}" for n, wr in bench_extras) + "]"
            sel = f"best_bench {best_bench:.2f}" if best_bench >= 0 else "best_bench --"
            line = (f"iter {it:4d}  mean_ep_reward {np.mean(ep_rewards):+.3f}  train_wr {wr:.2f}"
                    f"{bs}  {sel}  kl {mean_kl:.4f}"
                    f"{'' if stopped_epoch == args.epochs else f' (early-stop @ep{stopped_epoch})'}")
            if args.debug:
                line += (f"\n        raw_adv [{min(d['adv_min'] for d in dbgs):+.1f},"
                         f"{max(d['adv_max'] for d in dbgs):+.1f}] val "
                         f"[{min(d['val_min'] for d in dbgs):+.1f},{max(d['val_max'] for d in dbgs):+.1f}]"
                         f" ep_len {np.mean([d['ep_len'] for d in dbgs]):.0f}")
            print(line)

    torch.save({"model": net.state_dict(), "players": args.players}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
