"""parallel.py — CPU multiprocessing for the two CPU-bound, embarrassingly-parallel
phases of PPO: episode COLLECTION and EVAL.

After the forward is batched (see ppo.collect_batched / _batched_policy), both
collection and eval are bound by the Python game-sim + encode/decode, not the GPU
(the policy net is a tiny tensor). So a persistent Pool of CPU workers running whole
games scales ~linearly with cores — the same trick that makes the BC harvest fast.
Each worker runs the focal net on CPU; the main process keeps the GPU for the
batched PPO update.

Determinism: EVAL is greedy + fixed seeds, so the parallel win-rate is identical to
the sequential ppo.benchmark_winrate (verified by test_parallel.py). COLLECTION
samples actions, so workers use independent RNG (fine for PPO; not bit-reproducible).
Use --workers 1 for the deterministic in-process path.
"""
import contextlib, io, numpy as np, torch

_POOL = None
_NWORKERS = 1
_EXT_CACHE = {}          # path -> loaded external agent (per worker, reused across iters)


def _init_worker():
    torch.set_num_threads(1)        # N workers x many threads would oversubscribe cores


def init_pool(nworkers):
    """Create the persistent worker Pool (once). nworkers<=0 => all cores. Returns
    the actual worker count; 1 means 'no pool, run in-process' (caller falls back)."""
    global _POOL, _NWORKERS
    import multiprocessing as mp
    n = nworkers if nworkers > 0 else (mp.cpu_count() or 1)
    _NWORKERS = max(1, n)
    if _NWORKERS > 1 and _POOL is None:
        _POOL = mp.Pool(_NWORKERS, initializer=_init_worker)
    return _NWORKERS


def close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close(); _POOL.join(); _POOL = None


def _cpu_sd(state_dict):
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def _build_net(sd):
    from model import OrbitNet
    n = OrbitNet()
    n.load_state_dict(sd); n.eval()
    for p in n.parameters():
        p.requires_grad_(False)
    return n


def _load_externals(ext_paths):
    from ppo import _load_external_agent
    out = {}
    for name, path in ext_paths.items():
        if path not in _EXT_CACHE:
            _EXT_CACHE[path] = _load_external_agent(path)
        out[name] = _EXT_CACHE[path]
    return out


# --------------------------------------------------------------------------- #
# EVAL (greedy benchmark; deterministic -> identical to sequential)
# --------------------------------------------------------------------------- #
def _resolve_opp(spec, externals):
    import ow_base
    if spec.startswith("external:"):
        return externals[spec[len("external:"):]]
    if spec == "teacher":    return ow_base.net_roi_support
    if spec == "aggressive": return ow_base.net_roi_aggressive
    return getattr(ow_base, spec)


def _eval_worker(payload):
    from ppo import net_greedy_agent
    from quiet_kaggle import make
    from seeds import eval_seed
    sd, spec, ext_paths, idxs, num_players, eval_off = payload
    net = _build_net(sd)
    agent = net_greedy_agent(net, "cpu")
    opp = _resolve_opp(spec, _load_externals(ext_paths))
    wins = 0
    for i in idxs:
        slot = i % num_players
        order = [opp] * num_players
        order[slot] = agent
        e = make("orbit_wars", configuration={"seed": eval_seed(eval_off + i)}, debug=False)
        with contextlib.redirect_stderr(io.StringIO()):
            e.run(order)
        r = [s.reward if s.reward is not None else -1 for s in e.steps[-1]]
        wins += 1 if r[slot] == max(r) and r[slot] > 0 else 0
    return wins


def eval_winrate(net, spec, ext_paths, games, num_players, eval_off):
    """Parallel greedy benchmark. Same seeds / slot-rotation / scoring as the
    sequential ppo.benchmark_winrate, split round-robin across workers."""
    sd = _cpu_sd(net.state_dict())
    idx_chunks = [list(range(games))[w::_NWORKERS] for w in range(_NWORKERS)]
    payloads = [(sd, spec, ext_paths, ch, num_players, eval_off) for ch in idx_chunks if ch]
    return sum(_POOL.map(_eval_worker, payloads)) / max(games, 1)


# --------------------------------------------------------------------------- #
# COLLECTION (sampled rollouts; returns NumPy caches for the batched GPU update)
# --------------------------------------------------------------------------- #
def _collect_worker(payload):
    from ppo import collect_batched, make_opp_draw
    from ow_env import OrbitEnv
    sd, league_sds, mix_weights, ext_paths, seeds, num_players, max_steps = payload
    net = _build_net(sd)
    league = [_build_net(s) for s in league_sds] or [net]
    opp_draw = make_opp_draw(mix_weights, league, _load_externals(ext_paths), "cpu")
    envs = [OrbitEnv(max_steps=max_steps, num_players=num_players, focal=0) for _ in seeds]
    results = collect_batched(net, opp_draw, envs, seeds, "cpu")
    wins = [1 if envs[j]._terminal_reward(0) > 0 else 0 for j in range(len(envs))]
    return results, wins


def collect(net, self_pool, mix_weights, ext_paths, seeds, num_players, max_steps):
    """Parallel collection across workers. Returns (results, wins) in seed order;
    results[i] = (traj, adv, ret, dbg) with NumPy caches, wins[i] = win flag."""
    sd = _cpu_sd(net.state_dict())
    league_sds = [_cpu_sd(n.state_dict()) for n in self_pool]
    idx_chunks = [list(range(len(seeds)))[w::_NWORKERS] for w in range(_NWORKERS)]
    idx_chunks = [ch for ch in idx_chunks if ch]
    payloads = [(sd, league_sds, mix_weights, ext_paths, [seeds[i] for i in ch],
                 num_players, max_steps) for ch in idx_chunks]
    parts = _POOL.map(_collect_worker, payloads)
    results = [None] * len(seeds)
    wins = [0] * len(seeds)
    for ch, (res, wn) in zip(idx_chunks, parts):
        for k, gi in enumerate(ch):
            results[gi] = res[k]; wins[gi] = wn[k]
    return results, wins
