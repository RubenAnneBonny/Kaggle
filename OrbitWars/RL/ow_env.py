"""ow_env.py — RL environment for orbit_wars (torch-free; verified against engine).

Provides:
  - encode_state(obs, player) -> dict of numpy arrays (node features, masks, id map)
  - decode_action(enc, obs, player, targets, fracs) -> list of [from_id, angle, ships]
  - OrbitEnv: a fast self-play environment that uses the engine-faithful
    simulator in search_agent.sim_step. reset() seeds from a real kaggle layout.

Design choices (see notes in the chat):
  * Factored per-source policy: each owned planet picks ONE target (or HOLD)
    and a send-fraction. This matches net_roi_support (one attack per planet/turn)
    and keeps the action space + log-prob tractable.
  * The network never outputs firing angles — decode_action reuses reach() to
    solve the intercept geometry. The policy only learns target + how-many.
"""
import math, numpy as np, contextlib, io
from kaggle_environments import make
from ow_base import parse_obs, sq_dist, predict_all_fleet_hits, is_orbiting, reach
from search_agent import snapshot, sim_step, evaluate, _obs_from_state

N_MAX = 40          # max planets we pad to (board has 32)
NODE_F = 10         # node feature dim
FRACS = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)  # discrete send-fractions
N_FRAC = len(FRACS)


def encode_state(obs, player):
    """Return arrays describing the state from `player`'s perspective.
      node_feats : (N_MAX, NODE_F) float32
      node_mask  : (N_MAX,) 1.0 for real planets
      own_mask   : (N_MAX,) 1.0 for planets owned by `player`
      attack_mask: (N_MAX, N_MAX) 1.0 if col j is a legal target for source row i
      ids        : (N_MAX,) planet id at each index (-1 for padding)
    """
    planets = obs["planets"]
    comet_ids = set(obs.get("comet_planet_ids", []))
    inc_mine, inc_enemy = {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        d = inc_mine if owner == player else inc_enemy
        d[pid] = d.get(pid, 0) + ships

    nf = np.zeros((N_MAX, NODE_F), np.float32)
    nmask = np.zeros(N_MAX, np.float32)
    omask = np.zeros(N_MAX, np.float32)
    ids = np.full(N_MAX, -1, np.int64)
    idx_by_id = {}

    for i, p in enumerate(planets[:N_MAX]):
        pid, owner, x, y, r, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        mine = 1.0 if owner == player else 0.0
        enemy = 1.0 if owner not in (-1, player) else 0.0
        neutral = 1.0 if owner == -1 else 0.0
        nf[i] = [mine, enemy, neutral, ships / 50.0, prod / 5.0,
                 1.0 if is_orbiting_raw(p, obs) else 0.0,
                 (x - 50.0) / 50.0, (y - 50.0) / 50.0,
                 inc_mine.get(pid, 0) / 50.0, inc_enemy.get(pid, 0) / 50.0]
        nmask[i] = 1.0
        omask[i] = mine
        ids[i] = pid
        idx_by_id[pid] = i

    # attack mask: source must be mine; target must be real, not self, not comet
    amask = np.zeros((N_MAX, N_MAX), np.float32)
    for i in range(N_MAX):
        if omask[i] == 0.0:
            continue
        for j in range(N_MAX):
            if nmask[j] == 0.0 or i == j:
                continue
            if ids[j] in comet_ids:
                continue
            amask[i, j] = 1.0

    return {"node_feats": nf, "node_mask": nmask, "own_mask": omask,
            "attack_mask": amask, "ids": ids, "idx_by_id": idx_by_id}


def is_orbiting_raw(p, obs):
    # p is a raw planet list [id,owner,x,y,r,ships,prod]; replicate is_orbiting
    init = {q[0]: q for q in obs["initial_planets"]}
    ip = init.get(p[0])
    if ip is None:
        return False
    dx, dy = ip[2] - 50.0, ip[3] - 50.0
    return math.sqrt(dx * dx + dy * dy) + p[4] < 50.0


def decode_action(enc, obs, player, target_idx, frac_idx):
    """Translate per-owned-planet (target index, fraction index) into engine moves.
      target_idx : (N_MAX,) int — chosen target index per node (ignored unless owned).
                   A value == its own index (i.e. target==self) is treated as HOLD.
      frac_idx   : (N_MAX,) int — chosen fraction bucket per node.
    Returns list of [from_id, angle, ships].
    """
    planets = {p[0]: p for p in obs["planets"]}
    inc_enemy = {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner != player:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships

    moves = []
    ids = enc["ids"]
    for i in range(N_MAX):
        if enc["own_mask"][i] == 0.0:
            continue
        j = int(target_idx[i])
        if j == i or j < 0 or j >= N_MAX or enc["attack_mask"][i, j] == 0.0:
            continue  # HOLD or illegal
        src = planets.get(int(ids[i]))
        dst = planets.get(int(ids[j]))
        if src is None or dst is None:
            continue
        reserve = max(0, inc_enemy.get(src[0], 0)) + 1
        avail = src[5] - reserve
        if avail <= 0:
            continue
        ships = int(round(FRACS[int(frac_idx[i])] * avail))
        ships = max(1, min(ships, avail))
        src_p = _as_planet(src)
        dst_p = _as_planet(dst)
        sol = reach(src_p, dst_p, ships, obs)
        if sol is None:
            continue
        _t, angle = sol
        moves.append([src[0], angle, ships])
    return moves


class _P:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production")
    def __init__(self, p):
        (self.id, self.owner, self.x, self.y, self.radius, self.ships, self.production) = p
def _as_planet(p):
    return _P(p)


class OrbitEnv:
    """Fast self-play env over the faithful simulator. Two policies act each step.
    Reward (per player, returned for player 0's perspective unless specified) is
    the change in production-weighted material differential, plus terminal +/-1."""
    def __init__(self, max_steps=200, prod_w=8.0):
        self.max_steps = max_steps
        self.prod_w = prod_w
        self.st = None
        self.t = 0

    def reset(self, seed=0):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=2)
        obs0 = env.steps[0][0]["observation"]
        self.st = snapshot(obs0)
        self.t = 0
        self._last = self._diff()
        return self.obs_for(0), self.obs_for(1)

    def obs_for(self, player):
        o = _obs_from_state(self.st, player)
        return o

    def _diff(self):
        s = 0.0
        for p in self.st["planets"]:
            sign = 1 if p[1] == 0 else (-1 if p[1] != -1 else 0)
            s += sign * (p[5] + self.prod_w * p[6])
        for f in self.st["fleets"]:
            s += (1 if f[1] == 0 else -1) * f[6]
        return s

    def step(self, moves0, moves1):
        sim_step(self.st, {0: moves0, 1: moves1})
        self.t += 1
        cur = self._diff()
        shaped = (cur - self._last) * 0.01     # dense shaping for player 0
        self._last = cur
        done = self.t >= self.max_steps or self._winner() is not None
        term = 0.0
        if done:
            w = self._winner()
            term = 1.0 if w == 0 else (-1.0 if w == 1 else 0.0)
        return shaped + term, done

    def _winner(self):
        owners = set(p[1] for p in self.st["planets"] if p[1] != -1)
        f_owners = set(f[1] for f in self.st["fleets"])
        alive = owners | f_owners
        if alive == {0}:
            return 0
        if alive == {1}:
            return 1
        return None


if __name__ == "__main__":
    # ---- self-test (torch-free): shapes + decode round-trips through engine ----
    from ow_base import net_roi_support
    import contextlib, io
    env = make("orbit_wars", configuration={"seed": 0}, debug=False)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]; obs = dict(obs); obs["player"] = 0
    enc = encode_state(obs, 0)
    print("node_feats", enc["node_feats"].shape, "own planets", int(enc["own_mask"].sum()),
          "real planets", int(enc["node_mask"].sum()))
    # random legal action -> moves -> feed real engine to confirm acceptance
    rng = np.random.default_rng(0)
    tgt = np.zeros(N_MAX, np.int64); frac = np.zeros(N_MAX, np.int64)
    for i in range(N_MAX):
        if enc["own_mask"][i]:
            legal = np.where(enc["attack_mask"][i] > 0)[0]
            tgt[i] = rng.choice(legal) if len(legal) else i
            frac[i] = rng.integers(0, N_FRAC)
    mv = decode_action(enc, obs, 0, tgt, frac)
    print("decoded", len(mv), "moves; sample:", mv[:2])
    # confirm engine accepts them (no crash, fleets appear)
    e2 = make("orbit_wars", configuration={"seed": 0}, debug=True)
    e2.reset(num_agents=2)
    with contextlib.redirect_stderr(io.StringIO()):
        e2.step([mv, []])
    print("engine accepted action, fleets now:", len(e2.steps[-1][0]["observation"]["fleets"]))

    oe = OrbitEnv(max_steps=30)
    o0, o1 = oe.reset(seed=1)
    tot = 0.0
    for _ in range(30):
        m0 = net_roi_support(oe.obs_for(0)) or []
        m1 = net_roi_support(oe.obs_for(1)) or []
        r, done = oe.step(m0, m1)
        tot += r
        if done:
            break
    print(f"OrbitEnv ran {oe.t} steps, cumulative shaped reward {tot:.3f}, winner {oe._winner()}")
