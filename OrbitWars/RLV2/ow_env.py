"""ow_env.py (v2) — RL environment for orbit_wars (torch-free).

Major changes from v1:
  * NET-ATTACK feature: per planet, incoming-own minus incoming-enemy ships
    (a single signed number) added to the node features. NODE_F 10 -> 11.
  * CONTINUOUS fraction decode. The network outputs a fraction in [0,1] per
    source ("send this fraction of available ships"). Anything below FRAC_FLOOR
    (default 5%) is treated as "send nothing" — this gives a clean way to mean
    zero without depending on exact rounding. HOLD (the discrete head) is still
    the primary "do nothing"; the floor is the safety net.
  * COMET-SAFE / PLANET-SAFE launches. Every launch angle is checked with
    path_clear (using the comet's REAL trajectory). A blocked launch is skipped
    rather than flying a fleet through a comet/planet and losing it.
  * WITHIN-TURN COORDINATION. Sources accumulate ships toward a target (so two
    planets can jointly capture one neither could alone), with a generous cap to
    stop pointless dogpiling. The old hard "can't solo-capture -> skip" is OFF
    by default (it forbids coordinated attacks); it's available via hard_skip.
  * N-PLAYER OrbitEnv (2 or 4) at the engine's true 500-step cap, with faithful
    comets and the engine's exact end-of-game scoring (max ship total wins).
"""
import math, numpy as np
from kaggle_environments import make
from ow_base import (predict_all_fleet_hits, reach, is_orbiting, path_clear,
                     comet_path_map, _comet_pos)
from search_agent import snapshot, sim_step, _obs_from_state

N_MAX = 40            # pad to this many planet nodes (board has up to ~32 + comets)
NODE_F = 11           # v2 node feature dim (was 10; +net-attack)
FRAC_FLOOR = 0.05     # fractions below this == "send nothing"


# --------------------------------------------------------------------------- #
# State encoding
# --------------------------------------------------------------------------- #
def encode_state(obs, player):
    """Arrays describing the state from `player`'s perspective.
      node_feats : (N_MAX, NODE_F) float32
      node_mask  : (N_MAX,) 1.0 for real planets
      own_mask   : (N_MAX,) 1.0 for planets owned by `player`
      attack_mask: (N_MAX, N_MAX) 1.0 if col j is a legal target for source row i
      ids        : (N_MAX,) planet id at each index (-1 padding)
    Comets are included as nodes (so the net can 'see' them) but are NEVER legal
    targets (masked out of attack_mask)."""
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
        im = inc_mine.get(pid, 0)
        ie = inc_enemy.get(pid, 0)
        nf[i] = [mine, enemy, neutral, ships / 50.0, prod / 5.0,
                 1.0 if is_orbiting_raw(p, obs) else 0.0,
                 (x - 50.0) / 50.0, (y - 50.0) / 50.0,
                 im / 50.0, ie / 50.0,
                 (im - ie) / 50.0]                       # v2: net-attack
        nmask[i] = 1.0
        omask[i] = mine
        ids[i] = pid
        idx_by_id[pid] = i

    amask = np.zeros((N_MAX, N_MAX), np.float32)
    for i in range(N_MAX):
        if omask[i] == 0.0:
            continue
        for j in range(N_MAX):
            if nmask[j] == 0.0 or i == j:
                continue
            if ids[j] in comet_ids:                      # never target comets
                continue
            amask[i, j] = 1.0

    return {"node_feats": nf, "node_mask": nmask, "own_mask": omask,
            "attack_mask": amask, "ids": ids, "idx_by_id": idx_by_id}


def is_orbiting_raw(p, obs):
    init = {q[0]: q for q in obs["initial_planets"]}
    ip = init.get(p[0])
    if ip is None:
        return False
    dx, dy = ip[2] - 50.0, ip[3] - 50.0
    return math.sqrt(dx * dx + dy * dy) + p[4] < 50.0


class _P:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production")
    def __init__(self, p):
        (self.id, self.owner, self.x, self.y, self.radius, self.ships, self.production) = p
def _as_planet(p):
    return _P(p)


# --------------------------------------------------------------------------- #
# Action decoding  (continuous fraction, comet-safe, coordinated)
# --------------------------------------------------------------------------- #
def decode_action(enc, obs, player, target_idx, frac_value,
                  frac_floor=FRAC_FLOOR, coord_cap=2.0, hard_skip=False):
    """Translate per-owned-planet (target index, CONTINUOUS fraction) to moves.

      target_idx : (N_MAX,) int — chosen target index per node (HOLD == self/N_MAX).
      frac_value : (N_MAX,) float in [0,1] — fraction of AVAILABLE ships to send.
      frac_floor : fractions below this send nothing.
      coord_cap  : allow ships committed to a target to reach coord_cap * its
                   effective garrison before refusing further sources (lets two
                   planets co-capture one, but stops endless dogpiling). None
                   disables the cap entirely.
      hard_skip  : if True, restore v1 behaviour — a source whose own send can't
                   capture the target alone is skipped. OFF by default because it
                   forbids coordinated attacks.

    Returns list of [from_id, angle, ships]. Launches whose straight path would
    sweep a comet or another planet are skipped (comet-safety)."""
    planets = {p[0]: p for p in obs["planets"]}
    inc_mine_tgt, inc_enemy_tgt, inc_enemy_src = {}, {}, {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner == player:
            inc_mine_tgt[pid] = inc_mine_tgt.get(pid, 0) + ships
        else:
            inc_enemy_tgt[pid] = inc_enemy_tgt.get(pid, 0) + ships
            inc_enemy_src[pid] = inc_enemy_src.get(pid, 0) + ships

    def eff_garrison(dst):
        return (dst[5] + inc_enemy_tgt.get(dst[0], 0) - inc_mine_tgt.get(dst[0], 0))

    moves = []
    ids = enc["ids"]
    outgoing = {}          # src id -> ships already launched this turn
    committed = {}         # target id -> ships already committed this turn
    for i in range(N_MAX):
        if enc["own_mask"][i] == 0.0:
            continue
        j = int(target_idx[i])
        if j == i or j < 0 or j >= N_MAX or enc["attack_mask"][i, j] == 0.0:
            continue                                     # HOLD or illegal
        f = float(frac_value[i])
        if f < frac_floor:
            continue                                     # "send nothing"
        src = planets.get(int(ids[i]))
        dst = planets.get(int(ids[j]))
        if src is None or dst is None:
            continue
        reserve = max(0, inc_enemy_src.get(src[0], 0)) + 1
        avail = src[5] - outgoing.get(src[0], 0) - reserve
        if avail <= 0:
            continue
        ships = int(round(f * avail))
        ships = max(1, min(ships, avail))

        dst_is_mine = (dst[1] == player)
        if not dst_is_mine:
            eff = eff_garrison(dst)
            already = committed.get(dst[0], 0)
            if coord_cap is not None and eff > 0 and already >= coord_cap * eff:
                continue                                 # target already over-committed
            if hard_skip and (already + ships) < eff + 1:
                continue                                 # can't capture even with prior commits

        src_p = _as_planet(src)
        dst_p = _as_planet(dst)
        sol = reach(src_p, dst_p, ships, obs)
        if sol is None:
            continue
        t_star, angle = sol
        if not path_clear(src_p, angle, ships, t_star, obs, dst[0]):
            continue                                     # would hit a comet/planet -> skip
        moves.append([src[0], angle, ships])
        outgoing[src[0]] = outgoing.get(src[0], 0) + ships
        committed[dst[0]] = committed.get(dst[0], 0) + ships
    return moves


# --------------------------------------------------------------------------- #
# N-player self-play environment (faithful comets, 500-step cap)
# --------------------------------------------------------------------------- #
def _score(st, player):
    s = 0
    for p in st["planets"]:
        if p[1] == player:
            s += p[5]
    for f in st["fleets"]:
        if f[1] == player:
            s += f[6]
    return s


class OrbitEnv:
    """Fast self-play env over the faithful (comet-bearing) simulator.

    Supports 2 OR 4 players. Reward is returned from a FOCAL player's view
    (default 0): a small potential-style shaping on (my_score - best_opponent
    score) plus a dominant terminal +/-5 decided exactly like the engine
    (highest positive ship total wins; ties count as wins, matching reward=+1)."""
    def __init__(self, max_steps=500, num_players=2, shaping=0.003, focal=0):
        assert num_players in (2, 4)
        self.max_steps = max_steps
        self.num_players = num_players
        self.shaping = shaping
        self.focal = focal
        self.st = None
        self.t = 0

    def reset(self, seed=0):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=self.num_players)
        episode_seed = (getattr(env, "info", None) or {}).get("seed", seed)
        obs0 = env.steps[0][0]["observation"]
        self.st = snapshot(obs0, episode_seed)
        self.t = 0
        self._last = self._lead(self.focal)
        return [self.obs_for(p) for p in range(self.num_players)]

    def obs_for(self, player):
        return _obs_from_state(self.st, player)

    def _lead(self, me):
        mine = _score(self.st, me)
        others = [_score(self.st, p) for p in range(self.num_players) if p != me]
        return mine - (max(others) if others else 0)

    def step(self, moves_by_player):
        """moves_by_player: list/dict of move-lists, one per player slot."""
        if isinstance(moves_by_player, dict):
            actions = {p: (moves_by_player.get(p) or []) for p in range(self.num_players)}
        else:
            actions = {p: (moves_by_player[p] or []) for p in range(self.num_players)}
        sim_step(self.st, actions)
        self.t += 1
        cur = self._lead(self.focal)
        shaped = (cur - self._last) * self.shaping
        self._last = cur
        done = self.t >= self.max_steps or self._winner() is not None
        term = 0.0
        if done:
            term = self._terminal_reward(self.focal)
        return shaped + term, done

    def _winner(self):
        alive = set(p[1] for p in self.st["planets"] if p[1] != -1)
        alive |= set(f[1] for f in self.st["fleets"])
        if len(alive) == 1:
            return next(iter(alive))
        if len(alive) == 0:
            return -1
        return None

    def _terminal_reward(self, me):
        scores = [_score(self.st, p) for p in range(self.num_players)]
        mx = max(scores)
        if scores[me] == mx and mx > 0:                  # engine: tie at max also wins
            return 5.0
        return -5.0


if __name__ == "__main__":
    # ---- torch-free self-test: encode + decode round-trip through real engine ----
    import contextlib, io
    from ow_base import net_roi_support, net_roi_aggressive
    for NP in (2, 4):
        env = make("orbit_wars", configuration={"seed": 5}, debug=False)
        env.reset(num_agents=NP)
        obs = dict(env.steps[0][0]["observation"]); obs["player"] = 0
        enc = encode_state(obs, 0)
        print(f"[{NP}p] node_feats {enc['node_feats'].shape} own {int(enc['own_mask'].sum())} "
              f"real {int(enc['node_mask'].sum())} net-attack[0:3]={enc['node_feats'][:3,10]}")
        rng = np.random.default_rng(0)
        tgt = np.full(N_MAX, N_MAX, np.int64); frac = np.zeros(N_MAX, np.float32)
        for i in range(N_MAX):
            if enc["own_mask"][i]:
                legal = np.where(enc["attack_mask"][i] > 0)[0]
                if len(legal):
                    tgt[i] = rng.choice(legal); frac[i] = rng.uniform(0.3, 1.0)
        mv = decode_action(enc, obs, 0, tgt, frac)
        print(f"     decoded {len(mv)} comet-safe moves; sample {mv[:2]}")

    oe = OrbitEnv(max_steps=120, num_players=2)
    obss = oe.reset(seed=5)
    tot = 0.0
    with contextlib.redirect_stderr(io.StringIO()):
        for _ in range(120):
            m0 = net_roi_aggressive(oe.obs_for(0)) or []
            m1 = net_roi_support(oe.obs_for(1)) or []
            r, done = oe.step([m0, m1])
            tot += r
            if done:
                break
    print(f"[2p] env ran {oe.t} steps, shaped+term reward {tot:.3f}, winner {oe._winner()}, "
          f"comets alive {sorted(oe.st['comet_ids'])}")

    oe4 = OrbitEnv(max_steps=120, num_players=4)
    oe4.reset(seed=5)
    with contextlib.redirect_stderr(io.StringIO()):
        for _ in range(120):
            ms = [net_roi_support(oe4.obs_for(p)) or [] for p in range(4)]
            r, done = oe4.step(ms)
            if done:
                break
    print(f"[4p] env ran {oe4.t} steps, scores {[_score(oe4.st,p) for p in range(4)]}, "
          f"winner {oe4._winner()}")
