"""search_agent: a real lookahead agent.

Pieces:
  1. A faithful one-tick forward simulator (sim_step) matching the engine:
     launch -> production -> planet rotation -> fleet movement w/ swept
     collision -> combat resolution.
  2. An evaluation function on a simulated state (ship + production differential
     with a horizon weight on production).
  3. A search: generate a handful of candidate ROOT move-sets (variants of the
     net_roi_support policy), simulate each forward ROLLOUT_DEPTH ticks with both
     players following a fast default policy, evaluate, and play the best root.
"""
import math, copy, random
from ow_base import (parse_obs, sq_dist, predict_all_fleet_hits, reach,
                     how_many_send, how_many_send_improved, get_field,
                     is_orbiting, net_roi_support, CENTER)

# Faithful comet support: pull the engine's own comet generator + constants so
# the fast simulator spawns/moves/expires comets EXACTLY like the real game
# (v1 froze comets in place, which is wrong — they fly across at cometSpeed and
# vanish off-board, taking any ships on them). If the engine isn't importable
# (shouldn't happen on Kaggle), we degrade to "advance/expire only, no spawn".
try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import (
        generate_comet_paths as _gen_comet_paths,
        COMET_SPAWN_STEPS as _COMET_SPAWN_STEPS,
        COMET_RADIUS as _COMET_RADIUS,
        COMET_PRODUCTION as _COMET_PRODUCTION,
    )
except Exception:                                   # pragma: no cover
    _gen_comet_paths = None
    _COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
    _COMET_RADIUS, _COMET_PRODUCTION = 1.0, 1
COMET_SPEED = 4.0

BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0


def _swept(A, B, P0, P1, r):
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    return (-b + sq) / (2.0 * a) >= 0.0 and (-b - sq) / (2.0 * a) <= 1.0


def _pt_seg(C, A, B):
    ax, ay = A; bx, by = B; cx, cy = C
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(cx - ax, cy - ay)
    t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / L2))
    return math.hypot(cx - (ax + t * dx), cy - (ay + t * dy))


def snapshot(obs, episode_seed=0):
    """Mutable lightweight state from a live observation.

    Now also carries the live comet groups (deep-copied) and the episode seed
    so the simulator can advance/expire comets along their real paths AND spawn
    new ones at the engine's spawn steps with the engine's own RNG."""
    planets = [list(p) for p in obs["planets"]]          # [id,owner,x,y,r,ships,prod]
    fleets = [list(f) for f in obs["fleets"]]            # [id,owner,x,y,angle,from,ships]
    init = {p[0]: list(p) for p in obs["initial_planets"]}
    comets = []
    for g in (obs.get("comets", []) or []):
        comets.append({
            "planet_ids": list(g["planet_ids"]),
            "paths": [list(p) for p in g["paths"]],      # list of [ [x,y], ... ]
            "path_index": g["path_index"],
        })
    return {
        "planets": planets, "fleets": fleets, "init": init,
        "ang": obs.get("angular_velocity", 0.0),
        "step": obs.get("step", 0),
        "next_fid": obs.get("next_fleet_id", 100000),
        "comet_ids": set(obs.get("comet_planet_ids", [])),
        "comets": comets,
        "seed": episode_seed,
    }


def _spawn_comets(st):
    """Replicate the engine's mid-game comet spawn (steps 50/150/250/350/450)."""
    if _gen_comet_paths is None:
        return
    nxt = st["step"] + 1
    if nxt not in _COMET_SPAWN_STEPS:
        return
    comet_rng = random.Random(f"orbit_wars-comet-{st['seed']}-{nxt}")
    init_list = list(st["init"].values())
    paths = _gen_comet_paths(init_list, st["ang"], nxt,
                             list(st["comet_ids"]), COMET_SPEED, rng=comet_rng)
    if not paths:
        return
    next_id = max(p[0] for p in st["planets"]) + 1
    comet_ships = min(comet_rng.randint(1, 99), comet_rng.randint(1, 99),
                      comet_rng.randint(1, 99), comet_rng.randint(1, 99))
    group = {"planet_ids": [], "paths": [list(p) for p in paths], "path_index": -1}
    for i in range(len(paths)):
        pid = next_id + i
        group["planet_ids"].append(pid)
        st["comet_ids"].add(pid)
        row = [pid, -1, -99.0, -99.0, _COMET_RADIUS, comet_ships, _COMET_PRODUCTION]
        st["planets"].append(row)
        st["init"][pid] = row[:]
    st["comets"].append(group)


def sim_step(st, actions):
    """Advance one tick. actions = {player: [[from,angle,ships],...]}. Mutates st.
    Faithful to the engine including comet spawn/movement/expiry."""
    # (a) remove already-expired comets (path exhausted) before anything else
    _drop_expired(st)
    # (b) spawn new comets at the engine's spawn steps
    _spawn_comets(st)

    planets = st["planets"]; fleets = st["fleets"]
    pby = {p[0]: p for p in planets}

    # launch
    for pl, moves in actions.items():
        for mv in moves or []:
            if len(mv) != 3:
                continue
            fid, angle, ships = mv[0], mv[1], int(mv[2])
            p = pby.get(fid)
            if p is not None and p[1] == pl and ships > 0 and p[5] >= ships:
                p[5] -= ships
                sx = p[2] + math.cos(angle) * (p[4] + 0.1)
                sy = p[3] + math.sin(angle) * (p[4] + 0.1)
                fleets.append([st["next_fid"], pl, sx, sy, angle, fid, ships])
                st["next_fid"] += 1

    # production
    for p in planets:
        if p[1] != -1:
            p[5] += p[6]

    # planet positions: ordinary planets ROTATE; comets follow their PATH.
    # The engine rotates using the PRE-increment step, then the framework bumps
    # the step — so we rotate with `cur` and only advance the counter afterward.
    cur = st["step"]
    paths = {}
    for p in planets:
        if p[0] in st["comet_ids"]:
            continue                                  # handled below
        old = (p[2], p[3]); new = old
        ip = st["init"].get(p[0])
        if ip is not None:
            dx, dy = ip[2] - CENTER, ip[3] - CENTER
            r = math.sqrt(dx * dx + dy * dy)
            if r + p[4] < ROTATION_RADIUS_LIMIT:
                a0 = math.atan2(dy, dx) + st["ang"] * cur
                new = (CENTER + r * math.cos(a0), CENTER + r * math.sin(a0))
        paths[p[0]] = (old, new, True)

    # comet advance along precomputed paths (path_index += 1); mark newly expired
    expired = []
    for g in st["comets"]:
        g["path_index"] += 1
        idx = g["path_index"]
        for i, pid in enumerate(g["planet_ids"]):
            p = pby.get(pid)
            if p is None:
                continue
            old = (p[2], p[3])
            ppath = g["paths"][i]
            if idx >= len(ppath):
                expired.append(pid)
                paths[pid] = (old, old, True)         # stays put this tick, removed after combat
            else:
                new = (ppath[idx][0], ppath[idx][1])
                check = old[0] >= 0                   # first placement appears mid-tick: no collision
                paths[pid] = (old, new, check)

    st["step"] = cur + 1                              # advance the tick counter now
    # fleet movement + swept collision
    combat = {p[0]: [] for p in planets}
    remove = []
    for f in fleets:
        ships = f[6]
        speed = min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)
        old = (f[2], f[3])
        f[2] += math.cos(f[4]) * speed
        f[3] += math.sin(f[4]) * speed
        new = (f[2], f[3])
        hit = False
        for p in planets:
            pa = paths.get(p[0])
            if pa is None or not pa[2]:
                continue
            if _swept(old, new, pa[0], pa[1], p[4]):
                combat[p[0]].append(f); remove.append(f); hit = True; break
        if hit:
            continue
        if not (0 <= f[2] <= BOARD_SIZE and 0 <= f[3] <= BOARD_SIZE):
            remove.append(f); continue
        if _pt_seg((CENTER, CENTER), old, new) < SUN_RADIUS:
            remove.append(f); continue

    for p in planets:
        pa = paths.get(p[0])
        if pa is not None:
            p[2], p[3] = pa[1]
    st["fleets"] = [f for f in fleets if f not in remove]

    # combat
    for pid, flist in combat.items():
        p = pby.get(pid)
        if not p or not flist:
            continue
        ps = {}
        for f in flist:
            ps[f[1]] = ps.get(f[1], 0) + f[6]
        sp = sorted(ps.items(), key=lambda kv: kv[1], reverse=True)
        top_pl, top = sp[0]
        if len(sp) > 1:
            sec = sp[1][1]
            surv = top - sec
            if sp[0][1] == sp[1][1]:
                surv = 0
            owner = top_pl if surv > 0 else -1
        else:
            owner, surv = top_pl, top
        if surv > 0:
            if p[1] == owner:
                p[5] += surv
            else:
                p[5] -= surv
                if p[5] < 0:
                    p[1] = owner; p[5] = abs(p[5])

    # remove comets that expired this tick (flew off-board -> ships are lost)
    if expired:
        _remove_ids(st, set(expired))
    return st


def _drop_expired(st):
    """Pre-tick removal of comets whose path is already exhausted."""
    dead = set()
    for g in st["comets"]:
        idx = g["path_index"]
        for i, pid in enumerate(g["planet_ids"]):
            if idx >= len(g["paths"][i]):
                dead.add(pid)
    if dead:
        _remove_ids(st, dead)


def _remove_ids(st, dead):
    st["planets"] = [p for p in st["planets"] if p[0] not in dead]
    st["comet_ids"] -= dead
    for pid in dead:
        st["init"].pop(pid, None)
    for g in st["comets"]:
        # drop dead ids and their parallel paths together
        keep = [(pid, g["paths"][i]) for i, pid in enumerate(g["planet_ids"])
                if pid not in dead]
        g["planet_ids"] = [k[0] for k in keep]
        g["paths"] = [k[1] for k in keep]
    st["comets"] = [g for g in st["comets"] if g["planet_ids"]]


def evaluate(st, me, prod_w=8.0):
    """Material + production differential from me's perspective."""
    s_ship = s_prod = 0.0
    for p in st["planets"]:
        sign = 0
        if p[1] == me:
            sign = 1
        elif p[1] != -1:
            sign = -1
        s_ship += sign * p[5]
        s_prod += sign * p[6]
    for f in st["fleets"]:
        s_ship += (1 if f[1] == me else -1) * f[6]
    return s_ship + prod_w * s_prod


# ---- default policy used for rollouts (must accept a faux-obs) ----
def _obs_from_state(st, player):
    return {
        "planets": [list(p) for p in st["planets"]],
        "fleets": [list(f) for f in st["fleets"]],
        "initial_planets": [list(v) for v in st["init"].values()],
        "angular_velocity": st["ang"], "step": st["step"],
        "next_fleet_id": st["next_fid"],
        "comet_planet_ids": list(st["comet_ids"]),
        "comets": [{"planet_ids": list(g["planet_ids"]),
                    "paths": g["paths"], "path_index": g["path_index"]}
                   for g in st["comets"]],
        "player": player, "remainingOverageTime": 60,
    }


def _rollout_policy(st, player):
    try:
        return net_roi_support(_obs_from_state(st, player)) or []
    except Exception:
        return []


def make_search_agent(
    ROLLOUT_DEPTH=6,        # ticks to simulate forward
    PROD_W=8.0,             # production weight in eval
    N_VARIANTS=4,           # candidate root move-sets to compare
    OPP_MODEL="net",        # "net" = opponent plays net_roi_support, "idle" = nothing
):
    def candidate_roots(obs, me):
        """Generate a few root move-sets to compare. All are perturbations of
        the net_roi_support policy so the search only has to pick among sane plays."""
        base = net_roi_support(obs) or []
        roots = [("base", base)]

        # Variant: also send idle surplus from safe rear planets one hop forward
        c_player, mine, targets, comet_ids, _ = parse_obs(obs)
        enemies = [p for p in targets if p.owner not in (-1, me)]
        committed = {mv[0] for mv in base}
        if enemies:
            extra = list(base)
            front = sorted(mine, key=lambda m: min(sq_dist(m, e) for e in enemies))
            front_ids = {f.id for f in front[:max(1, len(front) // 2)]}
            for m in mine:
                if m.id in committed or m.id in front_ids or m.ships <= 3:
                    continue
                tgt = min((f for f in front if f.id != m.id),
                          key=lambda f: sq_dist(m, f), default=None)
                if tgt is None:
                    continue
                sol = reach(m, tgt, m.ships - 1, obs)
                if sol:
                    extra.append([m.id, sol[1], m.ships - 1])
            roots.append(("forward_push", extra))

        # Variant: hold everything (pure defense / economy) — base defense only
        # approximated by sending nothing offensive: keep only moves into owned planets
        mine_ids = {m.id for m in mine}
        hold = [mv for mv in base
                if any(p.id == mv[0] for p in mine)]  # keep launches; filter below
        # actually: "conservative" = drop the lowest-value half of attacks
        if len(base) > 1:
            roots.append(("conservative", base[:max(1, len(base) // 2)]))

        # Variant: aggressive — base plus push ALL surplus forward
        roots.append(("base2", base))
        return roots[:N_VARIANTS]

    def agent(obs):
        me = obs["player"]
        roots = candidate_roots(obs, me)
        best_moves, best_val = (roots[0][1] if roots else []), -1e18

        for _name, root in roots:
            st = snapshot(obs)
            opp = 1 - me
            # tick 0: my root vs opponent default
            opp_moves = _rollout_policy(st, opp) if OPP_MODEL == "net" else []
            sim_step(st, {me: root, opp: opp_moves})
            # rollout
            for _ in range(ROLLOUT_DEPTH - 1):
                a_me = _rollout_policy(st, me)
                a_op = _rollout_policy(st, opp) if OPP_MODEL == "net" else []
                sim_step(st, {me: a_me, opp: a_op})
            val = evaluate(st, me, PROD_W)
            if val > best_val:
                best_val, best_moves = val, root
        return best_moves

    agent.__name__ = "search_agent"
    return agent


search_agent = make_search_agent()


# ---------------------------------------------------------------------------
# Fast rollout policy: approximate net_roi_support's material flow WITHOUT the
# expensive angle/intercept solver. For rollouts we only need roughly-correct
# ship/production evolution to evaluate a root; exact firing angles don't
# matter because we re-search every real turn. We aim fleets straight at the
# target's CURRENT position (good enough over a short horizon) and size with a
# simple closed-form estimate.
# ---------------------------------------------------------------------------
def _fast_rollout(st, me):
    planets = st["planets"]
    mine = [p for p in planets if p[1] == me]
    if not mine:
        return []
    tgts = [p for p in planets if p[1] != me and p[0] not in st["comet_ids"]]
    if not tgts:
        return []
    # crude incoming accounting from fleets in flight
    inc = {}
    for f in st["fleets"]:
        # approximate which planet a fleet is heading to: nearest in its heading
        pass
    moves = []
    used_t = set()
    def d(a, b): return math.hypot(a[2] - b[2], a[3] - b[3])
    for src in mine:
        avail = src[5] - 1
        if avail <= 0:
            continue
        cand = [t for t in tgts if t[0] not in used_t and 0 < t[5] < avail]
        if not cand:
            continue
        # ROI: production / garrison - dist penalty
        best = max(cand, key=lambda t: t[6] / (t[5] + 1) - 0.02 * d(src, t))
        dist = d(src, best)
        # need to overcome garrison + production accrued over flight (~dist/3 ticks)
        need = int(best[5] + (best[6] if best[1] != -1 else 0) * (dist / 3.0) + 2)
        send = min(avail, max(need, 1))
        if send <= 0:
            continue
        angle = math.atan2(best[3] - src[3], best[2] - src[2])
        moves.append([src[0], angle, send])
        used_t.add(best[0])
    return moves


def make_search_agent_fast(
    ROLLOUT_DEPTH=8,
    PROD_W=8.0,
    N_VARIANTS=4,
    OPP_FAST=True,
):
    def candidate_roots(obs, me):
        base = net_roi_support(obs) or []
        roots = [("base", base)]
        c_player, mine, targets, comet_ids, _ = parse_obs(obs)
        enemies = [p for p in targets if p.owner not in (-1, me)]
        committed = {mv[0] for mv in base}
        if enemies:
            extra = list(base)
            front = sorted(mine, key=lambda m: min(sq_dist(m, e) for e in enemies))
            front_ids = {f.id for f in front[:max(1, len(front) // 2)]}
            for m in mine:
                if m.id in committed or m.id in front_ids or m.ships <= 3:
                    continue
                tgt = min((f for f in front if f.id != m.id),
                          key=lambda f: sq_dist(m, f), default=None)
                if tgt is None:
                    continue
                sol = reach(m, tgt, m.ships - 1, obs)
                if sol:
                    extra.append([m.id, sol[1], m.ships - 1])
            roots.append(("forward_push", extra))
        if len(base) > 1:
            roots.append(("conservative", base[:max(1, len(base) // 2)]))
        roots.append(("base2", base))
        return roots[:N_VARIANTS]

    def agent(obs):
        me = obs["player"]; opp = 1 - me
        roots = candidate_roots(obs, me)
        best_moves, best_val = (roots[0][1] if roots else []), -1e18
        for _name, root in roots:
            st = snapshot(obs)
            op0 = _fast_rollout(st, opp) if OPP_FAST else []
            sim_step(st, {me: root, opp: op0})
            for _ in range(ROLLOUT_DEPTH - 1):
                sim_step(st, {me: _fast_rollout(st, me),
                              opp: _fast_rollout(st, opp) if OPP_FAST else []})
            val = evaluate(st, me, PROD_W)
            if val > best_val:
                best_val, best_moves = val, root
        return best_moves

    agent.__name__ = "search_fast"
    return agent


search_fast = make_search_agent_fast()


def make_search_margin(ROLLOUT_DEPTH=8, PROD_W=8.0, MARGIN=15.0):
    """Only deviate from the net_roi_support base move if an alternative root
    beats it by at least MARGIN in simulated eval. Otherwise play base."""
    def candidate_roots(obs, me):
        base = net_roi_support(obs) or []
        roots = [("base", base)]
        _, mine, targets, comet_ids, _ = parse_obs(obs)
        enemies = [p for p in targets if p.owner not in (-1, me)]
        committed = {mv[0] for mv in base}
        if enemies:
            extra = list(base)
            front = sorted(mine, key=lambda m: min(sq_dist(m, e) for e in enemies))
            front_ids = {f.id for f in front[:max(1, len(front) // 2)]}
            for m in mine:
                if m.id in committed or m.id in front_ids or m.ships <= 3:
                    continue
                tgt = min((f for f in front if f.id != m.id), key=lambda f: sq_dist(m, f), default=None)
                if tgt is None:
                    continue
                sol = reach(m, tgt, m.ships - 1, obs)
                if sol:
                    extra.append([m.id, sol[1], m.ships - 1])
            roots.append(("forward_push", extra))
        return roots

    def agent(obs):
        me = obs["player"]; opp = 1 - me
        roots = candidate_roots(obs, me)
        def rollout_val(root):
            st = snapshot(obs)
            sim_step(st, {me: root, opp: _fast_rollout(st, opp)})
            for _ in range(ROLLOUT_DEPTH - 1):
                sim_step(st, {me: _fast_rollout(st, me), opp: _fast_rollout(st, opp)})
            return evaluate(st, me, PROD_W)
        base_root = roots[0][1]
        base_val = rollout_val(base_root)
        best_root, best_val = base_root, base_val
        for _nm, root in roots[1:]:
            v = rollout_val(root)
            if v > best_val + MARGIN:        # require a clear margin to deviate
                best_val, best_root = v, root
        return best_root
    agent.__name__ = "search_margin"
    return agent


search_margin = make_search_margin()
