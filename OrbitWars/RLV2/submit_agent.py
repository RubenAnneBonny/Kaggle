"""submit_agent.py (v2) — competition entry point.

The game is played with 2 OR 4 players and the strategies differ, so we train a
SEPARATE model for each and dispatch on the inferred player count:

    distinct non-neutral owners in initial_planets  ==  number of players
    > 2  -> use the 4-player model      else  -> use the 2-player model

Decoding is GREEDY and matches training: argmax target (or HOLD), Beta-MEAN
fraction, the 5% floor, and the comet/planet-safe launch filter. Any failure
falls back to the scripted net_roi_support so the agent never crashes a game.

Set the two checkpoint paths below (or pass them on the CLI for local eval).
On Kaggle, submit this file together with: model.py, ow_env.py, ow_base.py,
search_agent.py, and the two .pt checkpoints.
"""
import os, numpy as np, torch
from ow_env import encode_state, decode_action, N_MAX, FRAC_FLOOR
from model import OrbitNet, build_edge, frac_dist
from ow_base import net_roi_support

HOLD = N_MAX
MODEL_2P_PATH = os.environ.get("OW_MODEL_2P", "ppo2_best.pt")
MODEL_4P_PATH = os.environ.get("OW_MODEL_4P", "ppo4_best.pt")

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_NETS = {}                      # player_count -> loaded OrbitNet (lazy)
_NPLAYERS = None               # cached inference for this episode


def _load(path):
    net = OrbitNet().to(_DEV)
    net.load_state_dict(torch.load(path, map_location=_DEV)["model"])
    net.eval()
    return net


def _get_net(num_players):
    key = 4 if num_players > 2 else 2
    if key not in _NETS:
        path = MODEL_4P_PATH if key == 4 else MODEL_2P_PATH
        _NETS[key] = _load(path)
    return _NETS[key]


def _infer_players(obs):
    owners = set()
    for p in obs.get("initial_planets", []) or obs.get("planets", []):
        if p[1] >= 0:
            owners.add(p[1])
    return len(owners) if owners else 2


def _greedy_moves(net, obs, player):
    enc = encode_state(obs, player)
    if enc["own_mask"].sum() == 0:
        return []
    nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(_DEV)
    nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(_DEV)
    am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(_DEV)
    with torch.no_grad():
        tl, fab, _ = net(nf, build_edge(nf), nm, am)
    tgt = tl[0].argmax(-1).cpu().numpy()
    frac = frac_dist(fab[0]).mean.cpu().numpy()
    return decode_action(enc, obs, player, tgt, frac, frac_floor=FRAC_FLOOR)


def agent(obs):
    """Kaggle single-arg agent. Dispatches 2p/4p, greedy decode, safe fallback."""
    global _NPLAYERS
    try:
        player = obs["player"]
        if _NPLAYERS is None:
            _NPLAYERS = _infer_players(obs)
        net = _get_net(_NPLAYERS)
        return _greedy_moves(net, obs, player)
    except Exception:
        try:
            return net_roi_support(obs) or []
        except Exception:
            return []


# --------------------------------------------------------------------------- #
# Local evaluation over the EVAL seed pool (never trained on)
# --------------------------------------------------------------------------- #
def _local_eval(players, model_path, games, opponent="teacher"):
    import contextlib, io
    from quiet_kaggle import make
    from seeds import eval_seed
    import ow_base
    global _NPLAYERS, _NETS
    _NETS = {(4 if players > 2 else 2): _load(model_path)}
    opp_fn = (ow_base.net_roi_support if opponent == "teacher"
              else ow_base.net_roi_aggressive if opponent == "aggressive"
              else getattr(ow_base, opponent))

    def me(obs):
        global _NPLAYERS
        _NPLAYERS = players
        return _greedy_moves(_NETS[4 if players > 2 else 2], obs, obs["player"])

    wins = 0
    for i in range(games):
        slot = i % players
        order = [opp_fn] * players
        order[slot] = me
        e = make("orbit_wars", configuration={"seed": eval_seed(i)}, debug=False)
        with contextlib.redirect_stderr(io.StringIO()):
            e.run(order)
        r = [s.reward if s.reward is not None else -1 for s in e.steps[-1]]
        wins += 1 if r[slot] == max(r) and r[slot] > 0 else 0
    print(f"[{players}p] {model_path} vs {opponent}: win_rate {wins/games:.3f} over {games} eval games")
    return wins / games


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=2, choices=[2, 4])
    ap.add_argument("--model", required=True, help="checkpoint to evaluate")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--opponent", default="teacher")
    a = ap.parse_args()
    _local_eval(a.players, a.model, a.games, a.opponent)
