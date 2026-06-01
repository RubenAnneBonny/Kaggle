"""submit_agent.py — play orbit_wars with a trained OrbitNet checkpoint.

Loads weights once (module load) and exposes `agent(obs)` for kaggle.
Greedy action (argmax) for deterministic play. Falls back to net_roi_support
if anything goes wrong, so a bad checkpoint never forfeits.

Local eval against net_roi_support:
  python submit_agent.py --ckpt ppo.pt --games 50
"""
import argparse, numpy as np, torch
from ow_env import encode_state, decode_action
from model import OrbitNet, build_edge
from ow_base import net_roi_support

_NET = None
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_CKPT = "ppo.pt"   # change or set via load_ckpt()


def load_ckpt(path):
    global _NET, _CKPT
    _CKPT = path
    _NET = OrbitNet().to(_DEV)
    _NET.load_state_dict(torch.load(path, map_location=_DEV)["model"])
    _NET.eval()
    return _NET


def agent(obs):
    global _NET
    try:
        if _NET is None:
            load_ckpt(_CKPT)
        player = obs["player"]
        enc = encode_state(obs, player)
        if enc["own_mask"].sum() == 0:
            return []
        nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(_DEV)
        nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(_DEV)
        am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(_DEV)
        with torch.no_grad():
            tl, fl, _ = _NET(nf, build_edge(nf), nm, am)
        tgt = tl[0].argmax(-1).cpu().numpy()
        frac = fl[0].argmax(-1).cpu().numpy()
        return decode_action(enc, obs, player, tgt, frac)
    except Exception:
        return net_roi_support(obs)


if __name__ == "__main__":
    import contextlib, io
    from kaggle_environments import make
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ppo.pt")
    ap.add_argument("--games", type=int, default=50)
    args = ap.parse_args()
    load_ckpt(args.ckpt)
    wins = 0
    for i in range(args.games):
        order = [agent, net_roi_support] if i % 2 == 0 else [net_roi_support, agent]
        ai = 0 if i % 2 == 0 else 1
        env = make("orbit_wars", configuration={"seed": i}, debug=False)
        with contextlib.redirect_stderr(io.StringIO()):
            env.run(order)
        r = [s.reward if s.reward is not None else 0 for s in env.steps[-1]]
        wins += r[ai] > r[1 - ai]
    print(f"{args.ckpt} vs net_roi_support: {wins}/{args.games} = {wins/args.games*100:.1f}%")
