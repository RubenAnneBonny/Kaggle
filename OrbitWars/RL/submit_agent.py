"""submit_agent.py — play orbit_wars with a trained OrbitNet checkpoint.

Loads weights once (module load) and exposes `agent(obs)` for kaggle.
Greedy action (argmax) for deterministic play. Falls back to net_roi_support
if anything goes wrong, so a bad checkpoint never forfeits.

The agent decodes with capture_size=True — the SAME sizing used in training,
PPO, and replay. (Without it, decode_action would fall back to the network's
raw fraction head, which undershoots captures and plays a weaker, different
agent than the one that was trained.)

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
_HOLD_PENALTY = 0.0  # subtract from HOLD logit at decision time; >0 = attack more


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
        tl = tl[0].clone()
        if _HOLD_PENALTY:
            tl[:, -1] -= _HOLD_PENALTY      # last col is HOLD; lower it -> attack more
        tgt = tl.argmax(-1).cpu().numpy()
        frac = fl[0].argmax(-1).cpu().numpy()
        # capture_size=True: size attacks to actually take the target (matches
        # training / PPO / replay). This is the decode the policy was trained under.
        return decode_action(enc, obs, player, tgt, frac, capture_size=True)
    except Exception:
        return net_roi_support(obs)


if __name__ == "__main__":
    import contextlib, io
    from kaggle_environments import make
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ppo.pt")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--hold-penalty", type=float, default=0.0, dest="hold_penalty",
                    help="subtract from HOLD logit; >0 makes the agent attack more")
    args = ap.parse_args()
    globals()["_HOLD_PENALTY"] = args.hold_penalty
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
