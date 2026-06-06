"""replay.py (v2) — render a game for inspection.

Plays the v2 greedy agent (comet-safe, continuous fraction) against an opponent
on an EVAL-pool seed, writes a seed/result-stamped HTML you can open in a
browser, and prints a short per-step trace (ship totals + comet count).

Usage:
  python replay.py --players 2 --model ppo2_best.pt --opponent teacher --seed-idx 0
  python replay.py --players 4 --model ppo4_best.pt --slot 0 --seed-idx 3
"""
import argparse, contextlib, io, numpy as np, torch
from quiet_kaggle import make          # suppresses the OpenSpiel import banner
from model import OrbitNet, build_edge, frac_dist
from ow_env import encode_state, decode_action, FRAC_FLOOR
from seeds import eval_seed
import ow_base

_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_net(path):
    n = OrbitNet().to(_DEV)
    n.load_state_dict(torch.load(path, map_location=_DEV)["model"])
    n.eval()
    return n


def greedy_agent(net):
    def a(obs):
        try:
            enc = encode_state(obs, obs["player"])
            if enc["own_mask"].sum() == 0:
                return []
            nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(_DEV)
            nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(_DEV)
            am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(_DEV)
            with torch.no_grad():
                tl, fab, _ = net(nf, build_edge(nf), nm, am)
            tgt = tl[0].argmax(-1).cpu().numpy()
            frac = frac_dist(fab[0]).mean.cpu().numpy()
            return decode_action(enc, obs, obs["player"], tgt, frac, frac_floor=FRAC_FLOOR)
        except Exception:
            return ow_base.net_roi_support(obs) or []
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=2, choices=[2, 4])
    ap.add_argument("--model", required=True)
    ap.add_argument("--opponent", default="teacher")
    ap.add_argument("--slot", type=int, default=0, help="which slot the v2 agent plays")
    ap.add_argument("--seed-idx", type=int, default=0, dest="seed_idx",
                    help="index into the EVAL seed pool")
    ap.add_argument("--out", default="")
    ap.add_argument("--trace-every", type=int, default=50, dest="trace_every")
    a = ap.parse_args()

    net = load_net(a.model)
    me = greedy_agent(net)
    opp = (ow_base.net_roi_support if a.opponent == "teacher"
           else ow_base.net_roi_aggressive if a.opponent == "aggressive"
           else getattr(ow_base, a.opponent))
    order = [opp] * a.players
    order[a.slot] = me

    seed = eval_seed(a.seed_idx)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    with contextlib.redirect_stderr(io.StringIO()):
        steps = env.run(order)

    for t, stp in enumerate(steps):
        if t % a.trace_every and t != len(steps) - 1:
            continue
        o = stp[0]["observation"]
        scores = [0] * a.players
        for p in o.get("planets", []):
            if p[1] >= 0:
                scores[p[1]] += p[5]
        for f in o.get("fleets", []):
            scores[f[1]] += f[6]
        nc = len(o.get("comet_planet_ids", []) or [])
        print(f"step {t:3d}  scores {scores}  comets {nc}")

    r = [s.reward if s.reward is not None else -1 for s in env.steps[-1]]
    won = r[a.slot] == max(r) and r[a.slot] > 0
    result = "WIN" if won else "LOSS"
    print(f"\nseed {seed} | v2 agent (slot {a.slot}, {a.players}p) vs {a.opponent}: "
          f"{result}  rewards {r}")

    out = a.out or f"replay_{a.players}p_seed{seed}_{result}.html"
    html = env.render(mode="html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", out)


if __name__ == "__main__":
    main()
