"""replay.py — play ONE game (trained agent vs an opponent) and write an HTML
replay you can open in a browser to watch, plus a turn-by-turn text trace of
planet/ship counts so you can see exactly how the game is won or lost.

Usage:
  python replay.py --ckpt bc_best.pt --opponent net_roi_support --seed 0
  python replay.py --ckpt bc_best.pt --opponent nearest_planet --seed 3 --side 1

--side 0 (default) puts the trained agent as player 0; --side 1 swaps it.
Outputs: replay.html (open in browser) and prints a text trace + final result.
"""
import argparse, contextlib, io, numpy as np, torch
from kaggle_environments import make
import ow_base
from ow_base import net_roi_support
from ow_env import encode_state, decode_action, N_MAX
from model import OrbitNet, build_edge

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="bc_best.pt")
ap.add_argument("--opponent", default="net_roi_support",
                help="name of any ow_base agent, e.g. net_roi_support, nearest_planet")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--side", type=int, default=0, choices=[0, 1],
                help="which player slot the trained agent occupies")
ap.add_argument("--out", default="replay.html")
ap.add_argument("--hold-penalty", type=float, default=0.0, dest="hold_penalty")
ap.add_argument("--no-capture-size", action="store_true",
                help="use the network's fraction head instead of sizing attacks to capture")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = OrbitNet().to(dev)
net.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
net.eval()

def make_agent():
    err_count = {"n": 0}
    def agent(obs):
        try:
            enc = encode_state(obs, obs["player"])
            if enc["own_mask"].sum() == 0:
                return []
            nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(dev)
            nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(dev)
            am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(dev)
            with torch.no_grad():
                tl, fl, _ = net(nf, build_edge(nf), nm, am)
            tl = tl[0].clone()
            if args.hold_penalty:
                tl[:, -1] -= args.hold_penalty
            tgt = tl.argmax(-1).cpu().numpy()
            frac = fl[0].argmax(-1).cpu().numpy()
            return decode_action(enc, obs, obs["player"], tgt, frac,
                                 capture_size=not args.no_capture_size)
        except Exception as e:
            err_count["n"] += 1
            if err_count["n"] <= 3:
                import traceback; traceback.print_exc()
            return []
    agent._errs = err_count
    return agent

trained = make_agent()
opp_fn = getattr(ow_base, args.opponent)

# arrange players by --side
if args.side == 0:
    players = [trained, opp_fn]
    me_idx, me_name, opp_name = 0, f"AGENT({args.ckpt})", args.opponent
else:
    players = [opp_fn, trained]
    me_idx, me_name, opp_name = 1, f"AGENT({args.ckpt})", args.opponent

env = make("orbit_wars", configuration={"seed": args.seed}, debug=True)
with contextlib.redirect_stderr(io.StringIO()):
    env.run(players)

# ---- text trace ----
def counts(obs, pl):
    p = sum(1 for q in obs["planets"] if q[1] == pl)
    s = sum(q[5] for q in obs["planets"] if q[1] == pl)
    s += sum(f[6] for f in obs["fleets"] if f[1] == pl)
    return p, s

steps = env.steps
# In the rendered video, player 0 = ORANGE, player 1 = BLUE.
agent_color = "ORANGE" if me_idx == 0 else "BLUE"
opp_color = "BLUE" if me_idx == 0 else "ORANGE"
print(f"\nreplay: AGENT ({agent_color}, slot {me_idx}) vs {opp_name} ({opp_color}), seed {args.seed}")
print(f"  -> in the video, watch the {agent_color} planets; that's the trained agent.")
print("  step :  agent(planets, ships)   opp(planets, ships)")
n = len(steps)
# print every step so any video frame ('Step: N') maps to exactly one row
for t in range(n):
    obs = steps[t][0]["observation"]
    ap_, as_ = counts(obs, me_idx)
    op_, os_ = counts(obs, 1 - me_idx)
    print(f"  {t:4d} :  agent({ap_:2d}p,{as_:5d}s)    opp({op_:2d}p,{os_:5d}s)")

# peak + full per-step agent planet count, so a transient can't be missed
agent_curve = [counts(steps[t][0]["observation"], me_idx)[0] for t in range(n)]
print(f"  agent planet count: min {min(agent_curve)}, peak {max(agent_curve)}, final {agent_curve[-1]}")
print(f"  agent decision errors during game: {trained._errs['n']}"
      + ("  <-- attacks/decisions were crashing; that's the real bug" if trained._errs['n'] else ""))
r = [s.reward for s in steps[-1]]
won = r[me_idx] is not None and r[me_idx] > r[1 - me_idx]
print(f"  final rewards {r} -> agent {'WON' if won else 'LOST/DREW'} (game length {n})")

# ---- HTML replay ----
# Name the file by ckpt+opponent+seed so the video on screen is unambiguously the
# same game as this text trace (no mixing up runs).
import os, re
ck = re.sub(r"[^A-Za-z0-9]+", "", os.path.splitext(os.path.basename(args.ckpt))[0])
result = "WIN" if won else "LOSS"
out_name = args.out
if out_name == "replay.html":   # auto-name unless user overrode --out
    out_name = f"replay_{ck}_vs_{args.opponent}_seed{args.seed}_{result}.html"
try:
    html = env.render(mode="html", width=720, height=560)
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nwrote {out_name}")
    print(f"  ^ this video IS the game traced above: seed {args.seed}, "
          f"agent in slot {me_idx}, result {result}.")
    print("  The 'Step:' counter shown in the video matches the 'step' column above.")
except Exception as e:
    print("could not render HTML:", e)
