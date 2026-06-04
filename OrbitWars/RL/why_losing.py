"""why_losing.py — the agent makes correct attack/hold + target choices (94%),
yet loses ~97%. Two suspects: (A) wrong SEND-FRACTION (right target, wrong ship
count -> failed attacks), (B) distribution shift (errors compound on states the
teacher never visits). This tests both.

Run: python why_losing.py --ckpt bc_best.pt
"""
import argparse, contextlib, io, numpy as np, torch
from kaggle_environments import make
from ow_base import net_roi_support, parse_obs
from ow_env import encode_state, decode_action, N_MAX, FRACS, N_FRAC
from model import OrbitNet, build_edge

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="bc_best.pt")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = OrbitNet().to(dev); net.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); net.eval()

# ---- import the teacher's fraction labels exactly as bc.py computes them ----
import importlib.util
spec = importlib.util.spec_from_file_location("bcmod", "bc.py")
# bc.py imports torch at top; that's fine here
bcmod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(bcmod)
    teacher_decisions = bcmod.teacher_decisions
except Exception as e:
    print("could not import bc.teacher_decisions:", e); teacher_decisions = None

def net_decode(obs):
    enc = encode_state(obs, obs["player"])
    if enc["own_mask"].sum() == 0:
        return enc, None, None
    nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(dev)
    nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(dev)
    am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(dev)
    with torch.no_grad():
        tl, fl, _ = net(nf, build_edge(nf), nm, am)
    return enc, tl[0].argmax(-1).cpu().numpy(), fl[0].argmax(-1).cpu().numpy()

# ===== Test A: fraction-head accuracy on teacher states =====
if teacher_decisions is not None:
    states = []
    for seed in range(6):
        e = make("orbit_wars", configuration={"seed": seed}, debug=False)
        with contextlib.redirect_stderr(io.StringIO()):
            e.run([net_roi_support, net_roi_support])
        for stp in e.steps:
            o = dict(stp[0]["observation"]); o["player"] = 0
            if o.get("planets"):
                states.append(o)
    frac_tot = frac_hit = 0
    for o in states:
        enc, ntgt, nfrac = net_decode(o)
        if ntgt is None:
            continue
        _, tlab, flab = teacher_decisions(o, 0)
        for i in np.where(enc["own_mask"] > 0)[0]:
            if flab[i] >= 0:                      # teacher attacked from i
                frac_tot += 1
                frac_hit += (nfrac[i] == flab[i])
    print(f"[A] fraction-head accuracy on teacher attacks: "
          f"{frac_hit}/{frac_tot} = {frac_hit/max(frac_tot,1)*100:.1f}%  "
          f"(buckets={list(FRACS)})")

# ===== Test B: real self-play vs teacher — where does it collapse? =====
def agent(obs):
    enc, ntgt, nfrac = net_decode(obs)
    if ntgt is None:
        return []
    return decode_action(enc, obs, obs["player"], ntgt, nfrac)

for seed in [0, 1]:
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    with contextlib.redirect_stderr(io.StringIO()):
        env.run([agent, net_roi_support])
    # track ship/planet totals over time for each side
    print(f"\n[B] seed {seed}: trajectory (my_planets, my_ships | opp_planets, opp_ships)")
    steps = env.steps
    for t in range(0, len(steps), max(1, len(steps)//10)):
        pl = steps[t][0]["observation"]["planets"]
        mp = sum(1 for p in pl if p[1] == 0); ms = sum(p[5] for p in pl if p[1] == 0)
        op = sum(1 for p in pl if p[1] == 1); os_ = sum(p[5] for p in pl if p[1] == 1)
        print(f"   t={t:3d}  me({mp:2d}p,{ms:4d}s)  opp({op:2d}p,{os_:4d}s)")
    r = [s.reward for s in steps[-1]]
    print(f"   final reward {r}")
