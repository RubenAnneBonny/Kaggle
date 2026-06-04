"""capture_check.py — is the agent failing to EXPAND because its attacks don't
capture? Plays the agent and the teacher each (vs nearest_planet) and reports,
per launched attack fleet, how often it results in gaining a planet, plus the
distribution of send-fractions chosen. Run: python capture_check.py --ckpt bc_best.pt
"""
import argparse, contextlib, io, numpy as np, torch
from kaggle_environments import make
from ow_base import net_roi_support, nearest_planet
from ow_env import encode_state, decode_action, N_MAX, FRACS
from model import OrbitNet, build_edge

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="bc_best.pt")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = OrbitNet().to(dev); net.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); net.eval()

frac_counts = np.zeros(len(FRACS), dtype=int)

def agent(obs):
    enc = encode_state(obs, obs["player"])
    if enc["own_mask"].sum() == 0:
        return []
    nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(dev)
    nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(dev)
    am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(dev)
    with torch.no_grad():
        tl, fl, _ = net(nf, build_edge(nf), nm, am)
    tgt = tl[0].argmax(-1).cpu().numpy()
    frac = fl[0].argmax(-1).cpu().numpy()
    for i in np.where(enc["own_mask"] > 0)[0]:
        if tgt[i] != N_MAX:
            frac_counts[int(frac[i])] += 1
    return decode_action(enc, obs, obs["player"], tgt, frac)

def peak_planets(steps, player):
    return max(sum(1 for p in s[0]["observation"]["planets"] if p[1] == player) for s in steps)

def planet_curve(steps, player):
    return [sum(1 for p in s[0]["observation"]["planets"] if p[1] == player) for s in steps]

print("agent vs nearest_planet — planet count over time + final result:")
for seed in range(5):
    e = make("orbit_wars", configuration={"seed": seed}, debug=True)
    with contextlib.redirect_stderr(io.StringIO()):
        e.run([agent, nearest_planet])
    cur = planet_curve(e.steps, 0)
    opp = planet_curve(e.steps, 1)
    n = len(cur)
    pts = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    traj = "  ".join(f"t{p}:{cur[p]}v{opp[p]}" for p in pts)
    r = [s.reward for s in e.steps[-1]]
    print(f"  seed {seed}: {traj}   final {r}  (len {n})")

print("\nteacher vs nearest_planet — same, for reference:")
for seed in range(5):
    e = make("orbit_wars", configuration={"seed": seed}, debug=True)
    with contextlib.redirect_stderr(io.StringIO()):
        e.run([net_roi_support, nearest_planet])
    cur = planet_curve(e.steps, 0); opp = planet_curve(e.steps, 1); n = len(cur)
    pts = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    traj = "  ".join(f"t{p}:{cur[p]}v{opp[p]}" for p in pts)
    r = [s.reward for s in e.steps[-1]]
    print(f"  seed {seed}: {traj}   final {r}  (len {n})")
