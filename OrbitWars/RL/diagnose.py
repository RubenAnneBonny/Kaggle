"""diagnose.py — figure out WHY the trained agent loses.

Runs the agent on a few real mid-game states with NO exception swallowing,
prints any traceback, and dumps the decided moves next to what net_roi_support
would do on the same state. Run:  python diagnose.py --ckpt bc_best.pt
"""
import argparse, contextlib, io, numpy as np, torch, traceback
from kaggle_environments import make
from ow_base import net_roi_support
from ow_env import encode_state, decode_action, N_MAX
from model import OrbitNet, build_edge

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="bc_best.pt")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

net = OrbitNet().to(dev)
state = torch.load(args.ckpt, map_location=dev)
print("checkpoint keys:", list(state.keys()))
net.load_state_dict(state["model"])
net.eval()
print("loaded", args.ckpt, "on", dev)

# build a few real mid-game observations
e = make("orbit_wars", configuration={"seed": 0, "episodeSteps": 60}, debug=False)
with contextlib.redirect_stderr(io.StringIO()):
    e.run([net_roi_support, net_roi_support])

for step in (20, 40, 55):
    obs = dict(e.steps[step][0]["observation"]); obs["player"] = 0
    print("\n" + "=" * 60, "step", step)
    enc = encode_state(obs, 0)
    print("owned planets:", int(enc["own_mask"].sum()),
          "| legal targets total:", int(enc["attack_mask"].sum()))
    try:
        nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(dev)
        nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(dev)
        am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(dev)
        with torch.no_grad():
            tl, fl, val = net(nf, build_edge(nf), nm, am)
        tgt = tl[0].argmax(-1).cpu().numpy()
        frac = fl[0].argmax(-1).cpu().numpy()
        # how many owned planets choose HOLD vs an attack?
        own_idx = np.where(enc["own_mask"] > 0)[0]
        holds = sum(1 for i in own_idx if tgt[i] == N_MAX)
        attacks = len(own_idx) - holds
        print(f"network: {attacks} attacks, {holds} holds, value={val.item():.2f}")
        moves = decode_action(enc, obs, 0, tgt, frac)
        print("agent moves:", len(moves), moves[:5])
        base = net_roi_support(obs)
        print("net_roi_support moves:", len(base), base[:5])
    except Exception:
        traceback.print_exc()
