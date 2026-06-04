"""hold_bias.py — confirm the over-HOLD bias and find a fix without retraining.

The policy chooses, per planet, among N targets + a HOLD option. If HOLD's logit
is too high, the agent sits still and loses. We subtract a penalty from the HOLD
logit (equivalently: bias toward attacking) and measure, over many real states,
how often the agent attacks vs how often net_roi_support attacks. We want the
agent's attack rate to roughly match the teacher's.

Run: python hold_bias.py --ckpt bc_best.pt
"""
import argparse, contextlib, io, numpy as np, torch
from kaggle_environments import make
from ow_base import net_roi_support
from ow_env import encode_state, decode_action, N_MAX
from model import OrbitNet, build_edge

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="bc_best.pt")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

net = OrbitNet().to(dev)
net.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
net.eval()

# collect a few hundred real states
states = []
for seed in range(6):
    e = make("orbit_wars", configuration={"seed": seed}, debug=False)
    with contextlib.redirect_stderr(io.StringIO()):
        e.run([net_roi_support, net_roi_support])
    for stp in e.steps:
        obs = dict(stp[0]["observation"]); obs["player"] = 0
        if obs.get("planets"):
            states.append(obs)

def measure(hold_penalty):
    agent_atk = teacher_atk = owned_tot = matched = 0
    for obs in states:
        enc = encode_state(obs, 0)
        own_idx = np.where(enc["own_mask"] > 0)[0]
        if len(own_idx) == 0:
            continue
        nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(dev)
        nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(dev)
        am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(dev)
        with torch.no_grad():
            tl, fl, _ = net(nf, build_edge(nf), nm, am)
        tl = tl[0].clone()
        tl[:, N_MAX] -= hold_penalty          # lower HOLD logit -> attack more
        tgt = tl.argmax(-1).cpu().numpy()
        # teacher
        base = {m[0] for m in (net_roi_support(obs) or [])}
        for i in own_idx:
            owned_tot += 1
            a_atk = tgt[i] != N_MAX
            t_atk = int(enc["ids"][i]) in base
            agent_atk += a_atk
            teacher_atk += t_atk
            matched += (a_atk == t_atk)
    return agent_atk, teacher_atk, owned_tot, matched

print(f"states: {len(states)}")
print(f"{'penalty':>8} {'agent_atk%':>10} {'teacher_atk%':>13} {'act_match%':>11}")
for hp in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
    a, t, n, m = measure(hp)
    print(f"{hp:8.1f} {a/n*100:10.1f} {t/n*100:13.1f} {m/n*100:11.1f}")
