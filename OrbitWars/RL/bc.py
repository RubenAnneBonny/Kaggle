"""bc.py — Stage 1: behavioral cloning of net_roi_support into OrbitNet.

Harvests, per owned planet per turn, the teacher's decision:
  target index (or HOLD) + send-fraction bucket.
Then trains the network with cross-entropy (target) + cross-entropy (fraction,
only on turns where the teacher attacked). This is the CORRECT per-source
framing (one categorical choice per source), unlike independent per-pair labels.

Usage:
  python bc.py --games 300 --epochs 15 --out bc.pt
Outputs a checkpoint you then warm-start PPO from.
"""
import argparse, math, contextlib, io, numpy as np, torch, torch.nn.functional as F
from kaggle_environments import make
from ow_base import net_roi_support, parse_obs, sq_dist, predict_all_fleet_hits
from ow_env import encode_state, N_MAX, FRACS, N_FRAC
from model import OrbitNet, build_edge

HOLD = N_MAX  # index of the HOLD action in the (N_MAX+1)-way target head


def teacher_decisions(obs, player):
    """Return per-owned-source (target_index or HOLD, frac_bucket).
    Recovers the teacher's intended target by aligning its launch angle with the
    best-matching candidate, and its fraction from ships/available."""
    enc = encode_state(obs, player)
    planets = {p[0]: p for p in obs["planets"]}
    comet_ids = set(obs.get("comet_planet_ids", []))
    inc_enemy = {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner != player:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships

    moves = net_roi_support(obs) or []
    move_by_src = {}
    for fid, angle, ships in moves:
        move_by_src.setdefault(fid, []).append((angle, ships))

    tgt_label = np.full(N_MAX, HOLD, np.int64)
    frac_label = np.full(N_MAX, -1, np.int64)   # -1 = no frac loss (HOLD)
    for i in range(N_MAX):
        if enc["own_mask"][i] == 0.0:
            continue
        sid = int(enc["ids"][i])
        src = planets.get(sid)
        ms = move_by_src.get(sid)
        if not ms or src is None:
            continue
        angle, ships = ms[0]                    # teacher sends one attack/planet
        # best-aligned candidate target index
        best_j, best_dot = HOLD, 0.9
        for j in range(N_MAX):
            if enc["attack_mask"][i, j] == 0.0:
                continue
            dst = planets.get(int(enc["ids"][j]))
            dx, dy = dst[2] - src[2], dst[3] - src[3]
            nrm = math.hypot(dx, dy)
            if nrm < 1e-6:
                continue
            dot = (math.cos(angle) * dx + math.sin(angle) * dy) / nrm
            if dot > best_dot:
                best_dot, best_j = dot, j
        if best_j == HOLD:
            continue
        tgt_label[i] = best_j
        reserve = max(0, inc_enemy.get(sid, 0)) + 1
        avail = max(1, src[5] - reserve)
        f = ships / avail
        frac_label[i] = int(np.argmin(np.abs(FRACS - f)))
    return enc, tgt_label, frac_label


def harvest(games, opponent=net_roi_support, seed_offset=0):
    NF, NM, OM, AM, TL, FL = [], [], [], [], [], []
    for seed in range(seed_offset, seed_offset + games):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=2)
        with contextlib.redirect_stderr(io.StringIO()):
            steps = env.run([net_roi_support, opponent])
        for stp in steps:
            obs = dict(stp[0]["observation"]); obs["player"] = 0
            if not obs.get("planets"):
                continue
            enc, tl, fl = teacher_decisions(obs, 0)
            if enc["own_mask"].sum() == 0:
                continue
            NF.append(enc["node_feats"]); NM.append(enc["node_mask"])
            OM.append(enc["own_mask"]); AM.append(enc["attack_mask"])
            TL.append(tl); FL.append(fl)
    return (np.stack(NF), np.stack(NM), np.stack(OM),
            np.stack(AM), np.stack(TL), np.stack(FL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="bc.pt")
    ap.add_argument("--resume", default="", help="path to a checkpoint to continue training from")
    ap.add_argument("--seed-offset", type=int, default=0, dest="seed_offset",
                    help="first game seed to harvest; bump between runs for fresh data")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, "| harvesting", args.games, "games "
          f"(seeds {args.seed_offset}..{args.seed_offset + args.games - 1})...")
    NF, NM, OM, AM, TL, FL = harvest(args.games, seed_offset=args.seed_offset)
    print("dataset states:", NF.shape[0])

    NF = torch.tensor(NF); NM = torch.tensor(NM); OM = torch.tensor(OM)
    AM = torch.tensor(AM); TL = torch.tensor(TL); FL = torch.tensor(FL)
    n = NF.shape[0]; cut = int(0.9 * n)
    perm = torch.randperm(n)
    tr, va = perm[:cut], perm[cut:]

    net = OrbitNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        net.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        print("resumed from", args.resume, "(continuing training, not starting fresh)")

    def batch_loss(idx, train=True):
        nf = NF[idx].to(dev); nm = NM[idx].to(dev); om = OM[idx].to(dev)
        am = AM[idx].to(dev); tl = TL[idx].to(dev); fl = FL[idx].to(dev)
        edge = build_edge(nf)
        tgt_logits, frac_logits, _ = net(nf, edge, nm, am)
        B, N, _ = tgt_logits.shape
        own = om.bool().view(-1)
        tlog = tgt_logits.view(B * N, N + 1)[own]
        tlab = tl.view(B * N)[own]
        loss_t = F.cross_entropy(tlog, tlab)
        # fraction loss only where teacher attacked (label >= 0)
        flog = frac_logits.view(B * N, N_FRAC)[own]
        flab = fl.view(B * N)[own]
        m = flab >= 0
        loss_f = F.cross_entropy(flog[m], flab[m]) if m.any() else torch.tensor(0.0, device=dev)
        # accuracy of target choice on attacked sources
        atk = tlab != N_MAX
        acc = (tlog[atk].argmax(-1) == tlab[atk]).float().mean() if atk.any() else torch.tensor(0.0)
        return loss_t + 0.5 * loss_f, acc

    for ep in range(args.epochs):
        net.train()
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), args.bs):
            idx = p[i:i + args.bs]
            loss, _ = batch_loss(idx, True)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl, va_acc = batch_loss(va, False)
        print(f"epoch {ep:2d}  val_loss {vl.item():.3f}  val_target_acc {va_acc.item():.3f}")

    torch.save({"model": net.state_dict(), "opt": opt.state_dict()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
