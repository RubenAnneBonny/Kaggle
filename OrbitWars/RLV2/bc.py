"""bc.py (v2) — Stage 1 behavioral cloning of the AGGRESSIVE teacher.

Why aggressive: with a continuous fraction head, cloning the minimal-capture
teacher (net_roi_support) made the policy send too few ships. We instead clone
`net_roi_aggressive`, which commits a larger (still clear-path, still affordable)
fraction per attack, so BC starts the fraction head HIGH. PPO then tunes it down.

Per owned source we recover the teacher's decision:
  * target index (or HOLD), by matching the teacher's launch angle to each
    candidate's reach()-predicted intercept angle (falls back to direction),
  * a CONTINUOUS send-fraction = ships / available.

Losses: attack-weighted cross-entropy on the target (HOLD dominates, so we
upweight real attacks) + Beta negative-log-likelihood on the fraction (only on
sources the teacher actually launched from).

Train/eval separation is enforced by seeds.train_seed — you cannot harvest an
eval seed. Use --players 2 or 4 to build the two separate datasets/models.

Usage:
  python bc.py --players 2 --games 800 --epochs 120 --bs 256 --out bc2.pt
  python bc.py --players 4 --games 800 --epochs 120 --bs 256 --out bc4.pt
"""
import argparse, math, contextlib, io, numpy as np, torch, torch.nn.functional as F
from kaggle_environments import make
from ow_base import net_roi_aggressive, net_roi_support, reach, predict_all_fleet_hits
from ow_env import encode_state, N_MAX
from model import OrbitNet, build_edge, frac_dist
from seeds import train_seed

HOLD = N_MAX
TEACHER = net_roi_aggressive          # BC clones THIS


def _as_planet(p):
    from ow_env import _as_planet as f
    return f(p)


def teacher_decisions(obs, player):
    """Return (enc, target_label, frac_label) per owned source.
       target_label[i] in [0..N_MAX] (N_MAX == HOLD)
       frac_label[i]   in (0,1] for launched sources, else -1 (no frac loss)."""
    enc = encode_state(obs, player)
    planets = {p[0]: p for p in obs["planets"]}
    inc_enemy = {}
    for owner, ships, pid, tick in predict_all_fleet_hits(obs):
        if owner != player:
            inc_enemy[pid] = inc_enemy.get(pid, 0) + ships

    moves = TEACHER(obs) or []
    move_by_src = {}
    for fid, angle, ships in moves:
        move_by_src.setdefault(fid, []).append((angle, ships))

    tgt_label = np.full(N_MAX, HOLD, np.int64)
    frac_label = np.full(N_MAX, -1.0, np.float32)
    for i in range(N_MAX):
        if enc["own_mask"][i] == 0.0:
            continue
        sid = int(enc["ids"][i])
        src = planets.get(sid)
        ms = move_by_src.get(sid)
        if not ms or src is None:
            continue
        angle, ships = ms[0]
        # recover target index: cheap angle-to-candidate matching. (We tried
        # reach()-per-candidate for exactness but it makes harvest far too slow;
        # BC is only a warm start, so direction matching with a tolerance is the
        # right trade. PPO fixes any residual mislabeling.)
        best_j, best_err = HOLD, 0.30          # radians tolerance
        for j in range(N_MAX):
            if enc["attack_mask"][i, j] == 0.0:
                continue
            dst = planets.get(int(enc["ids"][j]))
            if dst is None:
                continue
            cand_ang = math.atan2(dst[3] - src[3], dst[2] - src[2])
            err = abs(math.atan2(math.sin(angle - cand_ang), math.cos(angle - cand_ang)))
            if err < best_err:
                best_err, best_j = err, j
        if best_j == HOLD:
            continue
        tgt_label[i] = best_j
        reserve = max(0, inc_enemy.get(sid, 0)) + 1
        avail = max(1, src[5] - reserve)
        frac_label[i] = float(min(1.0, max(1e-3, ships / avail)))
    return enc, tgt_label, frac_label


def harvest(games, players, seed_offset=0):
    NF, NM, OM, AM, TL, FL = [], [], [], [], [], []
    for g in range(games):
        seed = train_seed(seed_offset + g)
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=players)
        agents = [TEACHER] * players
        with contextlib.redirect_stderr(io.StringIO()):
            steps = env.run(agents)
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
    ap.add_argument("--players", type=int, default=2, choices=[2, 4])
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="bc.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed-offset", type=int, default=0, dest="seed_offset",
                    help="offset into the TRAIN seed pool (still can't touch eval seeds)")
    ap.add_argument("--attack-weight", type=float, default=5.0, dest="attack_weight")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev} | players {args.players} | harvesting {args.games} games "
          f"(train seeds {train_seed(args.seed_offset)}..{train_seed(args.seed_offset+args.games-1)})...")
    NF, NM, OM, AM, TL, FL = harvest(args.games, args.players, args.seed_offset)
    print("dataset states:", NF.shape[0])

    NF = torch.tensor(NF); NM = torch.tensor(NM); OM = torch.tensor(OM)
    AM = torch.tensor(AM); TL = torch.tensor(TL); FL = torch.tensor(FL)
    n = NF.shape[0]; cut = int(0.9 * n)
    perm = torch.randperm(n); tr, va = perm[:cut], perm[cut:]

    net = OrbitNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        net.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        print("resumed from", args.resume)

    def batch_loss(idx):
        nf = NF[idx].to(dev); nm = NM[idx].to(dev); om = OM[idx].to(dev)
        am = AM[idx].to(dev); tl = TL[idx].to(dev); fl = FL[idx].to(dev)
        edge = build_edge(nf)
        tgt_logits, frac_ab, _ = net(nf, edge, nm, am)
        B, N, _ = tgt_logits.shape
        own = om.bool().view(-1)
        tlog = tgt_logits.view(B * N, N + 1)[own]
        tlab = tl.view(B * N)[own]
        ce_t = F.cross_entropy(tlog, tlab, reduction="none")
        wts = torch.where(tlab != HOLD,
                          torch.tensor(args.attack_weight, device=dev),
                          torch.tensor(1.0, device=dev))
        loss_t = (ce_t * wts).sum() / wts.sum()
        # continuous fraction: Beta NLL on launched sources only
        fab = frac_ab.view(B * N, 2)[own]
        flab = fl.view(B * N)[own]
        m = flab >= 0
        if m.any():
            dist = frac_dist(fab[m])
            target = flab[m].clamp(1e-3, 1 - 1e-3)
            loss_f = -dist.log_prob(target).mean()
        else:
            loss_f = torch.tensor(0.0, device=dev)
        atk = tlab != HOLD
        acc = (tlog[atk].argmax(-1) == tlab[atk]).float().mean() if atk.any() else torch.tensor(0.0)
        # also report mean |frac_pred - frac_label| on launched sources
        if m.any():
            ferr = (frac_dist(fab[m]).mean - flab[m]).abs().mean()
        else:
            ferr = torch.tensor(0.0)
        return loss_t + 0.5 * loss_f, acc, ferr

    best_acc, best_path = -1.0, (args.out.replace(".pt", "_best.pt") if args.out.endswith(".pt") else args.out + "_best")
    for ep in range(args.epochs):
        net.train()
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), args.bs):
            loss, _, _ = batch_loss(p[i:i + args.bs])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl, acc, ferr = batch_loss(va)
        acc = acc.item(); flag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "players": args.players}, best_path)
            flag = "  <- new best, saved"
        print(f"epoch {ep:3d}  val_loss {vl.item():.3f}  target_acc {acc:.3f}  "
              f"frac_MAE {ferr.item():.3f}{flag}")

    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "players": args.players}, args.out)
    print(f"saved last -> {args.out} | best (acc={best_acc:.3f}) -> {best_path}")


if __name__ == "__main__":
    main()
