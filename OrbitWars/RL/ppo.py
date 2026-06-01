"""ppo.py — Stage 2: PPO self-play fine-tuning, warm-started from bc.pt.

Trains OrbitNet to beat its opponent on the fast engine-faithful simulator
(ow_env.OrbitEnv). Opponent = a frozen snapshot of the policy, refreshed
periodically (self-play). Reward = production-differential shaping + terminal
win/loss (from OrbitEnv).

This is where any improvement OVER net_roi_support comes from. BC alone only
imitates the teacher; PPO lets the policy discover better play.

Usage:
  python ppo.py --init bc.pt --iters 2000 --out ppo.pt
Validate the result with your 300-game rig against net_roi_support.
"""
import argparse, contextlib, io, numpy as np, torch, torch.nn.functional as F
from ow_env import OrbitEnv, encode_state, decode_action, N_MAX, N_FRAC
from model import OrbitNet, build_edge

HOLD = N_MAX


def policy_act(net, obs, player, dev, greedy=False):
    """Sample (or argmax) an action from the net for one observation.
    Returns engine moves + (logp, value, cached tensors for training)."""
    enc = encode_state(obs, player)
    nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(dev)
    nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(dev)
    am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(dev)
    edge = build_edge(nf)
    tgt_logits, frac_logits, value = net(nf, edge, nm, am)
    tgt_logits = tgt_logits[0]; frac_logits = frac_logits[0]   # (N,N+1),(N,N_FRAC)

    own = torch.tensor(enc["own_mask"], dtype=torch.bool, device=dev)
    td = torch.distributions.Categorical(logits=tgt_logits)
    fd = torch.distributions.Categorical(logits=frac_logits)
    if greedy:
        tgt = tgt_logits.argmax(-1); frac = frac_logits.argmax(-1)
    else:
        tgt = td.sample(); frac = fd.sample()
    # log-prob = sum over owned planets (target + frac-if-attacking)
    logp_t = td.log_prob(tgt)
    logp_f = fd.log_prob(frac)
    attacking = (tgt != HOLD) & own
    logp = (logp_t * own.float()).sum() + (logp_f * attacking.float()).sum()

    moves = decode_action(enc, obs, player,
                          tgt.detach().cpu().numpy(), frac.detach().cpu().numpy())
    cache = dict(nf=nf, nm=nm, am=am, edge=edge, tgt=tgt.detach(),
                 frac=frac.detach(), own=own)
    return moves, logp, value[0], cache


def recompute_logp_value(net, cache, dev):
    tgt_logits, frac_logits, value = net(cache["nf"], cache["edge"], cache["nm"], cache["am"])
    tgt_logits = tgt_logits[0]; frac_logits = frac_logits[0]
    own = cache["own"]
    td = torch.distributions.Categorical(logits=tgt_logits)
    fd = torch.distributions.Categorical(logits=frac_logits)
    logp_t = td.log_prob(cache["tgt"]); logp_f = fd.log_prob(cache["frac"])
    attacking = (cache["tgt"] != HOLD) & own
    logp = (logp_t * own.float()).sum() + (logp_f * attacking.float()).sum()
    ent = (td.entropy() * own.float()).sum() + (fd.entropy() * attacking.float()).sum()
    return logp, value[0], ent


def collect_episode(net, opp_net, env, seed, dev, gamma=0.99, lam=0.95):
    obs0, obs1 = env.reset(seed=seed)
    traj = []
    done = False
    while not done:
        mv0, logp, val, cache = policy_act(net, env.obs_for(0), 0, dev)
        with torch.no_grad():
            mv1, _, _, _ = policy_act(opp_net, env.obs_for(1), 1, dev)
        r, done = env.step(mv0, mv1)
        traj.append([logp, val, r, cache])
    # GAE
    adv, gae, nextv = [0.0] * len(traj), 0.0, 0.0
    for t in reversed(range(len(traj))):
        v = traj[t][1].item()
        delta = traj[t][2] + gamma * nextv - v
        gae = delta + gamma * lam * gae
        adv[t] = gae; nextv = v
    ret = [adv[t] + traj[t][1].item() for t in range(len(traj))]
    return traj, adv, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="bc.pt")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--episodes_per_iter", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--refresh", type=int, default=20, help="iters between opponent refresh")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--out", default="ppo.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = OrbitNet().to(dev)
    opp = OrbitNet().to(dev)
    if args.init:
        ck = torch.load(args.init, map_location=dev)
        net.load_state_dict(ck["model"]); print("warm-started from", args.init)
    opp.load_state_dict(net.state_dict())
    for p in opp.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = OrbitEnv(max_steps=args.max_steps)
    seed = 0
    for it in range(args.iters):
        # ---- collect ----
        batch = []
        ep_rewards = []
        for _ in range(args.episodes_per_iter):
            traj, adv, ret = collect_episode(net, opp, env, seed, dev)
            seed += 1
            ep_rewards.append(sum(t[2] for t in traj))
            a = np.array(adv); a = (a - a.mean()) / (a.std() + 1e-6)
            for t in range(len(traj)):
                batch.append((traj[t][3], traj[t][0].detach(), a[t], ret[t]))
        # ---- update ----
        for _ in range(args.epochs):
            np.random.shuffle(batch)
            for cache, old_logp, advt, rett in batch:
                logp, val, ent = recompute_logp_value(net, cache, dev)
                ratio = torch.exp(logp - old_logp)
                a_t = torch.tensor(advt, device=dev, dtype=torch.float32)
                s1 = ratio * a_t
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_t
                pol = -torch.min(s1, s2)
                vloss = F.mse_loss(val, torch.tensor(rett, device=dev, dtype=torch.float32))
                loss = pol + 0.5 * vloss - args.ent * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        if (it + 1) % args.refresh == 0:
            opp.load_state_dict(net.state_dict())
        if it % 10 == 0:
            print(f"iter {it:4d}  mean_ep_reward {np.mean(ep_rewards):+.3f}")
    torch.save({"model": net.state_dict()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
