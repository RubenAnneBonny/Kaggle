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

------------------------------------------------------------------------------
STABILITY FIXES (the reason an old run collapsed from ~0.3 win-rate to ~0.0
after a single iteration). See "Hard-won lessons" #9-#11 in README_rl.md.

  9.  MINIBATCHED UPDATES. The old loop called opt.step() ONCE PER TIMESTEP
      (~7000 steps/iter on 12 games), so the policy drifted miles from the
      data-collection policy *within* one iteration; ratios hit 700x the clip
      band at iter 0. Now we accumulate gradients over a minibatch and step
      once (~tens of steps/iter), which is what keeps standard PPO stable.

  10. APPROX-KL EARLY STOP. If the policy has already moved past --target-kl
      after an epoch, we stop updating on this batch instead of grinding the
      remaining epochs into divergence.

  11. FRACTION CREDIT-ASSIGNMENT. decode_action(capture_size=True) IGNORES the
      fraction head for enemy/neutral captures (it sizes attacks with
      how_many_send). So the sampled frac is a no-op for those moves and its
      log-prob must NOT enter the ratio. We only count frac log-prob where the
      fraction actually drove the move: friendly (own->own) ferries.
------------------------------------------------------------------------------
"""
import argparse, contextlib, io, numpy as np, torch, torch.nn.functional as F
from ow_env import OrbitEnv, encode_state, decode_action, N_MAX, N_FRAC
from model import OrbitNet, build_edge

HOLD = N_MAX


def _frac_active_mask(enc, tgt_np):
    """Per owned source: 1.0 iff its chosen target is a FRIENDLY planet, i.e.
    the case where decode_action actually uses the fraction head (a ferry).
    For enemy/neutral captures and HOLD the fraction is ignored, so its
    log-prob must be excluded from the PPO objective."""
    own = enc["own_mask"]
    am = enc["attack_mask"]
    out = np.zeros(N_MAX, np.float32)
    for i in range(N_MAX):
        if own[i] == 0.0:
            continue
        j = int(tgt_np[i])
        if j == HOLD or j < 0 or j >= N_MAX:
            continue
        if am[i, j] == 0.0:
            continue
        if own[j] == 1.0:            # friendly ferry -> fraction is used
            out[i] = 1.0
    return out


def policy_act(net, obs, player, dev, greedy=False):
    """Sample (or argmax) an action from the net for one observation.
    Returns engine moves + (logp, value, cached tensors for training).

    Note: the forward runs under no_grad — collection only needs the numbers
    (moves, value scalar, old log-probs). All gradients come from
    recompute_logp_value during the update, so tracking grad here only wasted
    memory (graphs were kept alive across a whole 150-step episode)."""
    enc = encode_state(obs, player)
    nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(dev)
    nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(dev)
    am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(dev)
    edge = build_edge(nf)
    with torch.no_grad():
        tgt_logits, frac_logits, value = net(nf, edge, nm, am)
    tgt_logits = tgt_logits[0]; frac_logits = frac_logits[0]   # (N,N+1),(N,N_FRAC)

    own = torch.tensor(enc["own_mask"], dtype=torch.bool, device=dev)
    td = torch.distributions.Categorical(logits=tgt_logits)
    fd = torch.distributions.Categorical(logits=frac_logits)
    if greedy:
        tgt = tgt_logits.argmax(-1); frac = frac_logits.argmax(-1)
    else:
        tgt = td.sample(); frac = fd.sample()

    tgt_np = tgt.detach().cpu().numpy()
    frac_np = frac.detach().cpu().numpy()
    # which planets' FRACTION actually influences the executed move (fix #11)
    frac_active = torch.tensor(_frac_active_mask(enc, tgt_np),
                               dtype=torch.bool, device=dev)

    # per-planet log-probs (NOT summed) — PPO must clip each component separately,
    # else the joint ratio exp(sum of N diffs) explodes and one update destroys the policy.
    logp_t = td.log_prob(tgt)
    logp_f = fd.log_prob(frac)
    # target log-prob counts for every owned planet (HOLD is a real choice);
    # fraction log-prob counts ONLY where the fraction was actually used.
    logp_per = logp_t * own.float() + logp_f * frac_active.float()
    logp = logp_per.sum()  # only used for logging/return compat

    moves = decode_action(enc, obs, player, tgt_np, frac_np, capture_size=True)
    cache = dict(nf=nf, nm=nm, am=am, edge=edge, tgt=tgt.detach(),
                 frac=frac.detach(), own=own, frac_active=frac_active,
                 old_logp_per=logp_per.detach())
    return moves, logp, value[0], cache


def recompute_logp_value(net, cache, dev):
    tgt_logits, frac_logits, value = net(cache["nf"], cache["edge"], cache["nm"], cache["am"])
    tgt_logits = tgt_logits[0]; frac_logits = frac_logits[0]
    own = cache["own"]
    frac_active = cache["frac_active"]
    td = torch.distributions.Categorical(logits=tgt_logits)
    fd = torch.distributions.Categorical(logits=frac_logits)
    logp_t = td.log_prob(cache["tgt"]); logp_f = fd.log_prob(cache["frac"])
    logp_per = logp_t * own.float() + logp_f * frac_active.float()   # per-planet
    active = own.float()                                             # which planets count
    ent = (td.entropy() * own.float()).sum() + (fd.entropy() * frac_active.float()).sum()
    return logp_per, active, value[0], ent


def collect_episode(net, opp_net, env, seed, dev, gamma=0.99, lam=0.95, opp_fn=None):
    obs0, obs1 = env.reset(seed=seed)
    traj = []
    done = False
    while not done:
        mv0, logp, val, cache = policy_act(net, env.obs_for(0), 0, dev)
        if opp_fn is not None:
            mv1 = opp_fn(env.obs_for(1)) or []
        else:
            mv1, _, _, _ = policy_act(opp_net, env.obs_for(1), 1, dev)
        r, done = env.step(mv0, mv1)
        # store the value as a plain float — no graph kept alive during collection
        traj.append([float(logp), float(val), r, cache])
    # GAE
    adv, gae, nextv = [0.0] * len(traj), 0.0, 0.0
    for t in reversed(range(len(traj))):
        v = traj[t][1]
        delta = traj[t][2] + gamma * nextv - v
        gae = delta + gamma * lam * gae
        adv[t] = gae; nextv = v
    ret = [adv[t] + traj[t][1] for t in range(len(traj))]
    # debug stats for this episode
    vals = [traj[t][1] for t in range(len(traj))]
    dbg = dict(ep_len=len(traj), ret_sum=sum(t[2] for t in traj),
               adv_min=min(adv), adv_max=max(adv),
               val_min=min(vals), val_max=max(vals), ret_min=min(ret), ret_max=max(ret))
    return traj, adv, ret, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="bc.pt")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--episodes_per_iter", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--minibatch", type=int, default=128, dest="minibatch",
                    help="states per gradient step. The OLD code stepped once PER "
                         "STATE (~7000 steps/iter), so the policy drifted far from "
                         "the collection policy inside one iteration and ratios "
                         "exploded. Accumulate gradients over this many states and "
                         "step once. This is the key stability fix.")
    ap.add_argument("--target-kl", type=float, default=0.03, dest="target_kl",
                    help="stop the remaining update epochs once the mean per-planet "
                         "approx-KL between old and new policy exceeds this. 0 to disable.")
    ap.add_argument("--max-grad-norm", type=float, default=0.5, dest="max_grad_norm",
                    help="gradient-norm clip applied once per minibatch step")
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start",
                    help="first game seed used for TRAINING collection (increments per "
                         "episode). MUST be disjoint from the eval seed range "
                         "(submit_agent.py uses 0..games-1) or you leak eval games into "
                         "training and inflate results. Push this high (e.g. 1_000_000) "
                         "for any run you'll evaluate on low seeds.")
    ap.add_argument("--refresh", type=int, default=20, help="iters between opponent refresh")
    ap.add_argument("--opponent", default="teacher",
                    help="'teacher' (=net_roi_support), 'self' (self-play), or the name "
                         "of any agent in ow_base.py, e.g. 'nearest_planet', 'most_production', "
                         "'defender', 'net_attacker' (weaker -> easier early signal)")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=1, dest="log_every",
                    help="print metrics every N iterations (default 1 = every iter)")
    ap.add_argument("--out", default="ppo.pt")
    ap.add_argument("--value-warmup", type=int, default=100, dest="value_warmup",
                    help="number of value-only gradient EPOCHS over a single batch of warmup "
                         "games (policy frozen). The BC checkpoint's value head is random; "
                         "calibrating it first prevents garbage advantages from destroying the "
                         "policy. This is cheap: games are collected ONCE, not per-epoch. 0 to skip.")
    ap.add_argument("--warmup-games", type=int, default=6, dest="warmup_games",
                    help="how many games to collect once for value warmup (default 6)")
    ap.add_argument("--debug", action="store_true", help="verbose per-iteration diagnostics")
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
    import ow_base
    if args.opponent == "self":
        opp_fn = None
    elif args.opponent == "teacher":
        opp_fn = ow_base.net_roi_support
    else:
        opp_fn = getattr(ow_base, args.opponent)
    print(f"training opponent: {args.opponent} | value_warmup={args.value_warmup} epochs "
          f"on {args.warmup_games} games | lr={args.lr} ent={args.ent} clip={args.clip} "
          f"eps/iter={args.episodes_per_iter} minibatch={args.minibatch} "
          f"target_kl={args.target_kl}")
    seed = args.seed_start
    best_wr = -1.0

    # ===== ONE-SHOT VALUE WARMUP =====
    # Collect a small batch of games ONCE, then train ONLY the value head for many
    # cheap epochs. This replaces the old "15 slow iterations" (each replaying games)
    # with a single round of game-playing + many fast value-only gradient passes.
    if args.value_warmup > 0:
        print(f"[warmup] collecting {args.warmup_games} games once for value calibration...")
        wbatch = []
        wwins = 0
        for _ in range(args.warmup_games):
            traj, adv, ret, dbg = collect_episode(net, opp, env, seed, dev, opp_fn=opp_fn)
            seed += 1
            wwins += 1 if env._terminal_reward() > 0 else 0
            for t in range(len(traj)):
                wbatch.append([traj[t][3], ret[t]])
        print(f"[warmup] collected {len(wbatch)} states (win_rate during collection "
              f"{wwins/args.warmup_games:.2f}). training value head for {args.value_warmup} epochs...")
        for ep in range(args.value_warmup):
            np.random.shuffle(wbatch)
            vloss_sum = 0.0
            # value warmup also minibatched, same reasoning as the main loop
            for start in range(0, len(wbatch), args.minibatch):
                chunk = wbatch[start:start + args.minibatch]
                opt.zero_grad()
                for cache, rett in chunk:
                    _, _, val, _ = recompute_logp_value(net, cache, dev)
                    vloss = F.mse_loss(val, torch.tensor(rett, device=dev, dtype=torch.float32))
                    (vloss / len(chunk)).backward()
                    vloss_sum += vloss.detach().item()
                # freeze everything except the value head
                for name, p in net.named_parameters():
                    if not name.startswith("value") and p.grad is not None:
                        p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if ep % max(1, args.value_warmup // 10) == 0 or ep == args.value_warmup - 1:
                print(f"[warmup] epoch {ep:3d}  value_loss {vloss_sum/len(wbatch):.3f}")
        print("[warmup] done — value head calibrated; starting PPO.")

    for it in range(args.iters):
        # ---- collect ----
        batch = []
        ep_rewards = []
        wins = 0
        dbgs = []
        for _ in range(args.episodes_per_iter):
            traj, adv, ret, dbg = collect_episode(net, opp, env, seed, dev, opp_fn=opp_fn)
            seed += 1
            ep_rewards.append(sum(t[2] for t in traj))
            wins += 1 if env._terminal_reward() > 0 else 0
            dbgs.append(dbg)
            for t in range(len(traj)):
                batch.append([traj[t][3], traj[t][3]["old_logp_per"], adv[t], ret[t]])
        # batch-level advantage normalization (more stable than per-episode)
        all_adv = np.array([b[2] for b in batch], dtype=np.float64)
        amean, astd = all_adv.mean(), all_adv.std() + 1e-6
        for b in batch:
            b[2] = (b[2] - amean) / astd

        # ---- update (minibatched; fixes #9-#11) ----
        ratios_seen, gradnorms, ploss_acc, vloss_acc = [], [], 0.0, 0.0
        nstep = 0
        stopped_epoch = args.epochs
        for ep in range(args.epochs):
            np.random.shuffle(batch)
            kl_sum, kl_n = 0.0, 0
            for start in range(0, len(batch), args.minibatch):
                chunk = batch[start:start + args.minibatch]
                opt.zero_grad()
                # accumulate gradients over the whole minibatch, then ONE step
                for cache, old_logp_per, advt, rett in chunk:
                    logp_per, active, val, ent = recompute_logp_value(net, cache, dev)
                    a_t = torch.tensor(advt, device=dev, dtype=torch.float32)
                    ratio = torch.exp(logp_per - old_logp_per)
                    s1 = ratio * a_t
                    s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_t
                    per_planet = -torch.min(s1, s2) * active
                    denom = active.sum().clamp(min=1)
                    pol = per_planet.sum() / denom
                    vloss = F.mse_loss(val, torch.tensor(rett, device=dev, dtype=torch.float32))
                    loss = (pol + 0.5 * vloss - args.ent * ent) / len(chunk)
                    loss.backward()
                    with torch.no_grad():
                        kl = ((old_logp_per - logp_per) * active).sum() / denom
                        kl_sum += float(kl); kl_n += 1
                        if args.debug:
                            r_active = ratio[active > 0]
                            if r_active.numel():
                                ratios_seen.append((r_active.min().item(), r_active.max().item()))
                            ploss_acc += pol.detach().item(); vloss_acc += vloss.detach().item(); nstep += 1
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()
                if args.debug:
                    gradnorms.append(float(gn))
            # KL early stop: don't grind the remaining epochs into divergence
            mean_kl = kl_sum / max(kl_n, 1)
            if args.target_kl > 0 and mean_kl > args.target_kl:
                stopped_epoch = ep + 1
                break

        if opp_fn is None and (it + 1) % args.refresh == 0:
            opp.load_state_dict(net.state_dict())
        wr = wins / args.episodes_per_iter
        if wr > best_wr:
            best_wr = wr
            torch.save({"model": net.state_dict(), "opt": opt.state_dict()},
                       args.out.replace(".pt", "_best.pt") if args.out.endswith(".pt") else args.out + "_best")
        if it % args.log_every == 0:
            line = (f"iter {it:4d}  mean_ep_reward {np.mean(ep_rewards):+.3f}  "
                    f"win_rate {wr:.2f}  best {best_wr:.2f}  kl {mean_kl:.4f}"
                    f"{'' if stopped_epoch == args.epochs else f' (early-stop @ep{stopped_epoch})'}")
            if args.debug:
                rmin = min(r[0] for r in ratios_seen) if ratios_seen else 0
                rmax = max(r[1] for r in ratios_seen) if ratios_seen else 0
                avg_len = np.mean([d["ep_len"] for d in dbgs])
                line += (f"\n        adv[norm] ~N(0,1) | raw_adv range "
                         f"[{min(d['adv_min'] for d in dbgs):+.1f},{max(d['adv_max'] for d in dbgs):+.1f}]"
                         f" | value range [{min(d['val_min'] for d in dbgs):+.1f},{max(d['val_max'] for d in dbgs):+.1f}]"
                         f" | ret range [{min(d['ret_min'] for d in dbgs):+.1f},{max(d['ret_max'] for d in dbgs):+.1f}]"
                         f"\n        ratio range [{rmin:.2f},{rmax:.2f}] (clip {1-args.clip:.1f}-{1+args.clip:.1f})"
                         f" | grad_norm avg {np.mean(gradnorms):.2f} max {np.max(gradnorms):.2f}"
                         f" | pol_loss {ploss_acc/max(nstep,1):+.3f} val_loss {vloss_acc/max(nstep,1):.2f}"
                         f" | ep_len {avg_len:.0f}")
            print(line)
    torch.save({"model": net.state_dict()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
