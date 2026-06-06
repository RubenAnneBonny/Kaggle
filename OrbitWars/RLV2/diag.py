r"""diag.py — one-shot health check for the PPO phase-1 setup.

Loads a checkpoint, collects a handful of episodes under the *phase-1 opponent
mix*, and reports the metrics that actually tell you whether PPO can learn:

  * value explained variance   (the #1 PPO health metric)
  * reward decomposition        (shaped dense vs terminal +/-5)
  * advantage signal            (does 'this node acted' correlate with advantage?)
  * activity                    (hold%, moves/turn) greedy AND sampled
  * greedy benchmark vs nearest_planet

Run:  ..\.venv\Scripts\python.exe diag.py --init bc2_best.pt
"""
import argparse, numpy as np, torch
import ppo, ow_base
from ow_env import OrbitEnv, N_MAX
from model import OrbitNet
from seeds import train_seed

HOLD = N_MAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="bc2_best.pt")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--bench-games", type=int, default=20)
    ap.add_argument("--refit-epochs", type=int, default=0,
                    help="if >0, value-only refit on a train split, then report "
                         "held-out explained variance (proves the return is learnable)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = OrbitNet().to(dev)
    ck = torch.load(args.init, map_location=dev)
    net.load_state_dict(ck["model"])
    print(f"loaded {args.init} on {dev}")

    # phase-1 opponent mix: 0.30 nearest_planet_smart, 0.30 comet_user, 0.40 self(=init)
    self_net = ppo._snapshot(net, dev)
    def opp_draw():
        r = np.random.rand()
        if r < 0.30:
            return ppo.Opponent("fn", fn=ow_base.nearest_planet_smart, dev=dev)
        if r < 0.60:
            return ppo.Opponent("fn", fn=ow_base.comet_user, dev=dev)
        return ppo.Opponent("net", nets=[self_net], dev=dev)

    env = OrbitEnv(max_steps=500, num_players=args.players, focal=0)

    all_ret, all_val, all_adv = [], [], []
    acted_flags, adv_per_node = [], []
    shaped_sums, term_vals, ep_lens, wins = [], [], [], 0
    hold_fracs, mvs = [], []
    ep_caches = []          # per-episode [(cache, return)] for the optional value refit

    for e in range(args.episodes):
        traj, adv, ret, dbg = ppo.collect_episode(net, opp_draw, env, train_seed(e), dev)
        ep_caches.append([(traj[t][3], ret[t]) for t in range(len(traj))])
        term = env._terminal_reward(0)
        wins += 1 if term > 0 else 0
        shaped_sums.append(sum(t[2] for t in traj) - term)
        term_vals.append(term)
        ep_lens.append(len(traj))
        hold_fracs.append(dbg["hold_frac"]); mvs.append(dbg["n_moves"])
        for t in range(len(traj)):
            v = traj[t][1]
            all_ret.append(ret[t]); all_val.append(v); all_adv.append(adv[t])
            cache = traj[t][3]
            own = cache["own"].float()
            n_own = float(own.sum())
            if n_own == 0:
                continue
            acted = float(((cache["tgt"] != HOLD).float() * own).sum())  # nodes that launched
            acted_flags.append(acted / n_own)        # fraction of owned nodes acting this step
            adv_per_node.append(adv[t])
        print(f"  ep {e:2d}  len {len(traj):3d}  shaped {shaped_sums[-1]:+8.2f}  "
              f"term {term:+.0f}  hold {dbg['hold_frac']:.0%}  mv {dbg['n_moves']:.1f}")

    ret = np.array(all_ret); val = np.array(all_val); adv = np.array(all_adv)
    ev = 1.0 - np.var(ret - val) / (np.var(ret) + 1e-9)   # value explained variance

    print("\n================ DIAGNOSIS ================")
    print(f"episodes {args.episodes}  train win-rate {wins/args.episodes:.2f}  "
          f"mean ep_len {np.mean(ep_lens):.0f}")
    print(f"reward:  shaped(dense) mean {np.mean(shaped_sums):+.2f} "
          f"[{np.min(shaped_sums):+.1f},{np.max(shaped_sums):+.1f}]   "
          f"terminal mean {np.mean(term_vals):+.2f}")
    print(f"         => dense term is {np.mean(np.abs(shaped_sums)):.1f} vs terminal 5.0 "
          f"(ratio {np.mean(np.abs(shaped_sums))/5.0:.1f}x)")
    print(f"value:   EXPLAINED VARIANCE {ev:+.3f}   "
          f"(<=0 means advantages are noise; >0.5 is healthy)")
    print(f"         return range [{ret.min():+.2f},{ret.max():+.2f}]  "
          f"value pred range [{val.min():+.2f},{val.max():+.2f}]")
    print(f"adv:     mean {adv.mean():+.3f}  std {adv.std():.3f}  "
          f"range [{adv.min():+.2f},{adv.max():+.2f}]")
    # does "acting more at a state" correlate with higher advantage? (learning signal)
    af = np.array(acted_flags); apn = np.array(adv_per_node)
    if af.std() > 1e-6 and apn.std() > 1e-6:
        corr = np.corrcoef(af, apn)[0, 1]
        print(f"signal:  corr(act_fraction, advantage) = {corr:+.3f}  "
              f"(per-state; ~0 means PPO can't tell acting from holding)")
    print(f"activity: hold {np.mean(hold_fracs):.0%}  moves/turn {np.mean(mvs):.2f}")

    # ---- optional: value-only refit, then held-out explained variance ----
    # The loaded value head was trained under the OLD reward; current EV says
    # nothing about whether the NEW return is learnable. Refit value-only on a
    # train split and measure EV on held-out episodes to answer that directly.
    if args.refit_epochs > 0:
        import torch.nn.functional as F
        cut = max(1, int(0.7 * len(ep_caches)))
        train_eps, val_eps = ep_caches[:cut], ep_caches[cut:]
        train_states = [sc for ep in train_eps for sc in ep]
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for epn in range(args.refit_epochs):
            np.random.shuffle(train_states)
            for s in range(0, len(train_states), 128):
                chunk = train_states[s:s + 128]; opt.zero_grad()
                for cache, rett in chunk:
                    _, _, val, _ = ppo.recompute_logp_value(net, cache, dev)
                    (F.mse_loss(val, torch.tensor(rett, device=dev, dtype=torch.float32))
                     / len(chunk)).backward()
                for nm, p in net.named_parameters():
                    if not nm.startswith("value") and p.grad is not None:
                        p.grad.zero_()
                opt.step()
        vr, vv = [], []
        with torch.no_grad():
            for ep in val_eps:
                for cache, rett in ep:
                    _, _, val, _ = ppo.recompute_logp_value(net, cache, dev)
                    vr.append(rett); vv.append(float(val))
        vr = np.array(vr); vv = np.array(vv)
        ev2 = 1.0 - np.var(vr - vv) / (np.var(vr) + 1e-9)
        print(f"refit:   held-out EXPLAINED VARIANCE after {args.refit_epochs}-epoch "
              f"value-only refit: {ev2:+.3f}  (value range now [{vv.min():+.2f},{vv.max():+.2f}])")

    # greedy benchmark vs nearest_planet (the phase-1 selection metric)
    wr = ppo.benchmark_winrate(net, dev, ow_base.nearest_planet, args.bench_games,
                               args.players, 0)
    print(f"greedy bench vs nearest_planet ({args.bench_games} games): {wr:.2f}")
    print("===========================================")


if __name__ == "__main__":
    main()
