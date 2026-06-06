r"""bench_collect.py — apples-to-apples: batched vs sequential collection for the
same 16 games, and batched vs per-state update for one minibatch. Tells us where
the remaining wall-clock is (GPU forward vs CPU game-sim) so we know whether the
next lever is more GPU batching or CPU multiprocessing.

Run:  ..\.venv\Scripts\python.exe bench_collect.py --init bc2_best.pt
"""
import argparse, time, numpy as np, torch
import ppo, ow_base
from ow_env import OrbitEnv
from model import OrbitNet
from seeds import train_seed

HOLD = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="bc2_best.pt")
    ap.add_argument("--games", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = OrbitNet().to(dev)
    net.load_state_dict(torch.load(args.init, map_location=dev)["model"])

    self_net = ppo._snapshot(net, dev)
    def opp_draw():
        r = np.random.rand()
        if r < 0.3: return ppo.Opponent("fn", fn=ow_base.nearest_planet_smart, dev=dev)
        if r < 0.6: return ppo.Opponent("fn", fn=ow_base.comet_user, dev=dev)
        return ppo.Opponent("net", nets=[self_net], dev=dev)

    seeds = [train_seed(i) for i in range(args.games)]

    # ---- batched collection ----
    envs = [OrbitEnv(max_steps=500, num_players=2, focal=0) for _ in range(args.games)]
    np.random.seed(0); torch.manual_seed(0)
    t0 = time.perf_counter()
    results = ppo.collect_batched(net, opp_draw, envs, seeds, dev)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_batched = time.perf_counter() - t0
    steps_b = sum(len(traj) for traj, *_ in results)

    # ---- sequential collection (old path) ----
    env = OrbitEnv(max_steps=500, num_players=2, focal=0)
    np.random.seed(0); torch.manual_seed(0)
    t0 = time.perf_counter()
    steps_s = 0
    for sd in seeds:
        traj, *_ = ppo.collect_episode(net, opp_draw, env, sd, dev)
        steps_s += len(traj)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_seq = time.perf_counter() - t0

    print(f"collection of {args.games} games:")
    print(f"  sequential (old): {t_seq:6.1f}s   ({steps_s} steps, {1000*t_seq/steps_s:.1f} ms/step)")
    print(f"  batched    (new): {t_batched:6.1f}s   ({steps_b} steps, {1000*t_batched/steps_b:.1f} ms/step)")
    print(f"  speedup: {t_seq/t_batched:.2f}x\n")

    # ---- update: per-state loop vs batched, one 128 minibatch ----
    caches = [t[3] for traj, _, _, _ in results for t in traj][:128]
    advs = np.random.randn(len(caches)).astype(np.float32)
    rets = np.array([0.0] * len(caches), np.float32)

    # batched update step
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        lp, own, val, ent, den = ppo._batched_policy(net, caches, dev, grad=True)
        old = ppo._stack(caches, "old_logp_per", dev, torch.float32)
        a = torch.as_tensor(advs, device=dev).unsqueeze(1)
        ratio = torch.exp(lp - old)
        pol = ((-torch.min(ratio*a, torch.clamp(ratio,0.8,1.2)*a)*own).sum(1)/den).mean()
        loss = pol + 0.5*torch.nn.functional.mse_loss(val, torch.as_tensor(rets,device=dev)) - 0.03*ent.mean()
        loss.backward(); net.zero_grad()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_upd_b = (time.perf_counter() - t0) / 5

    # per-state update step (old path)
    from model import build_edge
    def to_torch_cache(c):
        nf = torch.as_tensor(c["nf"]).unsqueeze(0).to(dev)
        return dict(nf=nf, nm=torch.as_tensor(c["nm"]).unsqueeze(0).to(dev),
                    am=torch.as_tensor(c["am"]).unsqueeze(0).to(dev), edge=build_edge(nf),
                    own=torch.as_tensor(c["own"], dtype=torch.bool, device=dev),
                    frac_active=torch.as_tensor(c["frac_active"], dtype=torch.bool, device=dev),
                    tgt=torch.as_tensor(c["tgt"], dtype=torch.long, device=dev),
                    frac=torch.as_tensor(c["frac"], dtype=torch.float32, device=dev))
    tcs = [to_torch_cache(c) for c in caches]
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        for c in tcs:
            lpo, active, vo, ento = ppo.recompute_logp_value(net, c, dev)
            (lpo.sum()*0 + vo*0 + ento*0).backward()
        net.zero_grad()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_upd_s = (time.perf_counter() - t0) / 5

    print(f"update, one 128-state minibatch (fwd+bwd):")
    print(f"  per-state (old): {1000*t_upd_s:6.0f} ms")
    print(f"  batched   (new): {1000*t_upd_b:6.0f} ms   ({t_upd_s/t_upd_b:.1f}x)")


if __name__ == "__main__":
    main()
