r"""prof.py — where does PPO wall-clock actually go? Times the hot components of
collection (the per-step inner loop) and one eval game, so we know whether the
bottleneck is the GPU policy forward or the CPU game-sim / encode / decode /
scripted-opponent code. That decides whether to parallelize on CPU (like BC) or
to batch on the GPU.

Run:  ..\.venv\Scripts\python.exe prof.py --init bc2_best.pt --episodes 4
"""
import argparse, time, numpy as np, torch
import ppo, ow_env, ow_base, search_agent, model
from ow_env import OrbitEnv
from model import OrbitNet
from seeds import train_seed

T = {}            # component -> [total_seconds, n_calls]
def _acc(name, dt):
    e = T.setdefault(name, [0.0, 0]); e[0] += dt; e[1] += 1

def wrap(mod, fname, label, sync=False):
    orig = getattr(mod, fname)
    def w(*a, **k):
        if sync and torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = orig(*a, **k)
        if sync and torch.cuda.is_available(): torch.cuda.synchronize()
        _acc(label, time.perf_counter() - t0)
        return r
    setattr(mod, fname, w)
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="bc2_best.pt")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = OrbitNet().to(dev)
    net.load_state_dict(torch.load(args.init, map_location=dev)["model"])
    print(f"loaded {args.init} on {dev}")

    # time the CPU-side hot functions (they're called from inside ppo via these modules)
    wrap(ow_env, "encode_state", "encode_state(cpu)")
    wrap(ow_env, "decode_action", "decode_action(cpu)")
    wrap(search_agent, "sim_step", "sim_step(cpu game-sim)")
    wrap(ow_base, "net_roi_support", "scripted_opp(cpu)")
    wrap(ow_base, "nearest_planet_smart", "scripted_opp(cpu)")
    wrap(ow_base, "comet_user", "scripted_opp(cpu)")
    # time the GPU policy forward (sync for honest numbers)
    orig_fwd = OrbitNet.forward
    def fwd(self, *a, **k):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter(); r = orig_fwd(self, *a, **k)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        _acc("net.forward(gpu)", time.perf_counter() - t0); return r
    OrbitNet.forward = fwd

    self_net = ppo._snapshot(net, dev)
    def opp_draw():
        r = np.random.rand()
        if r < 0.30: return ppo.Opponent("fn", fn=ow_base.nearest_planet_smart, dev=dev)
        if r < 0.60: return ppo.Opponent("fn", fn=ow_base.comet_user, dev=dev)
        return ppo.Opponent("net", nets=[self_net], dev=dev)

    env = OrbitEnv(max_steps=500, num_players=2, focal=0)

    t_collect = time.perf_counter()
    steps = 0
    for e in range(args.episodes):
        traj, *_ = ppo.collect_episode(net, opp_draw, env, train_seed(e), dev)
        steps += len(traj)
    collect_s = time.perf_counter() - t_collect

    # one eval game (greedy) to gauge per-game eval cost
    t_e = time.perf_counter()
    wr = ppo.benchmark_winrate(net, dev, ow_base.nearest_planet, 4, 2, 0)
    eval_s = time.perf_counter() - t_e

    print(f"\ncollected {args.episodes} episodes ({steps} steps) in {collect_s:.1f}s "
          f"-> {1000*collect_s/steps:.1f} ms/step")
    print(f"4 greedy eval games in {eval_s:.1f}s -> {eval_s/4:.2f} s/game\n")
    print(f"{'component':28s} {'total_s':>9s} {'calls':>8s} {'ms/call':>9s} {'%collect':>9s}")
    for name, (tot, n) in sorted(T.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:28s} {tot:9.2f} {n:8d} {1000*tot/max(n,1):9.2f} {100*tot/collect_s:8.1f}%")

    # extrapolate
    print(f"\nat 16 episodes/iter (~{int(16*steps/args.episodes)} steps): "
          f"~{16*collect_s/args.episodes:.0f}s collect/iter")
    print(f"eval tick (40 games x 2 opponents = 80): ~{80*eval_s/4:.0f}s")


if __name__ == "__main__":
    main()
