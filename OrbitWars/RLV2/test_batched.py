r"""test_batched.py — prove the batched collection/update path is numerically
identical to the original per-state path before trusting it for training.

Checks, on real collected states:
  (1) batched _batched_policy(B>1) == looping it per-state (B=1)  [no cross-sample
      contamination through padding / masks / distributions]
  (2) _batched_policy == the original recompute_logp_value semantics  [logp, value,
      entropy match the code the per-state trainer used]

Run:  ..\.venv\Scripts\python.exe test_batched.py --init bc2_best.pt
"""
import argparse, numpy as np, torch
import ppo, ow_base
from ow_env import OrbitEnv
from model import OrbitNet, build_edge
from seeds import train_seed


def to_torch_cache(c, dev):
    nf = torch.as_tensor(c["nf"]).unsqueeze(0).to(dev)
    nm = torch.as_tensor(c["nm"]).unsqueeze(0).to(dev)
    am = torch.as_tensor(c["am"]).unsqueeze(0).to(dev)
    return dict(nf=nf, nm=nm, am=am, edge=build_edge(nf),
                own=torch.as_tensor(c["own"], dtype=torch.bool, device=dev),
                frac_active=torch.as_tensor(c["frac_active"], dtype=torch.bool, device=dev),
                tgt=torch.as_tensor(c["tgt"], dtype=torch.long, device=dev),
                frac=torch.as_tensor(c["frac"], dtype=torch.float32, device=dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="bc2_best.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = OrbitNet().to(dev)
    net.load_state_dict(torch.load(args.init, map_location=dev)["model"])
    net.eval()

    self_net = ppo._snapshot(net, dev)
    def opp_draw():
        r = np.random.rand()
        if r < 0.3: return ppo.Opponent("fn", fn=ow_base.nearest_planet_smart, dev=dev)
        if r < 0.6: return ppo.Opponent("fn", fn=ow_base.comet_user, dev=dev)
        return ppo.Opponent("net", nets=[self_net], dev=dev)

    envs = [OrbitEnv(max_steps=120, num_players=2, focal=0) for _ in range(6)]
    results = ppo.collect_batched(net, opp_draw, envs, [train_seed(i) for i in range(6)], dev)
    caches = [t[3] for traj, _, _, _ in results for t in traj]
    # take a representative spread (varying owned counts / board states)
    caches = caches[::max(1, len(caches) // 130)][:130]
    print(f"testing on {len(caches)} real collected states")

    # (1) batched == per-state
    lp_b, own_b, v_b, ent_b, den_b = ppo._batched_policy(net, caches, dev, grad=False)
    e_lp = e_v = e_ent = 0.0
    for i, c in enumerate(caches):
        lp1, own1, v1, ent1, den1 = ppo._batched_policy(net, [c], dev, grad=False)
        e_lp = max(e_lp, float((lp_b[i] - lp1[0]).abs().max()))
        e_v = max(e_v, float((v_b[i] - v1[0]).abs()))
        e_ent = max(e_ent, float((ent_b[i] - ent1[0]).abs()))
    print(f"(1) batched(B={len(caches)}) vs per-state(B=1):  "
          f"max|dlogp| {e_lp:.2e}  max|dvalue| {e_v:.2e}  max|dent| {e_ent:.2e}")

    # (2) _batched_policy vs original recompute_logp_value
    d_lp = d_v = d_ent = 0.0
    for i, c in enumerate(caches):
        lpo, active, vo, ento = ppo.recompute_logp_value(net, to_torch_cache(c, dev), dev)
        d_lp = max(d_lp, float((lp_b[i] - lpo).abs().max()))
        d_v = max(d_v, float((v_b[i] - vo).abs()))
        d_ent = max(d_ent, float((ent_b[i] - ento).abs()))
    print(f"(2) batched vs recompute_logp_value:           "
          f"max|dlogp| {d_lp:.2e}  max|dvalue| {d_v:.2e}  max|dent| {d_ent:.2e}")

    tol = 2e-3
    ok = max(e_lp, e_v, e_ent, d_lp, d_v, d_ent) < tol
    print(f"\n{'PASS' if ok else 'FAIL'} (tolerance {tol:.0e}) — batched path is "
          f"{'equivalent' if ok else 'NOT equivalent'} to the per-state path")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
