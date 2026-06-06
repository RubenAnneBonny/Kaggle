r"""test_parallel.py — verify the multiprocessing path.

  (1) EVAL: parallel.eval_winrate == sequential ppo.benchmark_winrate EXACTLY
      (both greedy + same seeds on CPU -> identical decisions).
  (2) COLLECTION: parallel.collect returns one valid (traj,adv,ret,dbg)+win per
      seed, with non-empty NumPy caches the batched update can consume.

Run:  ..\.venv\Scripts\python.exe test_parallel.py --init bc2_best.pt
"""
import argparse, numpy as np, torch
import ppo, parallel, ow_base
from model import OrbitNet
from seeds import train_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="bc2_best.pt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games", type=int, default=16)
    args = ap.parse_args()

    net = OrbitNet()                      # CPU, so sequential and workers match bit-for-bit
    net.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    net.eval()
    nw = parallel.init_pool(args.workers)
    print(f"pool workers: {nw}")

    # (1) eval equivalence
    seq = ppo.benchmark_winrate(net, "cpu", ow_base.nearest_planet, args.games, 2, 0)
    par = parallel.eval_winrate(net, "nearest_planet", {}, args.games, 2, 0)
    print(f"(1) eval win-rate  sequential {seq:.4f}  parallel {par:.4f}  "
          f"-> {'MATCH' if abs(seq - par) < 1e-9 else 'MISMATCH'}")

    # (2) collection smoke
    self_pool = [ppo._snapshot(net, "cpu")]
    mix = {"script:nearest_planet_smart": 0.3, "script:comet_user": 0.3, "league": 0.4}
    seeds = [train_seed(i) for i in range(args.games)]
    results, wins = parallel.collect(net, self_pool, mix, {}, seeds, 2, 200)
    assert len(results) == args.games and len(wins) == args.games, "wrong count"
    assert all(r is not None for r in results), "a seed slot was never filled"
    states = sum(len(t[0]) for t in results)
    sample = results[0][0][0][3]          # first game, first step, cache
    keys_ok = all(k in sample for k in ("nf", "nm", "am", "own", "frac_active",
                                        "tgt", "frac", "old_logp_per"))
    print(f"(2) collected {len(results)} games, {states} states, wins {sum(wins)}; "
          f"cache keys {'OK' if keys_ok else 'MISSING'}; "
          f"nf shape {np.asarray(sample['nf']).shape}")

    ok = abs(seq - par) < 1e-9 and keys_ok and all(r is not None for r in results)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    parallel.close_pool()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
