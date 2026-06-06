# RESUME 2-PLAYER pipeline (PC). Skips BC and REUSES the cached warmups.
# Unlike run_2p.ps1 this does NOT retrain BC and does NOT delete the warmup
# caches — so both PPO phases skip their value-warmup collect/train when a cache
# is present (and build+cache it once if it isn't, e.g. phase-2's first run).
#   (reuses)   bc2_best.pt                 BC checkpoint from a previous run
#   (reuses)   ppo2_phase1_warmup.pt       phase-1 value calibration
#   (reuses)   ppo2_phase2_warmup.pt       phase-2 value calibration (built on 1st run if missing)
#   Stage 1  PPO phase-1   (weak/self curriculum; AUTO-STOPS when nearest bench plateaus)
#   Stage 2  PPO phase-2   (full hard mix, warm-started from phase-1 best; runs until Ctrl-C)
# Usage:   powershell -ExecutionPolicy Bypass -File .\run_2p_resume.ps1
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"        # live output through Tee-Object (no buffering "hang")

# Refuse to start if the BC checkpoint from a prior run is missing.
foreach ($f in @("bc2_best.pt")) {
  if (-not (Test-Path $f)) {
    Write-Output "!! missing $f - run run_2p.ps1 first; aborting"
    exit 1
  }
}

# The normalized-lead reward change INVALIDATES any old value-warmup cache (it was
# calibrated under the old unbounded shaping). Drop the stale phase-1 warmup so it
# rebuilds fresh under the current reward + opponent mix; --warmup-cache re-caches it.
Remove-Item ppo2_phase1_warmup.pt -ErrorAction SilentlyContinue

# ---------- Stage 1/2: PPO phase-1 (curriculum; auto-stops on nearest-bench plateau) ----------
# NOTE: ppo2_phase1_warmup.pt is intentionally kept (reused, not rebuilt).
Write-Output "=== [1/2] $(Get-Date -Format u)  PPO phase-1 2p (resume; reuse warmup; auto-stop on plateau) ==="
python -u ppo.py --players 2 --init bc2_best.pt --opponent self `
  --league-size 8 --league-add-every 10 `
  --vf-coef 0.25 --lr 3e-5 --target-kl 0.05 --ent 0.0 --episodes_per_iter 16 `
  --warmup-lr 1e-3 --warmup-games 12 --warmup-cache ppo2_phase1_warmup.pt `
  --mix-weak 0.30 --mix-scripted nearest_planet_smart:0.15 --mix-scripted comet_user:0.15 `
  --eval-every 5 --eval-games 30 --workers 8 `
  --eval-opponent nearest_planet `
  --early-stop-patience 8 --iters 400 --out ppo2_phase1.pt |
    Tee-Object -FilePath ppo2_phase1.log
if ($LASTEXITCODE -ne 0) { Write-Output "!! phase-1 failed (exit $LASTEXITCODE) - aborting"; exit 1 }

# ---------- Stage 2/2: PPO phase-2 (full hard mix; INDEFINITE - stop with Ctrl-C) ----------
# STABILITY FIX (the old phase-2 self-destructed ~iter 100 in a self-play entropy
# collapse): lr 1e-4 -> 5e-5, ent 0.003 -> 5e-4 (the constant entropy pressure that
# inflated entH 0.3->1.8), and the mix is rebalanced toward FIXED anchors (~65%)
# instead of 55% self-play league, so the collapse has nothing to feed on.
#   + GAUNTLET LADDER: each time teacher bench clears --promote-threshold the agent
#     snapshots itself to ppo2_gauntlet_<k>.pt, then keeps benchmarking AND sparring
#     against that past strong self (own ppo2_best_gauntlet_<k>.pt) — a self-made
#     curriculum so a long run keeps finding harder targets instead of saturating.
#   + COLLAPSE GUARD: auto-stop if entropy/passivity runs away (best files preserved).
# The phase-2 warmup cache was built under the OLD mix, so drop it (rebuilds fresh).
Remove-Item ppo2_phase2_warmup.pt -ErrorAction SilentlyContinue
Write-Output "=== [2/2] $(Get-Date -Format u)  PPO phase-2 2p (full mix; runs until Ctrl-C) ==="
python -u ppo.py --players 2 --init ppo2_phase1_best.pt --opponent self `
  --league-size 12 --league-add-every 20 `
  --vf-coef 0.5 --lr 5e-5 --target-kl 0.04 --ent 5e-4 --episodes_per_iter 16 `
  --warmup-lr 1e-3 --warmup-games 12 --warmup-cache ppo2_phase2_warmup.pt `
  --mix-teacher 0.20 --mix-weak 0.10 `
  --mix-scripted defender:0.05 --mix-scripted most_production:0.05 --mix-scripted comet_user:0.05 `
  --external rl_v1=..\RL\submission_orbitnet.py --mix-external rl_v1:0.20 `
  --mix-gauntlet 0.15 --promote-threshold 0.85 --promote-every 3 --gauntlet-max 8 `
  --collapse-entH 1.1 --collapse-hold 0.40 --collapse-patience 15 `
  --eval-every 8 --eval-games 30 --workers 8 `
  --eval-opponent teacher --bench-also external:rl_v1 --bench-also nearest_planet `
  --iters 100000 --out ppo2.pt |
    Tee-Object -FilePath ppo2_phase2.log
