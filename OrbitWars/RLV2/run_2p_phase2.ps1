# PHASE-2 ONLY (skips BC + phase-1). Phase 1 already worked, so this just (re)runs
# the stabilized phase-2 with the gauntlet ladder + collapse guard.
#
#   .\run_2p_phase2.ps1            # RERUN phase-2 fresh from the phase-1 best
#   .\run_2p_phase2.ps1 -FromBest  # CONTINUE from phase-2's saved best (ppo2_best.pt)
#
# -FromBest warm-starts weights from ppo2_best.pt (the iter-87 snapshot, teacher
# 0.67). It only loads WEIGHTS — the value head + optimizer are re-calibrated by a
# fresh value-warmup — so it's a clean, stable continuation under the fixed hparams.
# A full rerun from ppo2_phase1_best.pt is the safer default if you want a clean slate.
param([switch]$FromBest)
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"

$init = "ppo2_phase1_best.pt"
if ($FromBest) { $init = "ppo2_best.pt" }
if (-not (Test-Path $init)) {
  Write-Output "!! missing $init - run run_2p_resume.ps1 (or drop -FromBest); aborting"; exit 1
}

# Mix changed vs the original phase-2, so the old value-warmup cache is stale: rebuild.
Remove-Item ppo2_phase2_warmup.pt -ErrorAction SilentlyContinue
Write-Output "=== PPO phase-2 2p  init=$init  $(Get-Date -Format u) (stabilized + gauntlet; Ctrl-C to stop) ==="
python -u ppo.py --players 2 --init $init --opponent self `
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
