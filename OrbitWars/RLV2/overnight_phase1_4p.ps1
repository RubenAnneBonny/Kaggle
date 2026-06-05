# Overnight (4-PLAYER): (1) retrain BC 4p with the bounded Beta fraction head
# (fixes the conc-265 delta collapse), then (2) PPO phase-1 on the fresh, healthy
# warm-start. Mirror of overnight_phase1.ps1 but every artifact is renamed to a
# 4p-only path so it can run concurrently on another machine with NO file
# collisions (and therefore no git merge conflicts on outputs).
# Run from RLV2:   .\overnight_phase1_4p.ps1
Set-Location $PSScriptRoot

# The old warmup cache was built from the OLD (collapsed) BC -> stale. Force a rebuild.
Remove-Item ppo4_phase1_warmup.pt -ErrorAction SilentlyContinue

Write-Output "=== [1/2] $(Get-Date -Format u)  Retraining BC 4p (bounded fraction head) ==="
python bc.py --players 4 --games 800 --epochs 120 --bs 256 --out bc4.pt |
    Tee-Object -FilePath bc4_retrain.log
if ($LASTEXITCODE -ne 0) {
    Write-Output "!! BC retrain failed (exit $LASTEXITCODE) - aborting before PPO."
    exit 1
}

Write-Output "=== [2/2] $(Get-Date -Format u)  PPO phase-1 (4p) on fresh BC ==="
python ppo.py --players 4 --init bc4_best.pt --opponent self `
  --league-size 8 --league-add-every 10 `
  --vf-coef 0.25 --lr 1e-4 --target-kl 0.05 --ent 0.05 --episodes_per_iter 16 `
  --warmup-lr 1e-3 --warmup-cache ppo4_phase1_warmup.pt `
  --mix-scripted nearest_planet_smart:0.20 --mix-scripted comet_user:0.20 `
  --eval-every 5 --eval-games 30 `
  --eval-opponent nearest_planet --bench-also teacher `
  --iters 800 --out ppo4_phase1.pt |
    Tee-Object -FilePath ppo4_phase1.log

Write-Output "=== done $(Get-Date -Format u) ==="
