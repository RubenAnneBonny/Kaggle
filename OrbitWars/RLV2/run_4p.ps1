# Full 4-PLAYER pipeline (laptop). One file, run and walk away:
#   Stage 1  BC               (cached harvest, sequential + small batch: laptop RAM is tight)
#   Stage 2  PPO phase-1      (weak/self curriculum; AUTO-STOPS when nearest bench plateaus)
#   Stage 3  PPO phase-2      (full 4p hard mix, warm-started from phase-1 best; runs until Ctrl-C)
# Usage:   powershell -ExecutionPolicy Bypass -File .\run_4p.ps1
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"

# ---------- Stage 1/3: behavioral cloning ----------
Write-Output "=== [1/3] $(Get-Date -Format u)  BC 4p ==="
python -u bc.py --players 4 --games 800 --epochs 120 --bs 256 --out bc4.pt `
  --data-cache bc4_data.npz --patience 8 --min-epochs 20 |
    Tee-Object -FilePath bc4_retrain.log
if ($LASTEXITCODE -ne 0) { Write-Output "!! BC failed (exit $LASTEXITCODE) - aborting"; exit 1 }

# ---------- Stage 2/3: PPO phase-1 (curriculum; auto-stops on nearest-bench plateau) ----------
Remove-Item ppo4_phase1_warmup.pt -ErrorAction SilentlyContinue
Write-Output "=== [2/3] $(Get-Date -Format u)  PPO phase-1 4p (auto-stop on plateau) ==="
python -u ppo.py --players 4 --init bc4_best.pt --opponent self `
  --league-size 8 --league-add-every 10 `
  --vf-coef 0.25 --lr 1e-4 --target-kl 0.05 --ent 0.05 --episodes_per_iter 16 `
  --warmup-lr 1e-3 --warmup-cache ppo4_phase1_warmup.pt `
  --mix-scripted nearest_planet_smart:0.20 --mix-scripted comet_user:0.20 `
  --eval-every 3 --eval-games 40 `
  --eval-opponent nearest_planet --bench-also teacher `
  --early-stop-patience 8 --iters 400 --out ppo4_phase1.pt |
    Tee-Object -FilePath ppo4_phase1.log
if ($LASTEXITCODE -ne 0) { Write-Output "!! phase-1 failed (exit $LASTEXITCODE) - aborting"; exit 1 }

# ---------- Stage 3/3: PPO phase-2 (full 4p hard mix; INDEFINITE - stop with Ctrl-C) ----------
Remove-Item ppo4_phase2_warmup.pt -ErrorAction SilentlyContinue
Write-Output "=== [3/3] $(Get-Date -Format u)  PPO phase-2 4p (full mix; runs until Ctrl-C) ==="
python -u ppo.py --players 4 --init ppo4_phase1_best.pt --opponent self `
  --league-size 16 --league-add-every 20 `
  --vf-coef 0.25 --lr 1e-4 --target-kl 0.05 --ent 0.03 --episodes_per_iter 16 `
  --warmup-lr 1e-3 --warmup-cache ppo4_phase2_warmup.pt `
  --mix-teacher 0.12 --mix-aggressive 0.08 --mix-weak 0.05 `
  --mix-scripted defender:0.05 --mix-scripted most_production:0.05 --mix-scripted comet_user:0.05 `
  --external rl_v1=..\RL\submission_orbitnet.py --mix-external rl_v1:0.10 `
  --eval-every 5 --eval-games 40 `
  --eval-opponent teacher --bench-also external:rl_v1 --bench-also nearest_planet `
  --iters 100000 --out ppo4.pt |
    Tee-Object -FilePath ppo4_phase2.log
