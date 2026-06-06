# Full 2-PLAYER pipeline (PC). One file, run and walk away:
#   Stage 1  BC               (cached harvest, parallel, plateau early-stop)
#   Stage 2  PPO phase-1      (weak/self curriculum; AUTO-STOPS when nearest bench plateaus)
#   Stage 3  PPO phase-2      (full hard mix, warm-started from phase-1 best; runs until Ctrl-C)
# Usage:   powershell -ExecutionPolicy Bypass -File .\run_2p.ps1
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"        # live output through Tee-Object (no buffering "hang")

# ---------- Stage 1/3: behavioral cloning ----------
Write-Output "=== [1/3] $(Get-Date -Format u)  BC 2p ==="
python -u bc.py --players 2 --games 800 --epochs 120 --bs 512 --out bc2.pt `
  --data-cache bc2_data.npz --patience 8 --min-epochs 20 --harvest-workers 6 |
    Tee-Object -FilePath bc2_retrain.log
if ($LASTEXITCODE -ne 0) { Write-Output "!! BC failed (exit $LASTEXITCODE) - aborting"; exit 1 }

# ---------- Stage 2/3: PPO phase-1 (curriculum; auto-stops on nearest-bench plateau) ----------
Remove-Item ppo2_phase1_warmup.pt -ErrorAction SilentlyContinue   # rebuild for the fresh BC
Write-Output "=== [2/3] $(Get-Date -Format u)  PPO phase-1 2p (auto-stop on plateau) ==="
python -u ppo.py --players 2 --init bc2_best.pt --opponent self `
  --league-size 8 --league-add-every 10 `
  --vf-coef 0.5 --lr 1e-4 --target-kl 0.05 --ent 0.01 --episodes_per_iter 16 `
  --warmup-lr 1e-3 --warmup-cache ppo2_phase1_warmup.pt `
  --mix-scripted nearest_planet_smart:0.20 --mix-scripted comet_user:0.20 `
  --eval-every 3 --eval-games 40 `
  --eval-opponent nearest_planet --bench-also teacher `
  --early-stop-patience 8 --iters 400 --out ppo2_phase1.pt |
    Tee-Object -FilePath ppo2_phase1.log
if ($LASTEXITCODE -ne 0) { Write-Output "!! phase-1 failed (exit $LASTEXITCODE) - aborting"; exit 1 }

# ---------- Stage 3/3: PPO phase-2 (full hard mix; INDEFINITE - stop with Ctrl-C) ----------
Remove-Item ppo2_phase2_warmup.pt -ErrorAction SilentlyContinue   # different mix -> fresh warmup
Write-Output "=== [3/3] $(Get-Date -Format u)  PPO phase-2 2p (full mix; runs until Ctrl-C) ==="
python -u ppo.py --players 2 --init ppo2_phase1_best.pt --opponent self `
  --league-size 12 --league-add-every 20 `
  --vf-coef 0.5 --lr 1e-4 --target-kl 0.05 --ent 0.01 --episodes_per_iter 16 `
  --warmup-lr 1e-3 --warmup-cache ppo2_phase2_warmup.pt `
  --mix-teacher 0.15 `
  --mix-scripted defender:0.05 --mix-scripted most_production:0.05 --mix-scripted comet_user:0.05 `
  --external rl_v1=..\RL\submission_orbitnet.py --mix-external rl_v1:0.15 `
  --eval-every 5 --eval-games 40 `
  --eval-opponent teacher --bench-also external:rl_v1 --bench-also nearest_planet `
  --iters 100000 --out ppo2.pt |
    Tee-Object -FilePath ppo2_phase2.log
