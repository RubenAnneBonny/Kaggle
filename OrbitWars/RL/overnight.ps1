# overnight.ps1 — self-driving train/eval loop. Runs until YOU Ctrl+C.
#
# SEED HYGIENE (no data leakage):
#   - EVAL  uses seeds 0..(games-1)   -> reserved, never trained on.
#   - TRAIN uses seeds >= 1,000,000, and each cycle advances by 100,000 so every
#     cycle sees FRESH games disjoint from eval AND from other cycles.
#   (BC harvested 1000..1799; 0..999 were always meant for clean eval. This keeps
#    training clear of both BC and eval seeds.)
#
# Each cycle: train 25 PPO iters vs teacher, then a 100-game greedy eval
# (~20 min, 95% margin ~+/-9 pts at p~0.7). Chains each cycle from the previous
# _best so the policy keeps accumulating. Every cycle is timestamped + logged.
#
# Run:    powershell -ExecutionPolicy Bypass -File overnight.ps1
# Morning: Ctrl+C, open overnight_log.txt. Read the eval TREND across cycles
#          (not any single noisy point). Keep the latest cycle clearly above
#          the ~69.7% baseline; if flat, the teacher is tapped out.

$ErrorActionPreference = "Continue"

# ---- knobs ----
$start     = "ppo_v3_best.pt"   # baseline policy (~69.7%)
$iters     = 25                 # PPO iters per cycle
$eps       = 32
$lr        = "5e-5"
$ent       = "0.005"
$opp       = "teacher"
$games     = 100                # eval games/cycle: ~20 min, +/-9 pts
$seedbase  = 1000000            # training seeds start here (far above eval 0..99)
$seedstep  = 100000             # each cycle advances training seeds by this much
$log       = "overnight_log.txt"

"==== self-driving loop started $(Get-Date) ====" | Tee-Object -FilePath $log -Append
"start=$start iters/cycle=$iters eps=$eps lr=$lr ent=$ent opp=$opp eval_games=$games" |
    Tee-Object -FilePath $log -Append
"SEEDS: eval=0..$($games-1) (reserved) | train>=$seedbase, +$seedstep per cycle (disjoint)" |
    Tee-Object -FilePath $log -Append
"baseline to beat: ~69.7%. Read the eval TREND across cycles, not single points." |
    Tee-Object -FilePath $log -Append

$prev = $start
$c = 0
while ($true) {
    $c++
    $out       = "ppo_night_c$c.pt"
    $best      = "ppo_night_c${c}_best.pt"
    $seedstart = $seedbase + ($c - 1) * $seedstep
    "`n---- cycle $c  TRAIN START $(Get-Date)  init=$prev  train_seeds>=$seedstart ----" |
        Tee-Object -FilePath $log -Append

    python ppo.py --init $prev --iters $iters --episodes_per_iter $eps `
        --opponent $opp --lr $lr --ent $ent --value-warmup 0 `
        --seed-start $seedstart --out $out 2>&1 | Out-File "train_c$c.txt"

    "     cycle $c  TRAIN DONE  $(Get-Date)" | Tee-Object -FilePath $log -Append

    if (-not (Test-Path $best)) {
        "  WARNING: $best missing (cycle crashed). Re-using $prev next cycle." |
            Tee-Object -FilePath $log -Append
        continue
    }

    "  cycle $c  EVAL ($games games, seeds 0..$($games-1)) $(Get-Date):" |
        Tee-Object -FilePath $log -Append
    python submit_agent.py --ckpt $best --games $games 2>&1 |
        Tee-Object -FilePath $log -Append

    $prev = $best   # accumulate: next cycle builds on this one
}
