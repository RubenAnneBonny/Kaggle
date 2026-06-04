# Orbit Wars — Learned Agent Pipeline (BC → PPO self-play)

A graph-attention policy that learns to play `orbit_wars`. Two training stages:
**behavioral cloning (BC)** of `net_roi_support` for a warm start, then **PPO**
fine-tuning (vs the teacher, a weaker agent, or self-play) to try to surpass it.

> **Status / honest summary.** BC reaches ~0.47 per-decision match with the
> teacher but, on its own, plays an *inconsistent* whole game (great on some
> seeds, stuck on others) — the classic behavioral-cloning distribution-shift
> problem. The mechanical decode fixes below (capture-sizing, friendly
> reinforcement) make individual moves sound. Whether PPO can push the policy
> *past* `net_roi_support` is the open question; the pipeline, diagnostics, and
> reward are all now correct and ready for that experiment. If PPO doesn't clear
> the bar on available compute, `net_roi_support` remains the strongest agent we
> have, and this infra is reusable.

---

## Files

| file | role | needs torch |
|---|---|---|
| `ow_base.py` | existing helpers + agents incl. `net_roi_support` (the teacher) and weak agents (`nearest_planet`, `most_production`, `defender`, `net_attacker`) | no |
| `search_agent.py` | engine-faithful simulator (`sim_step`) reused by `ow_env` | no |
| `ow_env.py` | state encoder, action decoder (incl. capture-sizing), fast self-play env, reward | no |
| `model.py` | graph-attention policy + value network (`OrbitNet`) | **yes** |
| `bc.py` | Stage 1: harvest teacher decisions + train (behavioral cloning) | **yes** |
| `ppo.py` | Stage 2: PPO fine-tuning with value-warmup | **yes** |
| `submit_agent.py` | load a checkpoint and play / evaluate over N games | **yes** |
| `replay.py` | play ONE game, write a watchable HTML replay + text trace | **yes** |
| **Diagnostics** | | |
| `verify_fix.py` | confirm the latest `ow_env.py` (capture-size etc.) is actually loaded | **yes** |
| `diagnose.py` | dump the agent's decision on a few real states vs the teacher | **yes** |
| `hold_bias.py` | sweep a HOLD-logit penalty; compare agent vs teacher attack rates | **yes** |
| `why_losing.py` | fraction-head accuracy + real-trajectory planet/ship curves | **yes** |
| `capture_check.py` | per-step planet/ship trajectory + send-fraction distribution | **yes** |

---

## Design (why these choices)

- **Graph attention, not a CNN.** The state is ~32 planets at continuous
  positions; what matters is pairwise relationships (who can reach whom, garrison
  ratios), not pixel locality. `model.py` embeds each planet as a node and runs
  transformer-encoder layers with a **distance bias** added to attention logits,
  so geometry is explicit.
- **Factored per-source action.** Each owned planet chooses ONE target (or HOLD)
  plus a send-fraction bucket (`FRACS = [0.25, 0.5, 0.75, 1.0]`). Matches the
  teacher (one attack per planet/turn) and keeps the log-prob tractable.
- **The network never outputs firing angles or ship counts for captures.** It
  outputs *target* + *fraction*; `ow_env.decode_action` reuses `reach()` for the
  intercept geometry and, in capture-size mode, `how_many_send()` to size attacks
  to actually capture. The policy only learns the strategic part (which planet).
- **Reward matches the engine's true scoring.** The engine decides the winner by
  **total ships** (planets' ships + fleets), not planet count — by elimination,
  or by ship-score at the 200-step limit. `ow_env` shapes on the **ship**
  differential (small nudge) and gives a dominant ±5 terminal reward decided the
  same way the engine does. (The faithful simulator ignores comets for speed;
  the final agent plays the real comet-bearing env via `submit_agent.py`.)

### `decode_action` modes (important)
- **`capture_size=True` (DEFAULT in agent, replay, PPO).** Ignores the fraction
  head for *enemy/neutral* targets and sizes each attack with `how_many_send`
  (garrison + production-in-flight + incoming enemy reinforcement + spare). If a
  source can't afford a winning attack, it **skips** it rather than dribbling
  ships that can't capture. This fixed the "attacks too small, never expands"
  failure.
- **Friendly moves (own → own)** are sized as a **ferry**, not a capture: send
  `fraction × available` ships forward. (Capture logic on your own planet is
  meaningless and was silently dropping all reinforcement before the fix.)
- **`capture_size=False`** uses the network's fraction head directly (the
  original behavior; kept for comparison via `--no-capture-size`).

---

## Setup

```bash
pip install torch kaggle_environments numpy
# keep ALL files in one directory
python model.py        # shape/backward sanity — prints tensor shapes + param count
```

> **Windows / PowerShell note.** Python commands below are identical on Windows.
> Only *shell* utilities differ: use `ls file` (not `ls -la file`), `del file`
> (not `rm`), and `Measure-Command { ... }` to time a command. If a checkpoint
> edit seems ignored, delete the `__pycache__` folder and re-run.

---

## Full workflow

### Stage 0 — sanity check the environment (no training)
```bash
python verify_fix.py            # confirms capture-size + friendly-reinforce code is live
```

### Stage 1 — Behavioral cloning (warm start)
Harvest the teacher's decisions over many games and train the policy to imitate.

```bash
python bc.py --games 800 --epochs 120 --bs 256 --attack-weight 5 --seed-offset 1000 --out bc.pt
```

- Saves **`bc_best.pt`** (best val-accuracy epoch) and `bc.pt` (last epoch).
  Use `bc_best.pt` everywhere downstream.
- `--attack-weight 5` upweights non-HOLD decisions (HOLD dominates the labels,
  ~89%); without it, target accuracy plateaus low.
- `--seed-offset 1000` harvests seeds 1000–1799, leaving **seeds 0–999 free for
  clean evaluation** (don't evaluate on training seeds).
- Watch `val_target_acc`. It plateaus around **0.45–0.48**; that's a fine warm
  start. Stop when ~8–10 epochs pass with no new best (Ctrl+C is safe —
  `bc_best.pt` is already on disk). More BC epochs **cannot** beat the teacher;
  BC can only converge toward imitating it.

**Flags:** `--games --epochs --bs --lr --out --resume --seed-offset --attack-weight`

### Evaluate the BC agent (clean seeds)
```bash
python submit_agent.py --ckpt bc_best.pt --games 300
```
Prints e.g. `bc_best.pt vs net_roi_support: 147/300 = 49.0%`. Plays both sides
(alternating) to remove side bias. Uses `capture_size` by default.

**Flags:** `--ckpt --games --hold-penalty --no-capture-size`

### Watch a game (validate visually — do this often)
```bash
python replay.py --ckpt bc_best.pt --opponent net_roi_support --seed 0
```
- Writes a **seed/result-stamped** HTML, e.g.
  `replay_bcbest_vs_net_roi_support_seed0_LOSS.html` (open in a browser).
- Prints a **per-step** text trace (planets + ships, both sides). The video's
  "Step: N" matches the trace row N exactly.
- In the video: **player 0 = ORANGE, player 1 = BLUE**. With `--side 0`
  (default) the trained agent is orange.
- If the browser shows a stale game, hard-refresh (Ctrl+F5) or open the new
  uniquely-named file (the script no longer overwrites a single `replay.html`).

**Flags:** `--ckpt --opponent --seed --side --out --hold-penalty --no-capture-size`

### Stage 2 — PPO fine-tuning
PPO starts from `bc_best.pt`. Because BC trained only the policy, the **value
head is random** — so PPO first runs a **one-shot value warmup** (collect a few
games once, then train only the value head for many cheap epochs) before any
policy update. Without this, garbage advantages destroy the policy on update 1.

**Curriculum: start vs a weak opponent**, confirm it learns, then graduate.

```bash
# 1) smoke test vs a weak, beatable opponent — confirm it LEARNS before committing
python ppo.py --init bc_best.pt --iters 60 --episodes_per_iter 12 \
  --opponent nearest_planet --lr 1e-4 --ent 0.01 \
  --value-warmup 100 --warmup-games 6 --debug --out ppo_v1.pt

# 2) if win_rate holds & climbs, graduate up the ladder
python ppo.py --init ppo_v1_best.pt --iters 100 --episodes_per_iter 12 \
  --opponent most_production --lr 1e-4 --ent 0.01 --debug --out ppo_v2.pt

# 3) finally, train vs the teacher (the real target)
python ppo.py --init ppo_v2_best.pt --iters 200 --episodes_per_iter 12 \
  --opponent teacher --lr 5e-5 --ent 0.005 --debug --out ppo_final.pt
```

Saves `<out>` (last) and `<out:_best>` (best win-rate, e.g. `ppo_v1_best.pt`).

**Reading the run:**
- **Warmup phase:** `[warmup] epoch N value_loss X` — value_loss should drop
  sharply (e.g. 0.8 → <0.1) and flatten. That's the value head calibrating.
- **`win_rate`** should **hold** its starting level (printed as "win_rate during
  collection") on the first real iters — NOT collapse to 0.00. Collapse means a
  problem; holding/climbing means it's working.
- With `--debug`, each iter also prints: raw/normalized advantage range, value
  range, return range, **ratio range** (should stay ~`[0.8, 1.2]`, the clip
  band — exploding = instability), grad-norm, separate pol/val losses, ep_len.
- `win_rate` at 12 episodes/iter moves in steps of 1/12 ≈ 0.08, so judge the
  **trend over ~10 iters**, not a single noisy iter. `mean_ep_reward` is smoother.

**Cost:** the main loop plays `episodes_per_iter` full games per iter — on CPU
that's the slow part (~minutes/iter). Warmup is now cheap (collect once). Treat
`--iters 60` as a diagnostic; only commit a long run after win_rate clearly
holds & climbs post-warmup.

**Flags:** `--init --iters --episodes_per_iter --epochs --lr --clip --ent
--refresh --opponent --max_steps --log-every --out --value-warmup
--warmup-games --debug`

### Evaluate / submit the final agent
```bash
python submit_agent.py --ckpt ppo_final_best.pt --games 300
```
Validate at **300+ games** on seeds untouched by training. The in-game `agent()`
in `submit_agent.py` is the submission entry point (greedy; falls back to
`net_roi_support` on any exception).

---

## Diagnostics (when something looks wrong)

| symptom | tool | what it tells you |
|---|---|---|
| eval suspiciously low / unchanged after a fix | `verify_fix.py` | whether the new `ow_env.py` is actually loaded (vs stale file / `__pycache__`) |
| want to see a single decision | `diagnose.py` | agent's target/HOLD/value + moves vs the teacher on 3 real states |
| agent seems too passive/aggressive | `hold_bias.py` | sweeps a HOLD penalty; agent vs teacher attack% and match% |
| high move-accuracy but still loses | `why_losing.py` | fraction-head accuracy + planet/ship curves over real games |
| does it expand? capture? send enough? | `capture_check.py` | per-step planet/ship trajectory + send-fraction distribution |

All take `--ckpt` (default `bc_best.pt`). Run from the project directory.

---

## Hard-won lessons (debugging history)

These are real bugs we hit and fixed — documented so they're not re-introduced.

1. **Eval on training seeds inflates results.** BC harvested seeds 1000–1799;
   evaluate on 0–999. Use `--seed-offset` to separate train/eval ranges.
2. **Attacks were too small to capture.** The fraction head undershot, so
   captures failed and the agent stalled on 1 planet. Fixed with
   `capture_size=True` (size attacks with `how_many_send`, skip unaffordable).
3. **Friendly reinforcement was silently dropped.** Capture logic applied to your
   own planet produced nonsense and got skipped. Fixed: own→own moves ferry a
   fraction of available ships. (Re-enables forward-staging.)
4. **Reward optimized the wrong objective.** Early reward used production /
   elimination; the engine scores on **ships**. Fixed `ow_env._diff` /
   `_terminal_reward` to match the engine exactly.
5. **PPO joint log-prob exploded.** Summing log-probs over all planets made the
   ratio `exp(Σ diffs)` blow up; clipping was useless. Fixed: **per-planet ratio
   + clip**, then average.
6. **Random value head destroyed the policy on update 1.** BC trained only the
   policy; PPO's advantages were noise. Fixed: **one-shot value warmup** (train
   only the value head first), plus batch-level advantage normalization.
7. **Trust the replay video; verify against the per-step trace.** Several false
   diagnoses were caught by watching one game. The seed-stamped filenames + the
   per-step trace exist so the video and numbers can't drift apart (and to dodge
   browser caching of a stale `replay.html`).
8. **Methodology:** effects under ~10 points need ~300 games; n ≤ 40 is a hint,
   not a result. Always confirm a code change is *live* (`verify_fix.py`) before
   trusting an eval.

---

## Quick reference (typical end-to-end)

```bash
python bc.py --games 800 --epochs 120 --bs 256 --attack-weight 5 --seed-offset 1000 --out bc.pt
python submit_agent.py --ckpt bc_best.pt --games 300
python replay.py --ckpt bc_best.pt --opponent net_roi_support --seed 0
python ppo.py --init bc_best.pt --iters 60 --episodes_per_iter 12 --opponent nearest_planet --value-warmup 100 --warmup-games 6 --debug --out ppo_v1.pt
# graduate opponent: nearest_planet -> most_production -> defender -> net_attacker -> teacher
python submit_agent.py --ckpt ppo_final_best.pt --games 300
```
