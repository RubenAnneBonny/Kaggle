# Orbit Wars RL — v2

A graph-attention policy for the Kaggle *Orbit Wars* game, trained by behavioral
cloning of an aggressive scripted teacher and then PPO self-play, on a simulator
that is **faithful to the real engine including comets**. v2 is a clean rebuild
of v1 with the issues you flagged fixed and a few correctness bugs caught along
the way.

## What changed from v1

**Data leakage is now impossible.** `seeds.py` splits the seed line into two
disjoint pools — training draws from `train_seed(i)`, evaluation from
`eval_seed(i)`, 5,000,000 apart. BC harvest and PPO rollouts can only touch the
train pool; the benchmark and `submit_agent` local eval can only touch the eval
pool. There is no shared default seed to forget about anymore.

**Comets are simulated faithfully.** The single biggest correctness fix. Comets
do **not** orbit — they fly across the board along precomputed paths at speed 4,
spawn at steps 50/150/250/350/450, and vanish off-board, taking any ships on them
with them. v1 froze them in place as if they were capturable planets, which is
wrong on every count. The v2 simulator (`search_agent.py`) imports the engine's
own comet generator and reproduces spawn/advance/expiry exactly. It was validated
in **lockstep against the real engine for 160 ticks across two comet spawns with
zero mismatches** in planet positions, ownership, ships, fleets, and comet
trajectories. (That test also surfaced a latent rotation off-by-one: the engine
rotates planets on the *pre-increment* step; v1 incremented first.)

**The agent never targets comets and never shoots through them.** Comets are
included as graph nodes (so the network can see them) but are masked out of the
attack set, and every launch is checked with `path_clear`, which uses each
comet's *real* future positions. A fleet whose straight line would sweep a comet
or a non-target planet is skipped rather than flown into it. (Aiming at a comet's
*current* spot correctly does **not** trip the filter — the comet has moved on by
the time the fleet arrives.)

**Round cap is 500** (the engine's true `episodeSteps`), not 200.

**Continuous fraction output.** The fraction head is now a **Beta distribution**
over [0, 1] — "send this fraction of available ships" — instead of four discrete
buckets. Beta is the right family for a bounded action (clean PPO log-prob and
entropy, no boundary blow-ups). At inference we use the Beta mean. A fraction
below `FRAC_FLOOR` (5%) means "send nothing"; discrete **HOLD** is still the
primary no-op and is kept.

**Net-attack feature.** Each planet node gets a signed `incoming-own minus
incoming-enemy` feature (`NODE_F` 10 → 11).

**Aggressive BC teacher.** `net_roi_aggressive` commits a much larger (still
clear-path, still affordable) fraction per attack, so cloning starts the fraction
head high instead of undershooting. Tunables live in `AGGR_EXTRA_SPARE` and
`AGGR_MIN_ATTACK_FRAC` in `ow_base.py`.

**Coordinated attacks are allowed.** The old "can't afford to capture alone →
skip" hard rule forbade two planets jointly taking a target neither could take
alone. In v2 sources accumulate ships toward a target within a turn, with a
generous `coord_cap` to stop pointless dogpiling. The hard rule is available via
`--hard-skip` if you want it back for an early curriculum.

**2-player and 4-player models.** The game runs with 2 or 4 players and the
strategies differ, so you train one model for each. `OrbitEnv` and PPO take
`--players {2,4}`; `submit_agent` infers the player count at runtime and
dispatches: more than two distinct starting owners → the 4p model, else the 2p
model.

**Checkpoint opponents.** PPO can train against a frozen earlier policy
(`--opponent ckpt:PATH.pt`) or a sampled pool of them (`--opponent pool:GLOB`),
in addition to `self`, `teacher`, `aggressive`, and any scripted agent.

## Pipeline

Train the two models independently. Example (tune iters/games to your compute):

```
# ---- 2-player model ----
python bc.py  --players 2 --games 800 --epochs 120 --out bc2.pt
python ppo.py --players 2 --init bc2_best.pt --opponent self --iters 2000 --out ppo2.pt

# ---- 4-player model ----
python bc.py  --players 4 --games 800 --epochs 120 --out bc4.pt
python ppo.py --players 4 --init bc4_best.pt --opponent self --iters 2000 --out ppo4.pt
```

Evaluate on the eval pool (never trained on) and inspect games:

```
python submit_agent.py --players 2 --model ppo2_best.pt --games 300 --opponent teacher
python submit_agent.py --players 4 --model ppo4_best.pt --games 300 --opponent teacher
python replay.py --players 2 --model ppo2_best.pt --opponent teacher --seed-idx 0
```

For submission, point the agent at both checkpoints and submit the code +
checkpoints together:

```
export OW_MODEL_2P=ppo2_best.pt
export OW_MODEL_4P=ppo4_best.pt
# kaggle entry point: submit_agent.agent
```

## Files

- `seeds.py` — disjoint train/eval seed pools; the one source of truth for which
  seeds are which.
- `ow_base.py` — v1 helper library, plus v2 additions: `comet_path_map`,
  `path_clear`, `reach_safe`, and the `net_roi_aggressive` teacher.
- `search_agent.py` — the fast simulator. v2 rewrote `snapshot` / `sim_step` /
  `_obs_from_state` for faithful comet spawn/motion/expiry.
- `model.py` — `OrbitNet`: distance-biased graph attention, target+HOLD head,
  Beta fraction head, value head. `NODE_F = 11`.
- `ow_env.py` — `encode_state` (with net-attack), `decode_action` (continuous
  fraction, comet-safe, coordinated), and the n-player 500-step `OrbitEnv`.
- `bc.py` — Stage 1 behavioral cloning of the aggressive teacher; attack-weighted
  target CE + Beta-NLL fraction loss; `--players {2,4}`.
- `ppo.py` — Stage 2 PPO self-play; Beta fraction, n-player rollouts, flexible
  opponents incl. checkpoints, value warmup, KL early-stop, minibatched updates.
- `submit_agent.py` — competition entry; dispatches 2p/4p, greedy decode, scripted
  fallback; `--model ... --games ...` for local eval.
- `replay.py` — render a game to seed/result-stamped HTML with a per-step trace.

## Design toggles

- `FRAC_FLOOR` (ow_env, default 0.05) — fractions below this send nothing.
- `coord_cap` (decode / `--coord-cap`, default 2.0) — ships committed to one
  target may reach `coord_cap × effective garrison` before further sources are
  refused. `< 0` disables the trim entirely.
- `hard_skip` (decode / `--hard-skip`, default off) — restore v1
  solo-capture-or-skip (forbids coordination).
- `AGGR_EXTRA_SPARE`, `AGGR_MIN_ATTACK_FRAC` (ow_base) — how aggressive the BC
  teacher is.
- `shaping` (OrbitEnv, default 0.003) — potential-style reward on
  `(my_score − best_opponent_score)`; the terminal ±5 (engine rule: highest
  positive ship total wins, ties win) dominates.

## Hard-won lessons

Carried from v1: clip the PPO ratio **per planet**, not on the joint action
(the joint ratio is `exp` of a sum of N differences and explodes); step the
optimizer **once per minibatch**, not once per state; **warm up the value head**
before letting advantages move the policy; in self-play the training win-rate is
pinned near 0.5 and is meaningless — **select on a greedy benchmark vs a fixed
opponent**.

New in v2: the engine rotates planets on the **pre-increment** step (fix your
simulator's tick order or everything drifts); **comets follow paths, not
orbits**, and must spawn/expire to be faithful; **kaggle fills agent positional
args by count**, so any agent with extra parameters gets the framework's config
object bound to them — agents must be single-arg; with a continuous fraction the
**fraction log-prob must receive PPO credit on every non-HOLD source** (v1 only
counted ferries because captures ignored the fraction).

## Not yet ported

The v1 diagnostics (`verify_fix`, `diagnose`, `hold_bias`, `why_losing`,
`capture_check`, `cuda_check`) reference the removed discrete-fraction constants
(`FRACS` / `N_FRAC`) and `capture_size`, so they need updating for the continuous
head before they'll run. Say the word and I'll port them.
