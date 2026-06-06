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

**Rolling self-play league + per-slot opponent mix.** v1 self-play collapsed
because the agent only ever saw one frozen opponent at a time and overfit to it
— the benchmark dropped from ~0.50 to ~0.40 within a few dozen iters of pure
self-play. v2 makes the opponent a *distribution*, not a single net:

  * `--league-size N` keeps a rolling FIFO pool of the last `N` self-snapshots
    in memory. With `--league-size 1` (default) the pool is one element and
    behavior is byte-identical to before. `--league-add-every K` (default `0` →
    use `--refresh`) controls how often a fresh snapshot is appended.
  * `--mix-teacher`, `--mix-aggressive`, `--mix-weak` set per-slot probabilities
    of using a scripted opponent (`net_roi_support`, `net_roi_aggressive`,
    `nearest_planet`). `--mix-scripted NAME:WEIGHT` (repeatable) adds any other
    callable from `ow_base.py` — useful for strategically distinct shapes the
    teacher family doesn't cover. The remainder goes to the league. Anchors
    the policy to the real objective from iter 0 — no waiting for self-play
    to "settle".
  * Each non-focal slot in every episode draws **independently** from this
    distribution and keeps its identity for the whole game (coherent gameplay,
    not per-step identity swaps). In 4p this naturally varies the table:
    sometimes three league snapshots, sometimes teacher + league + weak,
    sometimes mixed with an external agent.
  * Setting any `--mix-*` flag (or `--league-size > 1` with `--opponent self`)
    promotes the run onto the per-episode-draw path; with no flags set the run
    follows the original single-opponent path bit-for-bit.

**External agents as sparring partners.** `--external NAME=PATH` loads a `.py`
file's `agent(obs)` callable and registers it under `NAME`. `--mix-external
NAME:WEIGHT` (repeatable) gives that agent a slot weight. The intended use is
training against the **prior Kaggle submission** — `RL/submission_orbitnet.py`
exports a self-contained `agent` (model weights inlined as base64) — so the new
model gets a direct sparring partner that represents real prior-art play. The
same agent can be used as a benchmark target with `--eval-opponent
external:NAME`.

## Pipeline

Train the two models independently. The PPO commands below use the rolling
league + opponent mix from iter 0 — this is the recommended path; the
single-opponent commands at the bottom are kept for back-compat sanity checks
only.

### One-command pipeline (recommended)

`run_2p.ps1` (PC) and `run_4p.ps1` (laptop) chain the whole thing end-to-end —
**run one file and walk away**:

```
powershell -ExecutionPolicy Bypass -File .\run_2p.ps1     # or run_4p.ps1
```

Each does three stages, logging live to per-stage `.log` files:

1. **BC** — cached harvest (`--data-cache`), parallel on the PC, plateau
   early-stop.
2. **PPO phase-1** — weak/self **curriculum** to lift the warm-start off the
   floor; **auto-stops** when the `nearest` benchmark plateaus
   (`--early-stop-patience`), since phase-1 only needs to be "good enough" to
   seed phase-2. Saves `ppoN_phase1_best.pt`. The mix must include a genuinely
   **winnable** opponent — `--mix-weak 0.30` (plain `nearest_planet`, the eval
   family; disjoint seeds, so no leak) — so the focal actually wins a fair share
   and the normalized advantages carry signal. Without one (the old mix was
   `nearest_planet_smart:0.30 + comet_user:0.30 + league`, all of which a
   20%-competent warm-start loses to) the policy gradient is zero-mean noise and
   the entropy bonus becomes the only coherent gradient: `entH` climbs, `hold%`
   and `conc` fall, `bench` decays — the monotone-collapse tell. Phase-1 therefore
   runs `--ent 0.0` (warm-started from BC, it needs no exploration bonus) at the
   stable `--lr 3e-5 --vf-coef 0.25`, keeping league ≈ 0.4 (most weight near-skill,
   per the opponent-mix philosophy below).
3. **PPO phase-2** — the full hard mix (teacher / rl_v1 / league), warm-started
   from phase-1's best, running **indefinitely** until you `Ctrl-C`. Saves
   `ppoN_best.pt` (the submit target) whenever the teacher benchmark improves.

`run_2p_resume.ps1` is the same pipeline **minus BC**: it warm-starts phase-1
from an existing `bc2_best.pt`. Use it to re-launch the PPO phases — e.g. after an
hp change — without paying for a BC retrain. It still rebuilds the **value-warmup**
each run (the cache is keyed to the reward + opponent mix; reusing a stale one
mis-calibrates the value head). The BC checkpoint is reward-independent, so it is
safely reused.

The stages below document the individual commands the orchestrator runs.

### Stage 1 — Behavioral cloning

```
# ---- 2-player (PC: bigger batch + parallel harvest) ----
python bc.py --players 2 --games 800 --epochs 120 --bs 512 --out bc2.pt `
  --data-cache bc2_data.npz --patience 8 --min-epochs 20 --harvest-workers 6

# ---- 4-player (laptop: smaller batch, sequential harvest — RAM-tight) ----
python bc.py --players 4 --games 800 --epochs 120 --bs 256 --out bc4.pt `
  --data-cache bc4_data.npz --patience 8 --min-epochs 20
```

Each run writes `bcN.pt` (last epoch) and `bcN_best.pt` (best eval).

Speed knobs (the harvest of 800 games is the expensive part, and 120 epochs is
overkill — the val plateau is usually reached by ~epoch 40):

- `--data-cache PATH.npz` — harvest the 800 games **once** and save the dataset
  there; every later run loads it and skips harvesting entirely. (Masks are
  stored as `uint8` and validation is minibatched, so even the 4p dataset stays
  within a small GPU / laptop RAM.) Delete the `.npz` to re-harvest after
  changing `--games`/`--players` (it warns on a mismatch).
- `--patience N` / `--min-epochs M` — **auto-detect the plateau**: stop once val
  `target_acc` hasn't improved for `N` epochs (never before epoch `M`), instead
  of always running all `--epochs`. The per-epoch log shows `(no-improve K)` so
  you can watch it count toward the stop. Default `8`/`20` (≈ stop near epoch 48
  for a plateau at 40); `--patience 0` runs all epochs.
- `--harvest-workers W` — parallelize the one-time harvest across `W` processes
  (`0` = all cores); it's CPU/NumPy-bound so it scales ~linearly, with a
  sequential fallback on any pool error. Each worker imports torch, so on a
  RAM-tight machine (the laptop) keep it at `1`; on the PC `6` is a good balance.
- `--bs` — larger batch (e.g. `512` on an 8 GB GPU) means fewer steps per epoch
  and better GPU use; the `B×N×N` pairwise tensor still fits.

### Stage 2 — PPO with rolling league + mixed opponents

Mix per non-focal slot, per episode: `--mix-*` are explicit weights, the
remainder goes to the league. Slots are drawn independently each episode, so 4p
games naturally vary in composition.

```
# ---- 2-player ----
python ppo.py --players 2 --init bc2_best.pt --opponent self `
  --league-size 12 --league-add-every 20 `
  --vf-coef 0.25 --lr 3e-5 --warmup-lr 1e-3 --warmup-cache ppo2_warmup.pt `
  --mix-teacher 0.15 `
  --mix-scripted defender:0.05 --mix-scripted most_production:0.05 `
  --mix-scripted comet_user:0.05 `
  --external rl_v1=..\RL\submission_orbitnet.py --mix-external rl_v1:0.15 `
  --eval-every 5 --eval-games 40 `
  --eval-opponent teacher --bench-also external:rl_v1 --bench-also nearest_planet `
  --iters 2000 --out ppo2.pt

# ---- 4-player (after bc4 finishes) ----
python ppo.py --players 4 --init bc4_best.pt --opponent self `
  --league-size 16 --league-add-every 20 `
  --vf-coef 0.25 --lr 3e-5 --warmup-lr 1e-3 --warmup-cache ppo4_warmup.pt `
  --mix-teacher 0.12 --mix-aggressive 0.08 --mix-weak 0.05 `
  --mix-scripted defender:0.05 --mix-scripted most_production:0.05 `
  --mix-scripted comet_user:0.05 `
  --external rl_v1=..\RL\submission_orbitnet.py --mix-external rl_v1:0.10 `
  --eval-every 5 --eval-games 40 `
  --eval-opponent teacher --bench-also external:rl_v1 --bench-also nearest_planet `
  --iters 2000 --out ppo4.pt
```

`--vf-coef 0.25 --lr 3e-5` are the stability settings for fine-tuning a
warm-started (BC-peaked) policy: the lower value coefficient softens the value
gradient that flows through the shared trunk into the policy, and the lower
learning rate keeps the peaked target logits from flipping. Combined with the
per-minibatch KL gate (`--target-kl`, default 0.03), they keep the run from
diverging. Once `kl` is stably under target and `bench` climbs, you can nudge
`--lr` back toward `1e-4` and `--vf-coef` toward `0.5` (the defaults) to speed up.

The value warmup trains **only the value head** (non-value grads are zeroed), so
it can't destabilize the policy and gets its **own** learning rate via
`--warmup-lr` (default `1e-3`), decoupled from the policy `--lr`. This matters
when you lower `--lr` for stability: at `3e-5` a 100-epoch warmup left the value
head under-calibrated (`value_loss ≈ 0.2` and still falling) because it inherited
the tiny policy lr; the dedicated `--warmup-lr` converges it regardless. A
poorly-calibrated value head means noisier advantages, so every KL-capped step
points in a worse direction — keep an eye on the final warmup `value_loss` and
raise `--value-warmup`/`--warmup-lr` if it hasn't flattened.

`--warmup-cache PATH` makes the one-shot value warmup reusable: the first run
collects the calibration games, trains the value head, and saves the result to
`PATH`; every later run with the same `PATH` **loads it and skips warmup**. The
warmup depends only on `--init` and the opponent mix — **not** on `--lr`,
`--vf-coef`, `--clip`, etc. — so a single cache is valid across an entire
hyperparameter sweep (the run stamps `init` + `mix` into the file and warns if
either changed). This is the recommended way to iterate: pay the warmup cost
once, then sweep lr/vf/clip for free. Delete the cache file to force a rebuild
(e.g. after retraining `bc2_best.pt` or changing the `--mix-*` weights).

The 2p mix per slot: 15% teacher, 5% defender (turtle), 5% most_production
(economic priority), 5% comet_user (comet-aware), 15% RL-v1 submission, 55%
league. The 4p mix: 12% teacher, 8% aggressive, 5% weak, 5% defender, 5%
most_production, 5% comet_user, 10% RL-v1, 50% league. The three
`--mix-scripted` agents are the only entries in `ow_base.py` that genuinely
differ from the teacher family — `defender` is the lone turtle, `most_production`
targets by production rather than ROI, and `comet_user` is the only scripted
agent that engages comet mechanics. The rest of the `net_attacker*` /
`net_roi_*` family are micro-iterations of the teacher and would dilute signal
without adding strategic information.

Before the first PPO step, a **baseline benchmark** runs so you can see where
the warm-start stands, and it seeds **every** opponent's best checkpoint with the
warm-start weights — a diverging run can no longer overwrite any `_best` with
something worse than where it began:

```
[baseline] bench vs teacher 0.45 [external:rl_v1 0.40  nearest_planet 0.85]  -> saved ppo2_best.pt, ppo2_best_external_rl_v1.pt, ppo2_best_nearest_planet.pt
```

**One best per benchmarked opponent.** Each opponent keeps its own rolling best
checkpoint, updated independently whenever the net beats its previous best vs
*that* opponent — so a policy that's great vs `nearest` but mediocre vs `teacher`
doesn't cost you the strong-vs-nearest checkpoint, and beating several opponents
in one tick saves several files:

- primary (`--eval-opponent`, e.g. teacher) → `ppoN_best.pt` (**the submit target**)
- each `--bench-also` opponent → `ppoN_best_<name>.pt`
  (`ppo2_best_external_rl_v1.pt`, `ppo2_best_nearest_planet.pt`)

Both runs also write a rolling `ppoN_itXXXX.pt` every `--save-every` iters. The
log line per eval tick looks like:

```
iter   42  mean_ep_reward +1.234  train_wr 0.62 `
           bench 0.55 [external:rl_v1 0.48  nearest_planet 0.92] `
           best_bench 0.57  kl 0.012  *saved ppo2_best.pt, ppo2_best_nearest_planet.pt
```

`bench` is vs teacher and `best_bench` tracks its best (the submit target). The
bracketed extras each drive their own `_best_<name>` file: rl_v1 = "am I beating
my own prior submission yet?", nearest = "am I still trivially handling weak
opponents, or did self-play break the basics?". `*saved ...` lists which best
files were written this tick.

`ev` is the value head's **explained variance** on the iter's collected states —
the single most diagnostic PPO health number. `ev ≤ 0` means the value baseline
explains none of the return variance, so the GAE advantages are noise and the
policy **cannot** learn no matter how you tune lr/ent/kl (this was the original
"stuck in phase-1" failure, caused by the unbounded reward — see the `shaping`
note in Knobs). With the normalized reward it should climb off ~0 within the
first few dozen iters as the value head warms in; >0.5 is healthy. If `ev` stays
pinned ≤0, the return is the problem, not the hyperparameters. Run `diag.py` for
the full reward/advantage breakdown.

**Eval-games sizing.** A win-rate over `n` games has resolution `1/n` and a
representativeness spread of `√(p(1−p)/n)`: at `n=20` that's 5 pp steps and
≈ ±11 pp; at `n=40`, 2.5 pp and ≈ ±8 pp. Since **every** benchmark now drives a
`_best` save, coarse eval makes selection lock onto lucky-seed checkpoints — so
`--eval-games 40` is the recommended floor (40→80 only shaves the spread another
~29 %, not worth the cost). The benchmark is deterministic for a fixed net
(greedy decode, scripted opponents, fixed eval-seed pool), so the issue is
resolution + small-sample representativeness, not run-to-run jitter.

Total eval cost per tick = `--eval-games × (1 + #bench-also)`. At
`--eval-games 40` with two extras that's 120 games per tick — note eval already
dominates wall-clock here (it's ~5× the 8 training episodes/iter at
`--eval-every 3`), so the default pairs `40` with **`--eval-every 5`** to keep
total eval cost (≈ 24 games/iter) close to the old `20 / every-3` setting while
getting finer resolution. Use `--eval-every 3` for more responsiveness (≈ 40
games/iter), drop `--eval-games` if you must pay less, or `--eval-every 1` for
per-iter granularity at ~120 games/iter.

**Where the wall-clock goes, and the batched path (measured, `prof.py` /
`bench_collect.py`).** The training loop now uses **batched forwards** everywhere:
collection steps all `episodes_per_iter` games in lockstep through one `B`-sized
focal forward (`collect_batched`), and the PPO/value update does one forward per
*minibatch* instead of one per state (`_batched_policy` / `_batched_value`). The
math is bit-equivalent to the old per-state path (`test_batched.py`: max |Δ| ~1e-5).
Measured on a 2p run:

| stage | per-state (old) | batched (new) | speedup |
|---|---|---|---|
| update, one 128-state minibatch | 1743 ms | 98 ms | **17.7×** |
| collection, 16 games | 61.9 s | 51.5 s | 1.20× |

The **update** was the hidden cost (128 batch-1 forward+backwards per minibatch ×
~20–40 minibatches/iter = 30–60 s/iter); batching it is the bulk of the first
per-iter win. **Collection** barely moved from batching because, once the focal
forward is batched, it is **CPU-bound** — game-sim + `encode`/`decode` Python, not
the GPU.

So collection *and* eval are parallelized across CPU workers (`parallel.py`,
`--workers N`; `0`=all cores, `1`=in-process). Each worker runs whole games on a CPU
copy of the net — the BC-harvest pattern — while the main process keeps the GPU for
the batched update. Eval is greedy + fixed-seed, so the parallel win-rate is
**bit-identical** to sequential (`test_parallel.py`). End-to-end on a 16-core box:

| | original | + batching | + workers (16) |
|---|---|---|---|
| per-iter (collect+update) | ~100–130 s | ~55 s | **~19 s** |
| eval tick | ~110–295 s | (same) | **~10–15 s** |

Net ~5–6× end to end. The old per-state path (`policy_act` / `collect_episode` /
`recompute_logp_value`) is kept for `diag.py` / `prof.py`. Phase-1 in the run
scripts also drops the (informational) teacher bench and uses `--eval-every 5
--eval-games 30 --workers 8`. (Further headroom: the league state-dicts are re-sent
to every worker each iter — caching them worker-side would cut per-iter IPC.)

### Stage 3 — Evaluation and inspection

```
python submit_agent.py --players 2 --model ppo2_best.pt --games 300 --opponent teacher
python submit_agent.py --players 4 --model ppo4_best.pt --games 300 --opponent teacher
python replay.py --players 2 --model ppo2_best.pt --opponent teacher --seed-idx 0
```

### Submission

```
$env:OW_MODEL_2P = "ppo2_best.pt"
$env:OW_MODEL_4P = "ppo4_best.pt"
# kaggle entry point: submit_agent.agent
```

### Back-compat sanity (reproduce a v1-style single-opp run)

With every new flag at its default, behavior is byte-identical to the prior
single-opponent path. Useful as a regression check after pulling changes.

```
python ppo.py --players 2 --init bc2_best.pt --opponent self --refresh 20 --iters 20
```

## Files

- `run_2p.ps1` / `run_4p.ps1` — one-command pipelines (BC -> PPO phase-1 ->
  PPO phase-2); see "One-command pipeline" above.
- `quiet_kaggle.py` — `from quiet_kaggle import make` instead of importing
  `kaggle_environments` directly; suppresses OpenSpiel's ~200-line import banner
  (the pyspiel C-extension writes to the OS stderr fd, so it's silenced via
  fd-level redirection during the import). Used by every module that needs `make`.
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
  target CE + Beta-NLL fraction loss; `--players {2,4}`. Harvest is cacheable
  (`--data-cache`), validation is minibatched, and training plateau-early-stops
  (`--patience`).
- `ppo.py` — Stage 2 PPO; Beta fraction, n-player rollouts, value warmup
  (cacheable via `--warmup-cache`), **per-minibatch KL gate**, minibatched
  updates. Opponents are drawn per slot per episode
  from a rolling self-snapshot league (`--league-size`), an optional scripted
  mix (`--mix-teacher/aggressive/weak`, plus any other `ow_base` callable via
  `--mix-scripted NAME:WEIGHT`), and any external `agent(obs)` callable
  registered via `--external NAME=PATH`. Defaults reproduce the single-opponent
  v1 path.
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
- `shaping` (OrbitEnv, default 1.0) — potential-style reward on the **normalized**
  lead `(my_score − best_opponent) / (my_score + best_opponent + 1)`, bounded in
  (−1, 1); the terminal ±5 (engine rule: highest positive ship total wins, ties
  win) dominates. **The normalization is load-bearing**, not cosmetic: the lead was
  previously the raw ship-count difference, which reached tens of thousands in long
  games, so the telescoped shaping hit ±100 and swamped the terminal ~20×. That
  made returns dominated by game length (long game ⇒ loss) rather than move quality,
  the value head could not fit them (explained variance ≈ 0), and PPO advantages
  were pure noise — the policy random-walked and never learned (the classic
  "stuck in phase-1" failure). Bounding the potential restores a predictable return
  where the terminal dominates. Verify health with `diag.py` (watch the **value
  explained variance** line: ≤0 ⇒ advantages are noise; >0.5 is healthy).
- `--league-size` (ppo.py, default `1`) — rolling FIFO pool of self-snapshots.
  `1` reproduces the old single-opp behavior; `>1` enables league play.
- `--league-add-every` (ppo.py, default `0`) — iters between appends. `0` falls
  back to `--refresh` (the legacy knob).
- `--mix-teacher`, `--mix-aggressive`, `--mix-weak` (ppo.py, all default `0.0`)
  — per-slot probabilities of the named scripted opponents
  (`net_roi_support`, `net_roi_aggressive`, `nearest_planet`). Sum across the
  whole mix must be ≤ 1; remainder goes to league.
- `--mix-scripted NAME:WEIGHT` (ppo.py, repeatable) — per-slot weight for any
  other callable in `ow_base.py` (e.g. `defender:0.05`, `most_production:0.05`,
  `comet_user:0.05`). Use these to add strategic shapes the teacher family
  doesn't cover.
- `--external NAME=PATH` + `--mix-external NAME:WEIGHT` (ppo.py, repeatable) —
  register a `.py` file's `agent(obs)` callable as a sparring partner with a
  given slot weight. Also usable as `--eval-opponent external:NAME` or
  `--bench-also external:NAME`.
- `--bench-also NAME` (ppo.py, repeatable) — extra benchmark opponent shown
  alongside the primary every eval tick. Same syntax as `--eval-opponent`
  (`teacher`, `aggressive`, `external:NAME`, or any `ow_base` callable). Each
  drives its **own** `ppoN_best_<name>.pt`; only the primary `--eval-opponent`
  writes the canonical `ppoN_best.pt` (the submit target).
- `--warmup-lr` (ppo.py, default `1e-3`) — LR for the value-head warmup only;
  decoupled from `--lr` so a low policy lr doesn't under-calibrate the value head.
- `--ent` (ppo.py, default `0.01`) — entropy bonus, applied as a **per-owned-node
  mean** (normalized by the focal player's owned-planet count, matching the policy
  loss). It is therefore board-size independent: `0.01` means the same exploration
  pressure whether the player owns 3 planets or 30. See the lessons note below —
  before the normalization fix this term was a raw *sum*, so the effective weight
  was `ent × n_own` and even `0.01` quietly behaved like ~`0.1`. Even normalized,
  a non-trivial `--ent` will randomize the policy whenever the advantage signal is
  weak (it's then the only coherent gradient). The pipeline runs phase-1 at
  `--ent 0.0` (warm-started from BC, no exploration needed; the winnable
  `--mix-weak` supplies the gradient) and phase-2 at a small `--ent 0.003` floor
  (a long, diverse run where a touch of entropy guards against premature collapse).

## Hard-won lessons

Carried from v1: clip the PPO ratio **per planet**, not on the joint action
(the joint ratio is `exp` of a sum of N differences and explodes); step the
optimizer **once per minibatch**, not once per state; **warm up the value head**
before letting advantages move the policy; in self-play the training win-rate is
pinned near 0.5 and is meaningless — **select `_best` on a greedy benchmark vs
a fixed stable opponent** (teacher), but **watch additional benchmarks
alongside it** (rl_v1 = "am I beating my own prior submission?", nearest = "am
I still trivially handling weak opponents?"). A single number hides regressions
and stalls that show up clearly across a difficulty ladder.

New in v2 (divergence fix): **gate KL per minibatch, not per epoch.** The old
check only fired after a full epoch (~batch/minibatch optimizer steps), by which
point a warm-started — and therefore very peaked — BC policy had already drifted
to KL ≫ `target_kl`. Now `target_kl` is checked after every minibatch and stops
the moment cumulative drift from the collection policy exceeds it, bounding the
update to ~`target_kl`. Two amplifiers make a warm-started policy fragile here:
(1) BC-peaked target logits flip on tiny weight moves, and (2) the **value head
shares the transformer trunk**, so a large early value loss yanks the policy
through the shared encoder — lower `--vf-coef` (e.g. 0.25) and/or `--lr` (e.g.
3e-5) for the first fine-tuning phase if KL is erratic.

**Normalize the entropy bonus the same way as the policy loss.** The policy loss
is a *mean* over owned nodes (`sum / denom`), but the entropy term was a raw *sum*
over owned nodes — so the effective entropy weight was `ent × n_own` and grew with
the board. With a player owning ~10–15 planets, a nominal `ent=0.05` acted like
~0.5, overwhelming the (normalized, noisy) advantage signal: a single step blew
past `target_kl`, and the only coherent gradient left was "increase entropy." The
tell is **monotone** collapse from iter 0 — `entH` rising, `hold%` and `conc`
falling together, `train_wr`→0, and the benchmark never once beating the warm
start (a value/advantage problem degrades *noisily* instead). Fix: divide the
entropy sum by the owned-node count in `recompute_logp_value`, so `--ent` is a
true per-node coefficient. Lowering `--ent` alone only masks this — the scale
still grows with the board, which bites hardest in phase-2 when the focal player
controls many planets.

Expect `early-stop @ep1` on (almost) every iter, and that's *healthy*, not a bug:
a peaked policy saturates the `target_kl` trust region partway through the first
pass, so it stops mid-epoch-0 and discards the rest of the collected data. The
reported `kl` looks small (≈ the mean of the sub-threshold minibatches that were
applied; the one that crossed `target_kl` triggers the stop and isn't averaged
in) — occasionally even slightly negative, which is just the crude per-source
estimator's sign noise. `--epochs` is therefore a *ceiling*, not a target: it
only binds once the policy is less peaked or you widen the trust region. **The
per-iter learning budget is `target_kl`, not `--lr` or `--epochs`** — once the
gate is the binding constraint, raising `--lr` just reaches the cap in fewer
minibatches; to actually move faster per iter, raise `--target-kl` (now safe
because it's gated). If reward stays flat with KL pinned at target, the limiter
is the opponent mix difficulty or value/advantage quality, not the step size.

New in v2: the engine rotates planets on the **pre-increment** step (fix your
simulator's tick order or everything drifts); **comets follow paths, not
orbits**, and must spawn/expire to be faithful; **kaggle fills agent positional
args by count**, so any agent with extra parameters gets the framework's config
object bound to them — agents must be single-arg; with a continuous fraction the
**fraction log-prob must receive PPO credit on every non-HOLD source** (v1 only
counted ferries because captures ignored the fraction).

On opponent diversity: a **single frozen self-opponent is a trap**. The PPO
policy will specialize against whatever it sees, and when the snapshot refreshes
the strategy shatters — that's the v1 collapse from `bench 0.50` to `0.40`. The
fix is to face a *distribution* of opponents from iter 0: a rolling pool of
past selves, plus enough scripted/external weight to anchor the policy to the
real objective. The mix can also be **too narrow**: if 80% of slots land on a
weak agent, the policy stops needing to defend and brittles in a different
direction. Keep most weight on near-skill (league + teacher), small weight on
adversarial probes (the prior submission), and only dial up `weak` in 4p where
"go after the weakest seat" is a real strategy. Per-slot draws must be
**independent** in n-player games — putting three identical opponents at one
table is unrealistic and trains a single narrow counter.

## Not yet ported

The v1 diagnostics (`verify_fix`, `diagnose`, `hold_bias`, `why_losing`,
`capture_check`, `cuda_check`) reference the removed discrete-fraction constants
(`FRACS` / `N_FRAC`) and `capture_size`, so they need updating for the continuous
head before they'll run. Say the word and I'll port them.
