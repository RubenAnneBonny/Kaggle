# Orbit Wars — Learned Agent Pipeline (BC → PPO self-play)

A graph-attention policy that learns to play orbit_wars. Two training stages:
**behavioral cloning** of `net_roi_support` (warm start), then **PPO self-play**
to surpass it.

## What's here
| file | role | needs torch |
|---|---|---|
| `ow_base.py` | your existing helpers + `net_roi_support` (the teacher) | no |
| `ow_env.py` | state encoder, action decoder, fast self-play env | no |
| `search_agent.py` | engine-faithful simulator (`sim_step`) reused by `ow_env` | no |
| `model.py` | graph-attention policy + value network | **yes** |
| `bc.py` | Stage 1: harvest teacher decisions + train | **yes** |
| `ppo.py` | Stage 2: PPO self-play fine-tuning | **yes** |
| `submit_agent.py` | load a checkpoint and play / evaluate | **yes** |

## Design (why these choices)
- **Graph attention, not a CNN.** The state is ~32 planets at continuous
  positions; what matters is pairwise relationships (who can reach whom, garrison
  ratios), not pixel locality. `model.py` embeds each planet as a node and runs
  transformer-encoder layers with a **distance bias** added to attention logits,
  so geometry is explicit.
- **Factored per-source action.** Each owned planet chooses ONE target (or HOLD)
  plus a send-fraction bucket. This matches `net_roi_support` (one attack per
  planet/turn) and keeps the policy log-prob tractable.
- **The network never outputs firing angles.** It outputs *target* + *how many*;
  `ow_env.decode_action` reuses `reach()` to solve the intercept geometry. The
  policy only learns the strategic part.
- **Reward** = production-weighted material differential (dense shaping) +
  terminal win/loss, computed in `ow_env.OrbitEnv` on the faithful simulator
  (verified to match the real engine exactly over 9 ticks; note the simulator
  ignores comets for speed — acceptable for training, and the final agent plays
  the real env with comets via `submit_agent.py`).

## Setup
```bash
pip install torch kaggle_environments numpy
# put all files in one directory
python model.py        # shape/backward sanity — should print tensor shapes + param count
```

## Stage 1 — Behavioral cloning (`bc.py`)
Harvest teacher decisions and imitate them.
```bash
python bc.py --games 300 --epochs 15 --out bc.pt
```
Watch `val_target_acc` climb — that's how often the net picks the same target
`net_roi_support` would (per-source argmax). Aim for >0.7 before Stage 2.

**Flags:**
| flag | default | what it does |
|---|---|---|
| `--games` | 200 | number of teacher games to harvest training data from. More = more data, slower harvest. |
| `--epochs` | 15 | passes over the harvested dataset. More = more fitting (watch for val_loss flattening/overfitting). |
| `--bs` | 128 | minibatch size (states per gradient step). Bigger = smoother, more memory. |
| `--lr` | 3e-4 | Adam learning rate. Lower if loss is unstable; higher to learn faster (riskier). |
| `--out` | `bc.pt` | path to write the checkpoint (weights + optimizer state). Overwrites if it exists. |
| `--resume` | `""` | path to an existing checkpoint to **continue** training from (loads weights + optimizer momentum). Omit to start from random weights. |
| `--seed-offset` | 0 | first game seed to harvest (seeds run `offset .. offset+games-1`). **Bump this between runs to harvest different games**, otherwise repeated runs see the same matches. |

**Continue training across multiple runs (accumulates, fresh data each time):**
```bash
python bc.py --games 300 --seed-offset 0   --out bc.pt
python bc.py --games 300 --seed-offset 300 --resume bc.pt --out bc.pt
python bc.py --games 300 --seed-offset 600 --resume bc.pt --out bc.pt
```
Without `--resume`, each run starts from scratch and overwrites — you'd lose prior
training. Without a changing `--seed-offset`, each run sees identical games.

Sanity-check the cloned policy in the real environment:
```bash
python submit_agent.py --ckpt bc.pt --games 50
```
A good clone lands near ~50% vs `net_roi_support` (it's imitating it).

## Stage 2 — PPO self-play (`ppo.py`)
Fine-tune to beat the teacher. Warm-start from the BC checkpoint:
```bash
python ppo.py --init bc.pt --iters 2000 --out ppo.pt
```
`mean_ep_reward` should trend upward. The opponent is a frozen copy of the policy,
refreshed every `--refresh` iters (self-play).

**Flags:**
| flag | default | what it does |
|---|---|---|
| `--init` | `bc.pt` | checkpoint to warm-start the policy from (use your BC output). Set `""` to start from random weights (much harder; not recommended). |
| `--iters` | 1000 | number of PPO iterations (collect-then-update cycles). The main "how long to train" knob. |
| `--episodes_per_iter` | 8 | self-play games collected per iteration before each update. Bigger = more stable gradients, slower. |
| `--epochs` | 4 | optimization passes over each batch of collected data (standard PPO reuse). |
| `--lr` | 1e-4 | Adam learning rate. Lower than BC because we're fine-tuning; drop further if it destabilizes. |
| `--clip` | 0.2 | PPO clip ratio — caps how far the policy moves per update. Standard; rarely change. |
| `--ent` | 0.01 | entropy bonus. **Raise** (e.g. 0.02–0.05) if the policy collapses to one move; lower for more exploitation. |
| `--refresh` | 20 | iterations between copying the current policy into the frozen opponent. Smaller = opponent keeps pace (harder, can be unstable); larger = more stable target. |
| `--max_steps` | 200 | max ticks per self-play episode before it's cut off. |
| `--out` | `ppo.pt` | path to write the fine-tuned checkpoint. |

To train against `net_roi_support` directly instead of self-play, swap the
opponent call in `ppo.collect_episode` (see the note at the bottom of this file).

Evaluate with YOUR large-sample rig (300+ games):
```bash
python submit_agent.py --ckpt ppo.pt --games 300
```

## Using / evaluating a checkpoint (`submit_agent.py`)
Wraps a trained checkpoint into a playable `agent(obs)` and self-evaluates it
against `net_roi_support` (alternating sides, prints win rate).

**Flags:**
| flag | default | what it does |
|---|---|---|
| `--ckpt` | `ppo.pt` | checkpoint to load and play with. |
| `--games` | 50 | number of evaluation games vs `net_roi_support`. Use 300+ for a trustworthy number. |

For Kaggle submission, the `agent` function in this file is the entry point;
upload it with the weights file and the modules it imports (`model.py`,
`ow_env.py`, `ow_base.py`, `search_agent.py`).

## Tips
- BC quality caps PPO's starting point — get target_acc high first.
- If PPO degrades vs `net_roi_support`, lower `--lr`, raise `--ent`, or refresh
  the opponent less often. Always validate at 300+ games (small samples lie —
  we learned this the hard way).
- To also train against `net_roi_support` directly (not just self-play), swap the
  opponent in `ppo.collect_episode` to call `net_roi_support(env.obs_for(1))`.
- The action space currently allows one attack per planet/turn. If you want
  multi-target splits, extend the target head to top-k with per-target fractions
  (bigger action space, needs more training).
```
