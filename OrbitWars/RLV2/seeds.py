"""seeds.py — the ONE source of truth for which seeds are training vs evaluation.

The v1 leak happened because BC, PPO rollouts and evaluation all defaulted to
seeds 0,1,2,... so they silently overlapped. Here we partition the seed line
into two *disjoint* blocks and force every script to draw from the right one:

    train seeds : [TRAIN_BASE, TRAIN_BASE + 1_000_000)
    eval  seeds : [EVAL_BASE,  EVAL_BASE  + 1_000_000)

The blocks are 1e6 apart, so a training run can harvest/roll out as many games
as it will ever need without colliding with the evaluation block. Scripts call
`train_seed(i)` / `eval_seed(i)` instead of passing raw ints, which makes a
train/eval overlap structurally impossible rather than a thing you must remember.

`assert_disjoint()` is called by the eval scripts as a tripwire.
"""

TRAIN_BASE = 0
EVAL_BASE = 5_000_000          # 5e6 gap; nothing trains anywhere near here
BLOCK = 1_000_000


def train_seed(i: int) -> int:
    """i-th training seed (BC harvest, PPO self-play, value warmup)."""
    return TRAIN_BASE + (int(i) % BLOCK)


def eval_seed(i: int) -> int:
    """i-th evaluation seed (submit_agent, benchmark, replay). Never trained on."""
    return EVAL_BASE + (int(i) % BLOCK)


def is_train_seed(s: int) -> bool:
    return TRAIN_BASE <= s < TRAIN_BASE + BLOCK


def is_eval_seed(s: int) -> bool:
    return EVAL_BASE <= s < EVAL_BASE + BLOCK


def assert_disjoint():
    assert EVAL_BASE >= TRAIN_BASE + BLOCK, "train/eval seed blocks overlap!"


assert_disjoint()
