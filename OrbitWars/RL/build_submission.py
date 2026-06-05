"""build_submission.py — assemble a SINGLE self-contained Kaggle submission file
for the trained OrbitNet agent.

Kaggle wants one .py file that defines `agent(obs)` and has no local imports.
This script inlines your real modules (ow_base.py, model.py, the inference half
of ow_env.py) and embeds the checkpoint weights as a base64 blob, then writes
one file you can submit directly.

Run it in your venv, from the folder that has ow_base.py / model.py / ow_env.py
and the checkpoint:

    python build_submission.py --ckpt ppo_night_c1_best.pt --out submission_orbitnet.py

Then submit submission_orbitnet.py.

IMPORTANT — verify two things before trusting the result:
  1. The Kaggle orbit_wars runtime must have PyTorch. Check in a competition
     notebook: `import torch`. If that fails, this agent cannot run on Kaggle and
     net_roi_support stays your submission.
  2. Smoke-test the built file locally:  python submission_orbitnet.py
     It should report the checkpoint loaded and a sane win-rate vs net_roi_support.
"""
import argparse, base64, os, re, sys


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def strip_import_lines(src, patterns):
    """Remove whole lines that match any regex in `patterns`."""
    out = []
    for line in src.splitlines():
        if any(re.match(p, line) for p in patterns):
            continue
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ppo_night_c1_best.pt")
    ap.add_argument("--out", default="submission_orbitnet.py")
    ap.add_argument("--owbase", default="ow_base.py")
    ap.add_argument("--model", default="model.py")
    ap.add_argument("--owenv", default="ow_env.py")
    args = ap.parse_args()

    for p in (args.ckpt, args.owbase, args.model, args.owenv):
        if not os.path.exists(p):
            sys.exit(f"ERROR: required file not found: {p}")

    # ---- 1) ow_base.py: replace the kaggle Planet import with a local namedtuple ----
    owbase = read(args.owbase)
    owbase = owbase.replace(
        "from kaggle_environments.envs.orbit_wars.orbit_wars import Planet",
        'Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", '
        '"ships", "production"])',
    )
    # drop any stray kaggle/search/torch imports just in case (ow_base is torch-free)
    owbase = strip_import_lines(owbase, [
        r"\s*from kaggle_environments", r"\s*import kaggle_environments",
        r"\s*from search_agent", r"\s*import torch",
    ])

    # ---- 2) model.py: keep everything up to the __main__ self-test ----
    model = read(args.model)
    cut = model.find('if __name__ == "__main__"')
    if cut != -1:
        model = model[:cut]

    # ---- 3) ow_env.py: take only the inference half (everything BEFORE OrbitEnv),
    #         and strip its imports (those names are provided by the header + ow_base).
    owenv = read(args.owenv)
    cut = owenv.find("class OrbitEnv")
    if cut == -1:
        sys.exit("ERROR: couldn't find 'class OrbitEnv' in ow_env.py — file changed?")
    owenv = owenv[:cut]
    owenv = strip_import_lines(owenv, [
        r"\s*import math", r"\s*from kaggle_environments", r"\s*from ow_base",
        r"\s*from search_agent", r"\s*import contextlib",
    ])

    # ---- 4) the checkpoint, base64-embedded ----
    with open(args.ckpt, "rb") as f:
        blob = base64.b64encode(f.read()).decode("ascii")

    header = (
        '"""Self-contained orbit_wars submission (trained OrbitNet).\n'
        "Built by build_submission.py — do not edit by hand.\n"
        '"""\n'
        "import math, io, base64\n"
        "from collections import namedtuple\n"
        "import numpy as np\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
    )

    footer = f'''
# ===================== trained weights (base64) =====================
_WEIGHTS_B64 = "{blob}"

_NET = None
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    global _NET
    buf = io.BytesIO(base64.b64decode(_WEIGHTS_B64))
    state = torch.load(buf, map_location=_DEV)
    net = OrbitNet().to(_DEV)
    net.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    net.eval()
    _NET = net
    return net


def agent(obs):
    """Kaggle entry point. Greedy OrbitNet with capture-sizing; falls back to
    net_roi_support on any error so it never forfeits."""
    global _NET
    try:
        if _NET is None:
            _load()
        player = obs["player"] if isinstance(obs, dict) else obs.player
        enc = encode_state(obs, player)
        if enc["own_mask"].sum() == 0:
            return []
        nf = torch.tensor(enc["node_feats"]).unsqueeze(0).to(_DEV)
        nm = torch.tensor(enc["node_mask"]).unsqueeze(0).to(_DEV)
        am = torch.tensor(enc["attack_mask"]).unsqueeze(0).to(_DEV)
        with torch.no_grad():
            tl, fl, _ = _NET(nf, build_edge(nf), nm, am)
        tgt = tl[0].argmax(-1).cpu().numpy()
        frac = fl[0].argmax(-1).cpu().numpy()
        return decode_action(enc, obs, player, tgt, frac, capture_size=True)
    except Exception:
        try:
            return net_roi_support(obs)
        except Exception:
            return []


if __name__ == "__main__":
    import contextlib
    from kaggle_environments import make
    print("device:", _DEV, "| weights bytes:", len(base64.b64decode(_WEIGHTS_B64)))
    wins = 0; N = 20
    for i in range(N):
        order = [agent, net_roi_support] if i % 2 == 0 else [net_roi_support, agent]
        ai = 0 if i % 2 == 0 else 1
        e = make("orbit_wars", configuration={{"seed": i}}, debug=False)
        with contextlib.redirect_stderr(io.StringIO()):
            e.run(order)
        r = [s.reward if s.reward is not None else 0 for s in e.steps[-1]]
        wins += r[ai] > r[1 - ai]
    print(f"built agent vs net_roi_support: {{wins}}/{{N}} = {{wins/N*100:.1f}}% (quick check)")
'''

    parts = [
        header,
        "\n# ===================== ow_base.py (inlined) =====================\n",
        owbase,
        "\n# ===================== model.py (inlined) =====================\n",
        model,
        "\n# ============ ow_env.py inference half (inlined) ============\n",
        owenv,
        footer,
    ]
    out_src = "\n".join(parts)

    # parse-check before writing
    import ast
    try:
        ast.parse(out_src)
    except SyntaxError as e:
        sys.exit(f"ERROR: assembled file has a syntax error: {e}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_src)
    kb = len(out_src) / 1024
    print(f"wrote {args.out}  ({kb:.0f} KB, weights {len(blob)/1024:.0f} KB base64)")
    print("next: python", args.out, " # smoke-test it plays before submitting")


if __name__ == "__main__":
    main()
