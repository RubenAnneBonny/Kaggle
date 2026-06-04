"""verify_fix.py — confirm the updated decode_action with capture_size + friendly
reinforcement is actually the code running on THIS machine. If this prints the
'OLD' diagnosis, your local files are stale (re-download) or .pyc cache is stale.
"""
import inspect, numpy as np, contextlib, io
from kaggle_environments import make
from ow_base import net_roi_support
import ow_env
from ow_env import encode_state, decode_action, N_MAX

src = inspect.getsource(decode_action)
has_capture = "capture_size" in src
has_friendly = "dst_is_mine" in src
print("ow_env loaded from:", ow_env.__file__)
print("decode_action has capture_size param:", has_capture)
print("decode_action has friendly-reinforcement branch:", has_friendly)

if not (has_capture and has_friendly):
    print("\n>>> OLD CODE IS RUNNING. Your local ow_env.py is stale.")
    print(">>> Re-download ow_env.py, and delete any __pycache__ folder, then retry.")
else:
    # show that capture_size changes the actual ships sent vs the old fraction path
    e = make("orbit_wars", configuration={"seed": 0, "episodeSteps": 60}, debug=False)
    with contextlib.redirect_stderr(io.StringIO()):
        e.run([net_roi_support, net_roi_support])
    obs = dict(e.steps[55][0]["observation"]); obs["player"] = 0
    enc = encode_state(obs, 0); ids = enc["ids"]
    planets = {p[0]: p for p in obs["planets"]}
    tgt = np.full(N_MAX, N_MAX, np.int64); frac = np.zeros(N_MAX, np.int64)  # frac=0 ->0.25
    for i in range(N_MAX):
        if enc["own_mask"][i]:
            legal = np.where(enc["attack_mask"][i] > 0)[0]
            if len(legal):
                si = planets[int(ids[i])]
                tgt[i] = min(legal, key=lambda j: (planets[int(ids[j])][2]-si[2])**2 + (planets[int(ids[j])][3]-si[3])**2)
    old = decode_action(enc, obs, 0, tgt, frac, capture_size=False)
    new = decode_action(enc, obs, 0, tgt, frac, capture_size=True)
    print("\nNEW CODE IS RUNNING.")
    print("  capture_size=False ships:", sorted(m[2] for m in old))
    print("  capture_size=True  ships:", sorted(m[2] for m in new))
    print("  (if these differ, the fix is active and the eval should reflect it)")
