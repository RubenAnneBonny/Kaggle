"""Import `kaggle_environments.make` with the OpenSpiel import banner silenced.

Importing kaggle_environments loads OpenSpiel, whose pyspiel C-extension prints
~200 lines of "Unknown game ... Available games are: ..." plus INFO logs. Those
come from the native layer writing to the OS stderr fd, so a Python-level
contextlib.redirect_stderr does NOT catch them — we redirect fds 1/2 at the OS
level for the duration of the import, and raise the kaggle logger above INFO.

All modules should do `from quiet_kaggle import make` instead of importing
kaggle_environments directly; the suppression then happens exactly once per
process (this module is cached after first import).
"""
import os, sys, logging

logging.getLogger("kaggle_environments").setLevel(logging.WARNING)

try:
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _saved_out, _saved_err = os.dup(1), os.dup(2)
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(_devnull, 1); os.dup2(_devnull, 2)
    try:
        from kaggle_environments import make            # noqa: E402  (suppressed import)
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(_saved_out, 1); os.dup2(_saved_err, 2)
        os.close(_devnull); os.close(_saved_out); os.close(_saved_err)
except Exception:
    # If fd redirection isn't available (odd stdio), fall back to a plain import
    # (accept the banner) rather than failing.
    from kaggle_environments import make                # noqa: E402

__all__ = ["make"]
