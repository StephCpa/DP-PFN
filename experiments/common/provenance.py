"""Provenance helpers shared by all experiments.

Every artifact written under artifacts/ carries a provenance block so that a
number in the report can be traced to the script, commit and seed that made it.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # pragma: no cover - provenance must never crash a run
        return "unknown"


def provenance(script: str, seed: int | None = None, extra: dict | None = None) -> dict:
    libs = {}
    for name in ("numpy", "scipy", "torch", "sklearn", "networkx"):
        try:
            mod = __import__(name)
            libs[name] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    block = {
        "script": os.path.relpath(script, REPO_ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "utc_time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": libs,
        "seed": seed,
        "argv": sys.argv[1:],
    }
    if extra:
        block.update(extra)
    return block


def artifact_dir(exp_id: str) -> Path:
    path = ARTIFACTS / exp_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(exp_id: str, name: str, payload: dict, script: str, seed: int | None = None) -> Path:
    out = artifact_dir(exp_id) / name
    payload = {"provenance": provenance(script, seed), **payload}
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return out
