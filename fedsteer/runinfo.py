"""Unique output names and provenance for runs and evaluations.

Every training run gets its own directory ``<out_dir prefix>_<stamp>`` and every
evaluation its own file, so nothing overwrites earlier results.  The stamp is
``YYYYmmdd-HHMMSS`` plus ``_j<SLURM_JOB_ID>`` inside a SLURM job, which links each
output to its job log.  A job script can fix the stamp in advance with
``FEDSTEER_STAMP`` so it knows the run directory before training starts.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import socket
import subprocess
import sys


def make_stamp() -> str:
    if os.environ.get("FEDSTEER_STAMP"):
        return os.environ["FEDSTEER_STAMP"]
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job = os.environ.get("SLURM_JOB_ID")
    return f"{stamp}_j{job}" if job else stamp


def git_info(repo: str = ".") -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def provenance(**extra) -> dict:
    return {
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "git": git_info(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        **extra,
    }


def record_run_info(run_dir: str, event: str, **extra) -> None:
    """Append a provenance entry ('created' / 'resumed' / ...) to run_dir/run_info.json."""
    path = os.path.join(run_dir, "run_info.json")
    info = json.load(open(path)) if os.path.exists(path) else {"events": []}
    info["events"].append({"event": event, **provenance(**extra)})
    with open(path, "w") as f:
        json.dump(info, f, indent=1)
