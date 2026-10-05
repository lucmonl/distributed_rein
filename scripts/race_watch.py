"""Race the same jobs on several clusters and cancel the losers once a winner RUNS.

    nohup setsid python scripts/race_watch.py --spec exp_log/launch/MATH-2_race.json \
        --log sbatch/logs/race_MATH-2.log >/dev/null 2>&1 &

Implements the user's rule (memory feedback-cluster-failover, 2026-10-04): queueing is the main
bottleneck, so submit everywhere, and cancel the other copies ONLY once one copy is actually
RUNNING -- never while all are pending, and never a copy that has started.

Spec (JSON):
  {"hosts": {"cc": null, "dtai": "dtai-1"},           # null = local squeue, else ssh host
   "races": [
     {"name": "G0",  "copies": {"cc": "111", "dtai": "222"}},
     {"name": "G2",  "copies": {...}, "dependents": {"cc": ["G1 cc id"], "dtai": ["G1 dtai id"]}},
     {"name": "E1fed", "group": "E1", "copies": {...}},  # a group is decided together: the host on
     {"name": "E1local", "group": "E1", "copies": {...}} # which ANY member starts first gets ALL
   ]}

Hosts listed in "preemptible" (e.g. cc's scavenger partition, where a started job can be
preempted and requeued from round 0) never win by STARTING -- only by COMPLETING; their pending
copies are cancelled like any other loser once a non-preemptible copy starts.

Every poll re-reads each job's state; a copy is cancelled only if a fresh check right before
`scancel` still says PENDING. Copies on a host that fails to answer are left alone. All actions
are appended to --log. Exits when no race has more than one live copy, or after --max_hours.
"""
import argparse, json, subprocess, time
from datetime import datetime

STARTED = {"RUNNING", "COMPLETING", "COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def run(host, cmd, timeout=90):
    full = cmd if host is None else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, " ".join(cmd)]
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout


def state(host, jobid, with_elapsed=False):
    """SLURM state of one job (and elapsed seconds), or None if the host did not answer."""
    try:
        rc, out = run(host, ["sacct", "-j", str(jobid), "-X", "-n", "-P", "-o", "State,ElapsedRaw"])
    except Exception:
        return (None, 0) if with_elapsed else None
    if rc != 0 or not out.strip():
        return (None, 0) if with_elapsed else None
    st, _, el = out.strip().splitlines()[0].partition("|")
    st = st.split()[0].rstrip("+") if st.strip() else None
    return (st, int(el or 0)) if with_elapsed else st


def log(path, msg):
    with open(path, "a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max_hours", type=float, default=96)
    ap.add_argument("--min_running_s", type=int, default=0,
                    help="a copy wins only after RUNNING this long (past model load and round 1), so a "
                         "fresh setup that crashes in its first minutes does not cost the other copies")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    hosts = spec["hosts"]
    races = spec["races"]
    preemptible = set(spec.get("preemptible", []))
    cancelled = set()                    # (host, jobid) cancelled by this watcher
    log(args.log, f"start: {len(races)} races on hosts {list(hosts)}")
    t_end = time.time() + 3600 * args.max_hours

    def cancel(host, jobid, why):
        if (host, jobid) in cancelled:
            return
        s = state(hosts[host], jobid)     # fresh check right before cancelling
        if s != "PENDING":
            log(args.log, f"skip cancel {host}:{jobid} ({why}): state is {s}")
            return
        if not args.dry_run:
            run(hosts[host], ["scancel", str(jobid)])
        cancelled.add((host, jobid))
        log(args.log, f"CANCEL {host}:{jobid} ({why})" + (" [dry run]" if args.dry_run else ""))

    while time.time() < t_end:
        st, el = {}, {}
        for r in races:
            for h, j in r["copies"].items():
                if (h, j) in cancelled:
                    st[(h, j)], el[(h, j)] = "CANCELLED", 0
                else:
                    st[(h, j)], el[(h, j)] = state(hosts[h], j, with_elapsed=True)
        # decide per group (a race without a group is its own group)
        groups = {}
        for r in races:
            groups.setdefault(r.get("group", r["name"]), []).append(r)
        live_races = 0
        for g, members in groups.items():
            winner = None
            for r in members:
                for h, j in r["copies"].items():
                    if h in preemptible:
                        won = st[(h, j)] == "COMPLETED"
                    elif st[(h, j)] in ("RUNNING", "COMPLETING"):
                        won = el[(h, j)] >= args.min_running_s
                    else:
                        won = st[(h, j)] == "COMPLETED"   # a copy that FAILED early must not win
                    if won:
                        winner = winner or h
            if winner is None:
                live_races += sum(1 for r in members
                                  if sum(st[(h, j)] in ("PENDING", None) for h, j in r["copies"].items()) > 1)
                continue
            for r in members:
                for h, j in r["copies"].items():
                    if h != winner and st[(h, j)] == "PENDING":
                        cancel(h, j, f"{r['name']}: group {g} started on {winner}")
                        for dep in r.get("dependents", {}).get(h, []):
                            cancel(h, dep, f"dependent of cancelled {r['name']} copy")
        if live_races == 0 and all(
                sum(st[(h, j)] in ("PENDING", None) for h, j in r["copies"].items()) <= 1 for r in races):
            log(args.log, "all races decided; exiting")
            return
        time.sleep(args.interval)
    log(args.log, "max_hours reached; exiting")


if __name__ == "__main__":
    main()
