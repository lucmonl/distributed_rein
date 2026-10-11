"""Std of the E1 steering metrics by test-set subsampling (CPU only; no training, no generation).

    python scripts/subsample_std.py --ref ARM \\
        --arm ARM=runs/<run>/evals/eval_round_0100__<stamp>.json \\
        --arm B4=runs/<run>/evals/eval_round_0100_b4_caa_L8_client__<stamp>.json ... \\
        [--seeds 0,1,2] [--frac 0.75] [--json out.json]

Protocol (CAL-31): every arm's SAVED test eval is reused (per-client scores and outputs per article
and alpha). For each seed, each client keeps `frac` of its test articles, drawn without replacement
by article (an article keeps all its alphas), stratified per client. The subsets are drawn ONCE per
seed from the article ids and applied to every arm, so differences between arms are paired.

The script refuses to run unless every arm has the same clients and the same test article ids per
client (`record_ids`) and the same alpha reference. Arms are given as explicit eval FILES, never
globs (run directories can also hold smoke-test evals with a few prompts).

Metrics are recomputed on each subset with fedsteer.metrics' own functions: metrics_for_client
(with the support stored in the eval, as compare_runs does), summarize (mean and worst client),
text_tie_metrics (near-tie). The per-alpha cells |pct - alpha| at alpha = 0 and 1 use the same
per-article formula as compare_runs.per_article. The full-test value is printed next to each
mean +- std as a check against the reported numbers.

The std is over the subsets: label it "<n> subsets x <frac> of test articles". With overlapping
subsets it understates the full-test sampling std (about 0.58x at 75%).
"""
import argparse, json, os, sys, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.data import alpha_reference_from_json  # noqa: E402
from fedsteer.metrics import metrics_for_client, summarize, text_tie_metrics  # noqa: E402

KEYS = [  # (label, how to get it from one subset's per-client results)
    ("pct_err", lambda S, T, C: S["pct_calib_err"]["mean"]),
    ("pct_err_worst", lambda S, T, C: S["pct_calib_err"]["worst"]),
    ("in_support", lambda S, T, C: S["pct_err_in_support"]["mean"]),
    ("out_support", lambda S, T, C: S["pct_err_out_support"]["mean"]),
    ("out_support_worst", lambda S, T, C: S["pct_err_out_support"]["worst"]),
    ("reach", lambda S, T, C: S["reach_rate"]["mean"]),
    ("spearman", lambda S, T, C: S["spearman"]["mean"]),
    ("spearman_worst", lambda S, T, C: S["spearman"]["worst"]),
    ("near_tie", lambda S, T, C: float(np.mean(T))),
    ("cell_a0", lambda S, T, C: float(np.mean([c[0] for c in C]))),
    ("cell_a1", lambda S, T, C: float(np.mean([c[-1] for c in C]))),
]


PER_CLIENT_KEYS = ("pct_calib_err", "pct_err_in_support", "pct_err_out_support", "reach_rate", "spearman")


def draw_subsets(n: dict, seeds, frac):
    """Per seed and client, `frac` of the client's articles (row indices), drawn without replacement.
    Deterministic in (seed, client name): every script and arm gets the same subsets."""
    out = {}
    for s in seeds:
        out[s] = {}
        for c in sorted(n):
            rng = np.random.default_rng([s, zlib.crc32(c.encode())])
            out[s][c] = np.sort(rng.choice(n[c], size=int(round(frac * n[c])), replace=False))
    return out


def load(path):
    ev = json.load(open(path))
    run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    ref_json = json.load(open(os.path.join(run, "alpha_reference.json")))
    return ev, ref_json


def subset_metrics(ev, ref, idx):
    """idx: client -> array of row indices (into that client's grid/outputs)."""
    alphas = ev["alphas"]
    per, ties, cells = {}, [], []
    for c, ix in idx.items():
        r = ev["clients"][c]
        g = np.asarray(r["grid"], dtype=float)[ix]
        per[c] = metrics_for_client(g, alphas, ref, support=tuple(r["support"]))
        outs = [r["outputs"][i] for i in ix]
        ties.append(text_tie_metrics(outs, g, alphas)["near_tie_rate"])
        err = np.abs(np.vectorize(ref.cdf)(g) - np.asarray(alphas)[None, :])     # compare_runs.per_article
        cells.append(err.mean(0))
    S = summarize(per, keys=["pct_calib_err", "pct_err_in_support", "pct_err_out_support", "reach_rate", "spearman"])
    out = {k: f(S, ties, cells) for k, f in KEYS}
    out["per_client"] = {c: {k: per[c].get(k) for k in PER_CLIENT_KEYS} | {"support": per[c].get("support")}
                         for c in per}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=path/to/eval.json")
    ap.add_argument("--ref", required=True, help="arm the per-subset differences are taken against")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--frac", type=float, default=0.75)
    ap.add_argument("--expect_prompts", type=int, default=200,
                    help="maximum test articles per client (a client may have fewer in the data, e.g. "
                         "Newsroom's theguardian.com has 180; every arm must have the same count)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--per_client", action="store_true",
                    help="also print each client's metrics (mean +- std over the subsets) and support")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = load(path) + (path,)
    if args.ref not in arms:
        sys.exit(f"--ref {args.ref} is not one of the arms")

    # ---- checks: same clients, same article ids per client, same alpha reference, full size
    base_name = args.ref
    base_ev, base_ref_json, _ = arms[base_name]
    problems = []
    for name, (ev, ref_json, path) in arms.items():
        if ref_json != base_ref_json:
            problems.append(f"{name}: alpha_reference.json differs from {base_name}'s")
        if sorted(ev["clients"]) != sorted(base_ev["clients"]):
            problems.append(f"{name}: clients {sorted(ev['clients'])} != {sorted(base_ev['clients'])}")
            continue
        for c in base_ev["clients"]:
            ids, bids = ev["clients"][c].get("record_ids"), base_ev["clients"][c].get("record_ids")
            if ids is None:
                problems.append(f"{name}/{c}: no record_ids in the eval file")
            elif ids != bids:
                problems.append(f"{name}/{c}: test article ids differ from {base_name}'s "
                                f"({len(set(ids) & set(bids))} of {len(bids)} shared; same order: {ids == bids})")
            if len(ev["clients"][c]["grid"]) != len(base_ev["clients"][c]["grid"]):
                problems.append(f"{name}/{c}: {len(ev['clients'][c]['grid'])} prompts, {base_name} has "
                                f"{len(base_ev['clients'][c]['grid'])}")
            if len(ev["clients"][c]["grid"]) > args.expect_prompts:
                problems.append(f"{name}/{c}: {len(ev['clients'][c]['grid'])} prompts, more than {args.expect_prompts}")
            if "outputs" not in ev["clients"][c]:
                problems.append(f"{name}/{c}: no saved outputs (near-tie needs them)")
    if problems:
        print("REFUSING: the arms are not on the same test articles:\n  " + "\n  ".join(problems))
        sys.exit(2)
    ref = alpha_reference_from_json(base_ref_json)
    clients = sorted(base_ev["clients"])
    n = {c: len(base_ev["clients"][c]["record_ids"]) for c in clients}

    # ---- subsets: drawn once per seed, per client, by article; applied to every arm
    subsets = draw_subsets(n, seeds, args.frac)
    full = {c: np.arange(n[c]) for c in clients}

    res = {name: {"full": subset_metrics(ev, ref, full),
                  "subsets": [subset_metrics(ev, ref, subsets[s]) for s in seeds]}
           for name, (ev, _, _) in arms.items()}

    label = f"{len(seeds)} subsets x {args.frac:.0%} of test articles"
    print(f"== E1 steering metrics, mean +- std over {label} (full-test value in brackets)")
    print(f"   arms on identical test articles: {', '.join(arms)}; {sum(n.values())} articles, {len(clients)} clients")
    print(f"   articles per client (full / per subset): "
          + ", ".join(f"{c} {n[c]}/{int(round(args.frac * n[c]))}" for c in clients))
    for k, _ in KEYS:
        print(f"\n-- {k}")
        for name in arms:
            v = np.array([r[k] for r in res[name]["subsets"]], dtype=float)
            print(f"   {name:26s} {v.mean():.3f} +- {v.std(ddof=1):.3f}   [{res[name]['full'][k]:.3f}]")
    if args.per_client:
        print(f"\n== per client, mean +- std over {label} (full-test value in brackets); support = the "
              "client's 5th-95th percentile on the global alpha scale")
        for k in PER_CLIENT_KEYS:
            print(f"\n-- {k}")
            print(f"   {'client':16s} {'support':>13s}" + "".join(f"{name:>28s}" for name in arms))
            for c in clients:
                sup = res[args.ref]["full"]["per_client"][c]["support"]
                cells = []
                for name in arms:
                    v = [r["per_client"][c][k] for r in res[name]["subsets"]]
                    f = res[name]["full"]["per_client"][c][k]
                    if any(x is None for x in v) or f is None:
                        cells.append(f"{'n/a':>28s}")
                    else:
                        v = np.array(v, dtype=float)
                        cells.append(f"{v.mean():.3f} +- {v.std(ddof=1):.3f} [{f:.3f}]".rjust(28))
                print(f"   {c:16s} [{sup[0]:.2f},{sup[1]:.2f}]  " + "".join(cells))
    print(f"\n-- per-subset difference in mean pct err vs {args.ref} (mean +- std over {label}; "
          f"negative = arm better). A gap is real only if it clearly exceeds its std.")
    refv = np.array([r["pct_err"] for r in res[args.ref]["subsets"]])
    diffs = {}
    for name in arms:
        if name == args.ref:
            continue
        d = np.array([r["pct_err"] for r in res[name]["subsets"]]) - refv
        diffs[name] = d.tolist()
        print(f"   {name:26s} {d.mean():+.4f} +- {d.std(ddof=1):.4f}   "
              f"(full test {res[name]['full']['pct_err'] - res[args.ref]['full']['pct_err']:+.4f})")
    if args.json:
        json.dump({"label": label, "seeds": seeds, "frac": args.frac, "ref": args.ref,
                   "files": {k: v[2] for k, v in arms.items()}, "results": res, "pct_err_diff_vs_ref": diffs},
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
