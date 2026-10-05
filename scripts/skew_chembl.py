"""Natural coverage skew for ChEMBL clients, by chemical-programme selection.

    python scripts/skew_chembl.py --out_dir data/chembl_deco_skew

Reproduces the *kind* of skew Newsroom publications have naturally, instead of
censoring molecules at an alpha threshold. Three properties of real skew that a
hard cut gets wrong, plus one regime Newsroom itself lacks:

1. **The mechanism is programme-level.** A library is skewed by which chemical
   programmes a company ran; a programme (a congeneric series sharing a scaffold)
   occupies a narrow property band, so whole series are kept or dropped. A weaker
   ramp then thins each surviving series' disfavoured tail, because SAR
   exploration concentrates where the team was optimising. Series selection alone
   cannot narrow a support below ~0.72: one ChEMBL series spans ~0.31 of the scale.
2. **Edges are soft.** nypost.com's support starts at 0.51 because it has *few*
   low-extractiveness summaries, not none. Keep-probabilities never reach 0, so
   every client retains a real tail.
3. **Strength is heterogeneous.** In Newsroom only 2 of 8 clients are strongly
   skewed (widths 0.44, 0.50); the rest are broad (0.67-0.96). Uniform windows
   would be *more* artificial than the real data.
4. **Middle-only clients.** Newsroom has none -- its clients are all one-sided --
   but a lead-optimisation shop that stays in the drug-like sweet spot is real,
   and a harder case: it needs *both* tails from other clients, so the direction
   must extrapolate in both directions for the same client.

Each client gets a role: a propensity shape over alpha, a centre and a strength.
    one-sided: p = (1 - beta) + beta * sigmoid(s * (alpha - centre) / tau)
    middle:    p = (1 - beta) + beta * exp(-0.5 * ((alpha - 0.5) / width)^2)

The ramp centre is a parameter rather than fixed at 0.5 because the natural alpha
medians only span 0.27-0.64 (the scale is built from these same clients), so a
ramp centred at 0.5 cannot push a high-side client's 5th percentile up to ~0.5.

Roles are assigned by natural alpha median: lowest -> low specialist, highest ->
high specialist, most central -> middle. Specialists on both sides are required,
or no client would hold what the others lack and transfer would be untestable.
"""
import argparse, collections, json, os, random

import numpy as np

# (name, shape, centre, beta, tau_or_width)
ROLES = [
    ("low_specialist",  "low",    0.46, 0.97, 0.07),
    ("high_specialist", "high",   0.54, 0.97, 0.07),
    ("middle_strong",   "middle", 0.50, 0.95, 0.15),
    ("middle_moderate", "middle", 0.50, 0.55, 0.24),
    ("low_moderate",    "low",    0.55, 0.55, 0.16),
    ("high_moderate",   "high",   0.45, 0.55, 0.16),
    ("broad",           "low",    0.50, 0.10, 0.20),
    ("untouched",       "low",    0.50, 0.00, 0.20),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/chembl_deco/data.jsonl")
    ap.add_argument("--clients_file", default="data/chembl_deco/clients.json")
    ap.add_argument("--out_dir", default="data/chembl_deco_skew")
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--mol_frac", type=float, default=0.8,
                    help="within-series thinning strength as a fraction of beta")
    ap.add_argument("--tau_mol_scale", type=float, default=2.0,
                    help="within-series ramp is this many times softer than the series ramp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    recs = [json.loads(l) for l in open(args.data)]
    spec = json.load(open(args.clients_file))
    parts = spec["rotations"][args.rotation]["participants"]

    train = collections.defaultdict(list)
    for r in recs:
        if r.get("split") == "train" and r["client"] in parts:
            train[r["client"]].append(r)
    sorted_scores = {c: np.sort([r["score"] for r in v]) for c, v in train.items()}

    def alpha(x):
        return float(np.mean([np.searchsorted(s, x, side="right") / len(s)
                             for s in sorted_scores.values()]))

    med = {c: alpha(float(np.median(sorted_scores[c]))) for c in parts}
    by_med = sorted(parts, key=lambda c: med[c])
    assign = {by_med[0]: ROLES[0], by_med[-1]: ROLES[1]}
    for c, role in zip(sorted(by_med[1:-1], key=lambda c: abs(med[c] - 0.5)), ROLES[2:]):
        assign[c] = role

    def keep_prob(a, shape, centre, beta, scale):
        if shape == "middle":
            r = float(np.exp(-0.5 * ((a - 0.5) / scale) ** 2))
        else:
            s = -1.0 if shape == "low" else 1.0
            r = float(1.0 / (1.0 + np.exp(-s * (a - centre) / scale)))
        return (1 - beta) + beta * r

    keep_rows, summary = [], {}
    for c in parts:
        name, shape, centre, beta, scale = assign[c]
        series = collections.defaultdict(list)
        for r in train[c]:
            series[r["scaffold"]].append(r)
        kept = []
        for rows in series.values():
            a_s = float(np.median([alpha(r["score"]) for r in rows]))
            if rng.random() >= keep_prob(a_s, shape, centre, beta, scale):
                continue
            for r in rows:
                if rng.random() < keep_prob(alpha(r["score"]), shape, centre,
                                            args.mol_frac * beta,
                                            scale * args.tau_mol_scale):
                    kept.append(r)
        keep_rows.extend(kept)
        a = np.array([alpha(r["score"]) for r in kept])
        lo, hi = (np.percentile(a, [5, 95]) if len(a) else (np.nan, np.nan))
        summary[c] = {"role": name, "shape": shape, "beta": beta,
                      "alpha_median_full": med[c], "n_train_full": len(train[c]),
                      "n_train_kept": len(kept),
                      "support": [round(float(lo), 3), round(float(hi), 3)],
                      "mass_below_0.25": float((a < 0.25).mean()) if len(a) else None,
                      "mass_above_0.75": float((a > 0.75).mean()) if len(a) else None}

    kept_ids = {id(r) for r in keep_rows}
    out = [r for r in recs
           if not (r.get("split") == "train" and r["client"] in parts) or id(r) in kept_ids]

    print(f"{'client':12s} {'role':17s} {'kept':>6s} {'%':>5s} {'support (5-95%)':>16s} "
          f"{'width':>6s} {'<0.25':>7s} {'>0.75':>7s}")
    for c in sorted(parts, key=lambda c: summary[c]["support"][0]):
        s = summary[c]; lo, hi = s["support"]
        print(f"  {c:10s} {s['role']:17s} {s['n_train_kept']:6,} "
              f"{100*s['n_train_kept']/s['n_train_full']:4.0f}%     [{lo:.2f}, {hi:.2f}] "
              f"{hi-lo:6.2f} {s['mass_below_0.25']:7.3f} {s['mass_above_0.75']:7.3f}")
    w = [summary[c]["support"][1] - summary[c]["support"][0] for c in parts]
    print(f"\nsupport widths: min {min(w):.2f} median {np.median(w):.2f} max {max(w):.2f}"
          f"   | Newsroom: 0.44 / 0.81 / 0.96")
    grid = np.linspace(0, 1, 11)
    cov = [np.mean([summary[c]["support"][0] <= g <= summary[c]["support"][1] for c in parts])
           for g in grid]
    print("clients covering each alpha: " + " ".join(f"{g:.1f}:{v:.2f}" for g, v in zip(grid, cov)))
    print(f"participant train rows: {len(keep_rows):,} (was {sum(len(v) for v in train.values()):,})")

    with open(os.path.join(args.out_dir, "data.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    json.dump(dict(spec, skew={"seed": args.seed, "rotation": args.rotation,
                               "mol_frac": args.mol_frac, "per_client": summary}),
              open(os.path.join(args.out_dir, "clients.json"), "w"), indent=1)
    print(f"wrote {args.out_dir}/data.jsonl")


if __name__ == "__main__":
    main()
