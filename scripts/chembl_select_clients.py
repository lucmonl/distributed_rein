"""Step 5: choose the client set.

Attribute: cLogP residual = cLogP(molecule) - cLogP(Murcko scaffold), i.e. how
lipophilic the decorations are. Chosen because it has both the most freedom
given the input (within-scaffold spread 0.31 of the global scale) and the
largest spread of client medians (0.81) -- see scripts/chembl_report.py.

Picks N targets that are spread over the global alpha scale, come from distinct
protein families, and share few molecules with each other. Writes
data/chembl/clients.json.
"""
import argparse, collections, json
import numpy as np
import pyarrow.parquet as pq


def mixture_cdf(per_client):
    grid = np.unique(np.concatenate(
        [np.percentile(v, np.linspace(0, 100, 201)) for v in per_client.values()]))
    cdfs = np.stack([np.searchsorted(np.sort(v), grid, side="right") / len(v)
                     for v in per_client.values()])
    mix = cdfs.mean(axis=0)
    return lambda x: np.interp(x, grid, mix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default="data/chembl/molecules.parquet")
    ap.add_argument("--stats", default="data/chembl/client_stats.json")
    ap.add_argument("--out", default="data/chembl/clients.json")
    ap.add_argument("--min_molecules", type=int, default=4000)
    ap.add_argument("--n_total", type=int, default=12)
    ap.add_argument("--max_overlap", type=float, default=0.15)
    ap.add_argument("--mode", choices=["even", "skewed"], default="even",
                    help="even: spread client medians over alpha, one per family. "
                         "skewed: prefer clients with NARROW supports, to maximise "
                         "the coverage gaps claim C1 is about (<=2 per family)")
    ap.add_argument("--max_per_family", type=int, default=2)
    args = ap.parse_args()

    rows = [r for r in pq.read_table(args.molecules).to_pylist() if r["scaffold_smiles"]]
    meta = json.load(open(args.stats))["meta"]
    by_t = collections.defaultdict(list)
    for r in rows:
        by_t[r["target_chembl_id"]].append(r)
    pool = [t for t, rs in by_t.items() if len(rs) >= args.min_molecules]

    resid = {t: np.array([r["clogp"] - r["scaf_clogp"] for r in by_t[t]]) for t in pool}
    F = mixture_cdf(resid)
    mols = {t: {r["molecule_chembl_id"] for r in by_t[t]} for t in pool}

    def overlap(a, b):
        return len(mols[a] & mols[b]) / min(len(mols[a]), len(mols[b]))

    med = {t: float(F(np.median(resid[t]))) for t in pool}
    width = {t: float(F(np.percentile(resid[t], 95)) - F(np.percentile(resid[t], 5)))
             for t in pool}

    def fam_of(t):
        m = meta.get(t, {})
        return m.get("class_l2") or m.get("class_l1") or t

    chosen, fam_n = [], collections.Counter()

    def ok(t):
        if t in chosen:
            return False
        cap = 1 if args.mode == "even" else args.max_per_family
        if fam_n[fam_of(t)] >= cap:
            return False
        return all(overlap(t, c) <= args.max_overlap for c in chosen)

    if args.mode == "even":
        order = [(w, sorted(pool, key=lambda t: abs(med[t] - w)))
                 for w in np.linspace(0.1, 0.9, args.n_total)]
        for _, cands in order:
            for t in cands:
                if ok(t):
                    chosen.append(t); fam_n[fam_of(t)] += 1
                    break
    else:
        # narrowest supports first, alternating between the low and high half of
        # the scale so the selected clients' gaps are complementary
        low = sorted([t for t in pool if med[t] < 0.5], key=lambda t: width[t])
        high = sorted([t for t in pool if med[t] >= 0.5], key=lambda t: width[t])
        while len(chosen) < args.n_total and (low or high):
            src = low if (len(chosen) % 2 == 0 and low) or not high else high
            for t in list(src):
                src.remove(t)
                if ok(t):
                    chosen.append(t); fam_n[fam_of(t)] += 1
                    break
    for t in sorted(pool, key=lambda t: width[t]):   # backfill
        if len(chosen) >= args.n_total:
            break
        if ok(t):
            chosen.append(t); fam_n[fam_of(t)] += 1

    chosen.sort(key=lambda t: med[t])
    sub = {t: resid[t] for t in chosen}
    F2 = mixture_cdf(sub)   # reference built from the chosen clients only

    print(f"selected {len(chosen)} clients (attribute = cLogP residual)\n")
    print(f"{'target':14s} {'name':30s} {'family':20s} {'n':>6s} {'raw med':>8s} "
          f"{'a(med)':>7s} {'support':>18s} {'role':>12s}")
    # stratified rotation 0: every 3rd client held out, like the Newsroom rotations
    held = set(chosen[1::3][:4])
    out = {}
    for t in chosen:
        v = sub[t]
        lo, m, hi = np.percentile(v, [5, 50, 95])
        role = "held-out" if t in held else "participant"
        mt = meta.get(t, {})
        fam = " ".join([mt.get("class_l1", ""), mt.get("class_l2", "")]).strip()
        print(f"{t:14s} {mt.get('pref_name','?')[:30]:30s} {fam[:20]:20s} {len(v):6,} "
              f"{m:8.2f} {F2(m):7.2f}   [{F2(lo):.2f}, {F2(hi):.2f}] {role:>12s}")
        out[t] = {"pref_name": mt.get("pref_name", ""), "family": fam,
                  "n_molecules": len(v), "role": role,
                  "raw_median": float(m),
                  "alpha_median": float(F2(m)),
                  "support": [float(F2(lo)), float(F2(hi))]}

    ov = [overlap(a, b) for i, a in enumerate(chosen) for b in chosen[i + 1:]]
    print(f"\npairwise molecule overlap: mean {np.mean(ov):.3f}, max {max(ov):.3f}")
    gaps = [1 - (o["support"][1] - o["support"][0]) for o in out.values()]
    print(f"mean uncovered share of the alpha scale per client: {np.mean(gaps):.2f}")
    cov = np.mean([[o["support"][0] <= a <= o["support"][1] for a in np.linspace(0, 1, 11)]
                   for o in out.values()], axis=0)
    print("share of clients covering each alpha:",
          " ".join(f"{a:.1f}:{c:.2f}" for a, c in zip(np.linspace(0, 1, 11), cov)))

    json.dump({"attribute": "clogp_residual", "clients": out}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
