"""Step 4: pick the attribute and the client set.

Answers four questions:
  Q1 (R6) Given the model input, is the attribute still free?  -> within-scaffold
          spread of the attribute on the global percentile scale.
  Q2 (R3) Do clients have different, partly disjoint supports? -> per-client table.
  Q3      What does a constant output score?  -> the trivial-baseline error.
  Q4      Do clients share inputs (scaffolds), so a shared direction is meaningful?
"""
import argparse, json
import numpy as np
import pyarrow.parquet as pq

ATTRS = ["clogp", "qed", "tpsa", "mw", "rotb", "fsp3", "hbd", "heavy_atoms"]


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
    ap.add_argument("--n_clients", type=int, default=12)
    ap.add_argument("--min_molecules", type=int, default=4000)
    args = ap.parse_args()

    t = pq.read_table(args.molecules).to_pydict()
    meta = json.load(open(args.stats))["meta"]
    n = len(t["target_chembl_id"])
    tid = np.array(t["target_chembl_id"]); mid = np.array(t["molecule_chembl_id"])
    scaf = np.array(t["scaffold_smiles"])
    vals = {a: np.array(t[a], dtype=float) for a in ATTRS}

    counts = {u: int((tid == u).sum()) for u in np.unique(tid)}
    pool = [u for u in sorted(counts, key=lambda u: -counts[u])
            if counts[u] >= args.min_molecules]
    print(f"targets with >= {args.min_molecules} molecules: {len(pool)}")

    print("\n" + "=" * 104)
    print("Q1 (R6)  Is the attribute free given the input (the Murcko scaffold)?")
    print("         Spread of the attribute WITHIN a scaffold, on the global percentile scale.")
    print("=" * 104)
    print(f"{'attribute':12s} {'scaffolds>=5':>12s} {'median within-scaffold p5-p95 width':>38s} {'corr(scaf,mol)':>16s}")
    uniq, inv, cnt = np.unique(scaf, return_inverse=True, return_counts=True)
    big = np.where(cnt >= 5)[0]
    allstats = json.load(open(args.stats))["attrs"]
    for a in ATTRS:
        per_client = {u: vals[a][tid == u] for u in pool}
        F = mixture_cdf(per_client)
        g = F(vals[a])
        widths = []
        for s in big:
            gi = g[inv == s]
            widths.append(np.percentile(gi, 95) - np.percentile(gi, 5))
        print(f"{a:12s} {len(big):12,} {np.median(widths):38.2f} "
              f"{allstats[a]['scaffold_corr']:+16.2f}")

    print("\n" + "=" * 104)
    print("Q4  Input sharing across clients")
    print("=" * 104)
    sub = np.isin(tid, pool)
    sc_by_client = {u: set(scaf[tid == u]) for u in pool}
    shared = {}
    for u in pool:
        others = set().union(*[sc_by_client[v] for v in pool if v != u])
        shared[u] = len(sc_by_client[u] & others) / max(1, len(sc_by_client[u]))
    print(f"  distinct scaffolds over the pool: {len(set(scaf[sub])):,}")
    print(f"  share of a client's scaffolds also seen at another client: "
          f"mean {np.mean(list(shared.values())):.2f}, "
          f"min {min(shared.values()):.2f}, max {max(shared.values()):.2f}")
    mol_shared = []
    mol_by_client = {u: set(mid[tid == u]) for u in pool}
    for u in pool:
        others = set().union(*[mol_by_client[v] for v in pool if v != u])
        mol_shared.append(len(mol_by_client[u] & others) / len(mol_by_client[u]))
    print(f"  share of a client's MOLECULES also measured at another client: "
          f"mean {np.mean(mol_shared):.2f}, max {max(mol_shared):.2f}")

    for attr in ("clogp", "qed", "tpsa"):
        per_client = {u: vals[attr][tid == u] for u in pool}
        F = mixture_cdf(per_client)
        rows = []
        for u in pool:
            v = per_client[u]
            lo, q1, med, q3, hi = np.percentile(v, [5, 25, 50, 75, 95])
            rows.append((u, len(v), med, F(lo), F(med), F(hi)))
        rows.sort(key=lambda r: r[4])
        print("\n" + "=" * 104)
        print(f"Q2 (R3)  Per-client supports on the global alpha scale -- attribute = {attr}")
        print("=" * 104)
        print(f"{'target':14s} {'name':34s} {'family':16s} {'n':>6s} {'raw med':>8s} "
              f"{'a(med)':>7s} {'support [a5, a95]':>19s}")
        for u, k, med, a_lo, a_med, a_hi in rows:
            m = meta.get(u, {})
            fam = " ".join([m.get("class_l1", ""), m.get("class_l2", "")]).strip()[:16]
            print(f"{u:14s} {m.get('pref_name','?')[:34]:34s} {fam:16s} {k:6,} "
                  f"{med:8.2f} {a_med:7.2f}   [{a_lo:.2f}, {a_hi:.2f}]")
        # Q3: trivial baseline -- always emit the client's own median
        errs = []
        for u, k, med, *_ in rows:
            a_const = F(med)
            grid = np.linspace(0, 1, 5)
            errs.append(np.abs(grid - a_const).mean())
        print(f"  constant-output percentile error (always the client median), "
              f"5-point grid: mean {np.mean(errs):.3f}, best client {min(errs):.3f}")


if __name__ == "__main__":
    main()
