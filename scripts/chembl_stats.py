"""Step 3: join activities to SMILES, compute RDKit descriptors, and report
per-client (per-target) attribute statistics.

Reports, for each candidate attribute:
  * per-client median / support on the GLOBAL percentile scale (the plan's
    alpha_mode=global equal-weight mixture CDF), i.e. the Newsroom table;
  * a heterogeneity score (spread of client medians, mean support width);
  * corr(scaffold value, molecule value) -- how much the input determines the
    attribute (the R6 "is alpha free given x" check).

Writes data/chembl/molecules.parquet and data/chembl/client_stats.json.
"""
import argparse, csv, gzip, json, os
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ATTRS = {
    "mw": lambda m: Descriptors.MolWt(m),
    "clogp": lambda m: Crippen.MolLogP(m),
    "tpsa": lambda m: rdMolDescriptors.CalcTPSA(m),
    "qed": lambda m: QED.qed(m),
    "hbd": lambda m: rdMolDescriptors.CalcNumHBD(m),
    "hba": lambda m: rdMolDescriptors.CalcNumHBA(m),
    "rotb": lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
    "arom_rings": lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "fsp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
}


def load_smiles(path, wanted):
    out = {}
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            p = line.split("\t")
            if p[0] in wanted:
                out[p[0]] = p[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activities", default="data/chembl/activities_filtered.parquet")
    ap.add_argument("--meta", default="data/chembl/target_meta.csv")
    ap.add_argument("--chemreps", default="data/chembl/chembl_37_chemreps.txt.gz")
    ap.add_argument("--out_dir", default="data/chembl")
    ap.add_argument("--n_clients", type=int, default=40,
                    help="how many top targets to report on")
    ap.add_argument("--min_molecules", type=int, default=2000)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    with open(args.meta) as f:
        meta = {r["target_chembl_id"]: r for r in csv.DictReader(f)}
    sel = [t for t, r in sorted(meta.items(), key=lambda kv: -int(kv[1]["n_molecules"]))
           if int(r["n_molecules"]) >= args.min_molecules][: args.n_clients]
    sel_set = set(sel)
    print(f"{len(sel)} candidate clients (>= {args.min_molecules} molecules)")

    act = pq.read_table(args.activities).to_pydict()
    pairs = {}
    for tid, mid, val, typ in zip(act["target_chembl_id"], act["molecule_chembl_id"],
                                  act["standard_value"], act["standard_type"]):
        if tid in sel_set:
            pairs.setdefault((tid, mid), []).append((val, typ))
    print(f"{len(pairs):,} distinct (target, molecule) pairs")

    mids = {m for _, m in pairs}
    print(f"{len(mids):,} distinct molecules; reading SMILES ...")
    smiles = load_smiles(args.chemreps, mids)
    print(f"  resolved {len(smiles):,} ({100*len(smiles)/len(mids):.1f}%)")

    # descriptors, computed once per molecule
    desc, scaf_desc, bad = {}, {}, 0
    for i, (mid, smi) in enumerate(smiles.items()):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad += 1
            continue
        try:
            desc[mid] = {k: float(fn(m)) for k, fn in ATTRS.items()}
            s = MurckoScaffold.GetScaffoldForMol(m)
            if s is not None and s.GetNumHeavyAtoms() > 0:
                scaf_desc[mid] = {k: float(fn(s)) for k, fn in ATTRS.items()}
                scaf_desc[mid]["_smiles"] = Chem.MolToSmiles(s)
        except Exception:
            bad += 1
        if (i + 1) % 25000 == 0:
            print(f"  descriptors {i+1:,}/{len(smiles):,}")
    print(f"  unparseable/failed: {bad}")

    # per-client molecule lists
    clients = {}
    for (tid, mid) in pairs:
        if mid in desc:
            clients.setdefault(tid, []).append(mid)

    records = []
    for tid, ms in clients.items():
        for mid in ms:
            r = {"target_chembl_id": tid, "molecule_chembl_id": mid,
                 "smiles": smiles[mid]}
            r.update(desc[mid])
            sc = scaf_desc.get(mid)
            r["scaffold_smiles"] = sc["_smiles"] if sc else ""
            for k in ATTRS:
                r[f"scaf_{k}"] = sc[k] if sc else float("nan")
            records.append(r)
    import pyarrow as pa
    tbl = pa.Table.from_pylist(records)
    os.makedirs(args.out_dir, exist_ok=True)
    pq.write_table(tbl, os.path.join(args.out_dir, "molecules.parquet"))
    print(f"\nwrote molecules.parquet: {tbl.num_rows:,} (target, molecule) rows")

    # ---- heterogeneity analysis, per attribute -------------------------------
    order = sorted(clients, key=lambda t: -len(clients[t]))
    stats = {}
    print("\n" + "=" * 100)
    for attr in ATTRS:
        per_client = {t: np.array([desc[m][attr] for m in clients[t]]) for t in order}
        # equal-weight mixture CDF over clients (the plan's global alpha scale)
        grid = np.unique(np.concatenate([np.percentile(v, np.linspace(0, 100, 201))
                                         for v in per_client.values()]))
        cdfs = np.stack([np.searchsorted(np.sort(v), grid, side="right") / len(v)
                         for v in per_client.values()])
        mix = cdfs.mean(axis=0)

        def F(x):
            return np.interp(x, grid, mix)

        rows = []
        for t in order:
            v = per_client[t]
            lo, med, hi = np.percentile(v, [5, 50, 95])
            rows.append({"target": t, "n": len(v), "raw_median": float(med),
                         "g_median": float(F(med)), "g_lo": float(F(lo)),
                         "g_hi": float(F(hi))})
        meds = np.array([r["g_median"] for r in rows])
        widths = np.array([r["g_hi"] - r["g_lo"] for r in rows])
        # how much does the scaffold (the model input) determine the attribute?
        both = [m for m in desc if m in scaf_desc]
        cc = float(np.corrcoef([desc[m][attr] for m in both],
                               [scaf_desc[m][attr] for m in both])[0, 1]) if both else float("nan")
        stats[attr] = {"client_median_spread": float(meds.max() - meds.min()),
                       "client_median_std": float(meds.std()),
                       "mean_support_width": float(widths.mean()),
                       "scaffold_corr": cc, "rows": rows}
        print(f"{attr:12s} client-median spread {meds.max()-meds.min():.2f} "
              f"(std {meds.std():.3f})  mean support width {widths.mean():.2f}  "
              f"corr(scaffold, molecule) = {cc:+.2f}")

    with open(os.path.join(args.out_dir, "client_stats.json"), "w") as f:
        json.dump({"attrs": stats,
                   "meta": {t: meta[t] for t in order if t in meta}}, f, indent=1)
    print(f"\nwrote client_stats.json")


if __name__ == "__main__":
    main()
