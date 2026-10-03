"""Molecule-specific quality metrics for an eval file (the ChEMBL task's
counterpart of scripts/score_quality.py).

    python scripts/score_molecules.py --run runs/X --eval runs/X/evals/eval_*.json

Reports, per client and per alpha:
  validity          share of generations RDKit can parse            <- gate G0
  retention         share that still contain the requested core     <- gate G0
  uniqueness        distinct canonical SMILES / valid generations
  novelty           valid generations not in the client's train set
  qed, mw, tpsa     off-target descriptors (specificity)
  ref_*             the same quantities on the real molecules of that
                    scaffold at a comparable alpha (data/chembl_fed/refs.json)

Writes <eval dir>/mols_<eval name> next to the eval file.
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fedsteer.data import ClientQuantiles, alpha_reference_from_json, client_support, read_jsonl
from rdkit import Chem

from fedsteer.molecules import (assembled_descriptors, canonical, descriptors,
                                keeps_scaffold, rejoin_decorations)


def load_supports(run: str) -> dict:
    """Per-client (lo, hi) of the alpha axis its own training data covers, so validity
    and retention can be split the way the percentile error already is. Without this
    split an alpha outside the client's support is scored as if it were a normal
    request, and a model that degrades only where it is extrapolating looks uniformly
    broken."""
    qp = os.path.join(run, "client_quantiles.json")
    rp = os.path.join(run, "alpha_reference.json")
    if not (os.path.exists(qp) and os.path.exists(rp)):
        return {}
    local = {c: ClientQuantiles(v) for c, v in json.load(open(qp)).items()}
    gref = alpha_reference_from_json(json.load(open(rp)))
    if gref is None:
        return {}
    return {c: client_support(local[c], gref) for c in local}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="eval json file(s) or globs")
    ap.add_argument("--data", default="data/chembl_fed/data.jsonl")
    args = ap.parse_args()

    recs = read_jsonl(args.data)
    by_id = {r["url"]: r for r in recs}
    train_canon = {}
    for r in recs:
        if r.get("split") != "train":
            continue
        if "core_attached" in r:
            m = rejoin_decorations(r["core_attached"], r["target"])
            c = Chem.MolToSmiles(m) if m is not None else None
        else:
            c = canonical(r["target"])
        train_canon.setdefault(r["client"], set()).add(c)

    supports = load_supports(args.run)
    files = [f for pat in args.eval for f in sorted(glob.glob(pat))]
    if not files:
        raise SystemExit("no eval files matched")

    for path in files:
        d = json.load(open(path))
        alphas = d["alphas"]
        out = {}
        for c, res in d["clients"].items():
            texts, ids = res["outputs"], res["record_ids"]
            per_alpha = []
            for j, a in enumerate(alphas):
                gens = [texts[i][j] for i in range(len(texts))]
                valid, keep, canons, desc = [], [], [], []
                for g, rid in zip(gens, ids):
                    rec = by_id.get(rid, {})
                    deco = "core_attached" in rec
                    # decoration format: score the molecule the generation assembles to
                    dd = assembled_descriptors(g, rec) if deco else descriptors(g)
                    valid.append(dd["valid"])
                    if dd["valid"]:
                        desc.append(dd)
                        canons.append(dd["smiles"] if deco else canonical(g))
                        if deco:
                            keep.append(dd["keeps_core"])
                        else:
                            k = keeps_scaffold(g, rec.get("scaffold", ""))
                            if k is not None:
                                keep.append(float(k))
                row = {"alpha": a, "n": len(gens), "validity": float(np.mean(valid))}
                if desc:
                    row["retention"] = float(np.mean(keep)) if keep else None
                    row["uniqueness"] = len(set(canons)) / len(canons)
                    seen = train_canon.get(c, set())
                    row["novelty"] = float(np.mean([s not in seen for s in canons]))
                    for k in ("qed", "mw", "tpsa", "clogp", "heavy_atoms"):
                        row[k] = float(np.mean([x[k] for x in desc]))
                if c in supports:
                    lo, hi = supports[c]
                    row["in_support"] = bool(lo <= a <= hi)
                per_alpha.append(row)
            out[c] = per_alpha

        agg = {}
        for k in ("validity", "retention", "uniqueness", "novelty", "qed", "mw", "tpsa"):
            vals = [r[k] for rows in out.values() for r in rows if r.get(k) is not None]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "worst": float(min(vals))}
            # the same quantity restricted to alphas the client's own data covers, and to
            # those it does not: a model may decorate correctly in-support and only break
            # down where the knob is pushed outside the training distribution
            for tag, want in (("in_support", True), ("out_support", False)):
                v = [r[k] for rows in out.values() for r in rows
                     if r.get(k) is not None and r.get("in_support") is want]
                if v:
                    agg.setdefault(k, {})[tag] = float(np.mean(v))
        dst = os.path.join(os.path.dirname(path), "mols_" + os.path.basename(path))
        json.dump({"eval": os.path.basename(path), "per_client": out, "summary": agg},
                  open(dst, "w"), indent=1)

        print(f"\n=== {os.path.basename(path)}")
        print(f"{'client':14s} " + " ".join(f"a={a:<4g} valid retain uniq" for a in alphas))
        for c, rows in out.items():
            cells = " ".join(
                f"{'':7s}{r['validity']:5.2f} {(r.get('retention') or 0):5.2f} "
                f"{(r.get('uniqueness') or 0):5.2f}" for r in rows)
            print(f"{c:14s} {cells}")
        print("summary:", json.dumps({k: round(v["mean"], 3) for k, v in agg.items()}))
        for k in ("validity", "retention"):
            a = agg.get(k, {})
            if "in_support" in a:
                print(f"  {k:10s} mean {a['mean']:.3f} | in-support {a['in_support']:.3f} | "
                      f"out-of-support {a['out_support']:.3f} | worst alpha {a['worst']:.3f}")
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
