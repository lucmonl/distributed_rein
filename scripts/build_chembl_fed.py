"""Build the federated ChEMBL splits (scaffold-disjoint).

    python scripts/build_chembl_fed.py --out_dir data/chembl_fed

Writes, in the same layout as data/newsroom_fed/:
  data.jsonl    {client, split, prompt, target, score, scaffold, ...}
  clients.json  {clients, median_residual, rotations}
  refs.json     per (client, split, scaffold): the other real molecules of that
                scaffold, for same-alpha quality references

Splits are disjoint BY SCAFFOLD: a scaffold seen in train never appears in dev or
test, or the answer leaks. Dev/test scaffolds are drawn from those with >= 5
molecules in the client, so each held-out prompt has real molecules at several
alphas to compare generations against.
"""
import argparse, collections, json, os, random, sys

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.molecules import rejoin_decorations, split_core_decorations  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROMPT = ("Design a ligand for {name}.\n"
          "Decorate this core scaffold: {scaffold}\n"
          "Answer with one SMILES string.")

# Decoration format (entry 35): the core carries numbered attachment points and the
# target is only the decorations, so the model cannot rewrite or renumber the core.
PROMPT_DECO = ("Design a ligand for {name}.\n"
               "Decorate this core, keeping every attachment point: {core}\n"
               "Answer with the decorations only, as [n*]-labelled fragments "
               "joined by '.', one per attachment point.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default="data/chembl/molecules.parquet")
    ap.add_argument("--clients", default="data/chembl/clients.json")
    ap.add_argument("--out_dir", default="data/chembl_fed")
    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--n_dev", type=int, default=50)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--min_scaffold_mols", type=int, default=3,
                    help="dev/test scaffolds must have at least this many real molecules")
    ap.add_argument("--max_scaffold_mols", type=int, default=30,
                    help="and at most this many, so held-out scaffolds do not eat the "
                         "largest congeneric series out of the training pool")
    ap.add_argument("--format", choices=["smiles", "deco"], default="smiles",
                    help="smiles: target is the whole molecule (entries 31-34). "
                         "deco: target is only the decorations, zipped onto the core "
                         "from the prompt -- core preservation becomes structural.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    spec = json.load(open(args.clients))
    clients = spec["clients"]
    rows = pq.read_table(args.molecules).to_pylist()

    by_client = collections.defaultdict(list)
    dropped_zero = 0
    for r in rows:
        c = r["target_chembl_id"]
        if c not in clients or not r["scaffold_smiles"]:
            continue
        resid = r["clogp"] - r["scaf_clogp"]
        if resid == 0.0:          # molecule == its own scaffold: nothing decorated
            dropped_zero += 1
            continue
        by_client[c].append(dict(r, score=float(resid)))
    print(f"dropped {dropped_zero:,} rows with residual exactly 0")

    out, refs, summary = [], {}, {}
    for c in clients:
        rs = by_client[c]
        name = clients[c]["pref_name"]
        by_scaf = collections.defaultdict(list)
        for r in rs:
            by_scaf[r["scaffold_smiles"]].append(r)
        big = sorted([s for s, v in by_scaf.items()
                      if args.min_scaffold_mols <= len(v) <= args.max_scaffold_mols])
        rng.shuffle(big)
        need = args.n_dev + args.n_test
        if len(big) < need:
            raise SystemExit(f"{c}: only {len(big)} scaffolds with "
                             f"{args.min_scaffold_mols}-{args.max_scaffold_mols} "
                             f"molecules, need {need}")
        held = big[:need]
        dev_scaf, test_scaf = held[: args.n_dev], held[args.n_dev :]
        held_set = set(held)

        def make(r, split):
            """One record, or None if the decoration split fails (deco format only)."""
            rec = {"client": c, "split": split, "score": r["score"],
                   "scaffold": r["scaffold_smiles"],
                   "target_name": name, "molecule_chembl_id": r["molecule_chembl_id"],
                   # generic record id slot used by evaluate.record_ids and
                   # baselines.pick_shots (named `url` for the Newsroom task)
                   "url": f"{c}/{r['molecule_chembl_id']}"}
            if args.format == "smiles":
                rec["prompt"] = PROMPT.format(name=name, scaffold=rec["scaffold"])
                rec["target"] = r["smiles"]
            else:
                sp = split_core_decorations(r["smiles"], rec["scaffold"])
                if sp is None:
                    return None
                core_attached, deco = sp
                # keep only pairs that zip back to the original molecule, so the
                # training label `score` stays exactly the molecule's own attribute
                back = rejoin_decorations(core_attached, deco)
                if back is None or Chem.MolToSmiles(back) != Chem.MolToSmiles(
                        Chem.MolFromSmiles(r["smiles"])):
                    return None
                rec["core_attached"] = core_attached
                rec["prompt"] = PROMPT_DECO.format(name=name, core=core_attached)
                rec["target"] = deco
            return rec

        def emit(scaffolds, split):
            kept = 0
            for s in scaffolds:
                mols = sorted(by_scaf[s], key=lambda r: r["molecule_chembl_id"])
                pick = mols[len(mols) // 2]          # median-id molecule as the reference target
                rec = make(pick, split)
                if rec is None:
                    continue
                out.append(rec)
                kept += 1
                refs.setdefault(c, {}).setdefault(split, {})[s] = [
                    {"smiles": m["smiles"], "score": m["score"]}
                    for m in mols if m["molecule_chembl_id"] != pick["molecule_chembl_id"]]
            return kept

        n_dev_kept = emit(dev_scaf, "dev")
        n_test_kept = emit(test_scaf, "test")

        train_pool = [r for r in rs if r["scaffold_smiles"] not in held_set]
        rng.shuffle(train_pool)
        train, attempted = [], 0
        for r in train_pool:
            if len(train) >= args.max_train:
                break
            attempted += 1
            rec = make(r, "train")
            if rec is not None:
                out.append(rec)
                train.append(r)
        sc = np.array([r["score"] for r in train])
        summary[c] = {"pref_name": name, "n_train": len(train), "n_dev": n_dev_kept,
                      "n_test": n_test_kept, "median_residual": float(np.median(sc)),
                      "train_scaffolds": len({r["scaffold_smiles"] for r in train}),
                      "split_yield": round(len(train) / max(attempted, 1), 3)}
        print(f"  {c:14s} {name[:30]:30s} train {len(train):5,} "
              f"({summary[c]['train_scaffolds']:5,} scaffolds)  dev {n_dev_kept}  "
              f"test {n_test_kept}  median residual {summary[c]['median_residual']:+.2f}"
              f"  yield {summary[c]['split_yield']:.2f}")

    order = sorted(clients, key=lambda c: summary[c]["median_residual"])
    rotations = [{"held_out": order[i::3][:4],
                  "participants": [c for c in order if c not in set(order[i::3][:4])]}
                 for i in range(3)]

    with open(os.path.join(args.out_dir, "data.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    json.dump({"clients": order,
               "median_residual": {c: summary[c]["median_residual"] for c in order},
               "pref_name": {c: summary[c]["pref_name"] for c in order},
               "rotations": rotations,
               "attribute": "clogp_residual_deco" if args.format == "deco" else "clogp_residual",
               "format": args.format},
              open(os.path.join(args.out_dir, "clients.json"), "w"), indent=1)
    json.dump(refs, open(os.path.join(args.out_dir, "refs.json"), "w"))
    print(f"\nwrote {len(out):,} records to {args.out_dir}/data.jsonl")
    print(f"rotation 0 held out: {rotations[0]['held_out']}")


if __name__ == "__main__":
    main()
