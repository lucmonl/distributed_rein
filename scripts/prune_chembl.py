"""Create a separate decoration dataset excluding parent-ligand shortcut records.

    python scripts/prune_chembl.py --source_dir data/chembl_deco_skew \
        --out_dir data/chembl_deco_skew_pruned

Apply the same structural rule to every split and client; never inspect model
outputs or alpha values to select records. Drop records, rather than editing salt
targets into new answers. Recompute retained scores and train-only references.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen

from fedsteer.data import ClientQuantiles, GlobalQuantiles, client_support, read_jsonl
from fedsteer.molecules import assemble_ligand, core_clogp


def prune_record(rec):
    """Return (retained record with recomputed score, None) or (None, reason)."""
    # Dataset targets must be exactly the decoration string, not a prose/fence
    # extraction that silently accepts an incomplete training answer.
    target = rec["target"].strip()
    if not target or any(c.isspace() for c in target):
        return None, "target_format"
    mol, reason = assemble_ligand(rec["core_attached"], target)
    if reason:
        return None, reason
    core = Chem.MolFromSmiles(rec["scaffold"])
    if core is None or len(Chem.GetMolFrags(core)) != 1 or not mol.HasSubstructMatch(core):
        return None, "scaffold_mismatch"
    score = float(Crippen.MolLogP(mol) - core_clogp(rec["scaffold"]))
    return dict(rec, score=score), None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_dataset(source_dir, out_dir, rotation=0):
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}; choose a new dataset directory")
    spec = json.loads((source_dir / "clients.json").read_text())
    records = read_jsonl(str(source_dir / "data.jsonl"))
    participants = spec["rotations"][rotation]["participants"]
    kept, rejected = [], []
    counts = defaultdict(lambda: defaultdict(lambda: {"before": 0, "kept": 0, "rejected": Counter()}))
    train_scores = defaultdict(list)
    max_score_change = 0.0
    for rec in records:
        row = counts[rec["client"]][rec["split"]]
        row["before"] += 1
        clean, reason = prune_record(rec)
        if reason:
            row["rejected"][reason] += 1
            rejected.append({k: rec[k] for k in ("url", "client", "split")} | {"reason": reason})
            continue
        row["kept"] += 1
        kept.append(clean)
        max_score_change = max(max_score_change, abs(clean["score"] - rec["score"]))
        if rec["split"] == "train":
            train_scores[rec["client"]].append(clean["score"])

    missing = set(spec["clients"]) - set(train_scores)
    if missing:
        raise ValueError(f"clients have no retained training examples: {sorted(missing)}")
    # Removing rows must preserve the original per-client scaffold partition.
    scaffold_splits = defaultdict(set)
    for rec in kept:
        scaffold_splits[(rec["client"], rec["scaffold"])].add(rec["split"])
    if any(len(splits) > 1 for splits in scaffold_splits.values()):
        raise ValueError("retained data contains a scaffold in multiple splits of a client")

    local = {c: ClientQuantiles.fit(scores) for c, scores in train_scores.items()}
    reference = GlobalQuantiles({c: local[c] for c in participants})
    support = {c: client_support(local[c], reference) for c in participants}
    # Original skew/median statistics describe the source, not the pruned data.
    # Archive the complete source metadata separately, retain only assignment/name
    # metadata here, and recompute statistics from this dataset's training records.
    new_spec = {k: spec[k] for k in ("clients", "pref_name", "rotations", "format") if k in spec}
    new_spec.update(attribute="clogp_residual_deco_strict",
                    median_residual={c: statistics.median(s) for c, s in train_scores.items()},
                    pruning={"version": 1, "source_dir": str(source_dir), "manifest": "pruning_manifest.json"})
    manifest = {
        "version": 1, "source_dir": str(source_dir), "rdkit_version": rdBase.rdkitVersion,
        "source_sha256": {f: sha256(source_dir / f) for f in ("data.jsonl", "clients.json")},
        "policy": "drop records with unlabelled components, invalid/mismatched attachments, or non-single-ligand assembly",
        "splits": "all splits filtered identically; retained records keep their original split and prompt",
        "reference_policy": "recomputed from retained training scores only; equal-weight participant CDF mixture",
        "rotation": rotation, "participants": participants,
        "before": len(records), "kept": len(kept), "dropped": len(rejected),
        "reasons": dict(Counter(r["reason"] for r in rejected)),
        "per_client_split": counts, "support_05_95": support,
        "max_retained_score_change": max_score_change,
    }
    out_dir.mkdir(parents=True)
    for name, rows in (("data.jsonl", kept), ("rejected.jsonl", rejected)):
        with (out_dir / name).open("w") as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False) + "\n")
    artifacts = {"clients.json": new_spec, "source_clients.json": spec,
                 "client_quantiles.json": {c: q.sorted_scores for c, q in local.items()},
                 "alpha_reference.json": reference.to_json(), "pruning_manifest.json": manifest}
    for name, obj in artifacts.items():
        (out_dir / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    (out_dir / "README.md").write_text(
        "# Pruned ChEMBL decoration dataset\n\n"
        f"Source: `{source_dir}`. Kept {len(kept):,}/{len(records):,} records.\n\n"
        "Records with disconnected extras or invalid attachments are dropped from every split. "
        "Retained targets, prompts and split assignments are preserved; scores are recomputed. "
        "This is filtering, not chemical salt neutralization or rewriting targets.\n\n"
        "Use `clogp_residual_deco_strict` for the single-ligand contract; "
        "`clogp_residual_deco_pruned` is a supplemental output-repair sensitivity check. "
        "Train-only CDFs are rebuilt, so these alpha values are not the old run's scale. "
        f"The exported global reference uses rotation {rotation}; training rebuilds the reference "
        "for its configured participants.\n\n"
        "See `pruning_manifest.json` for counts, rules and source hashes, "
        "`rejected.jsonl` for rejected IDs and reasons, and `source_clients.json` for original metadata.\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_dir", default="data/chembl_deco_skew")
    ap.add_argument("--out_dir", default="data/chembl_deco_skew_pruned")
    ap.add_argument("--rotation", type=int, default=0)
    args = ap.parse_args()
    report = build_dataset(args.source_dir, args.out_dir, args.rotation)
    print(f"kept {report['kept']}/{report['before']}; rejected {report['dropped']}: {report['reasons']}")
    for c in report["participants"]:
        s = report["per_client_split"][c]
        print(c, " ".join(f"{split} {v['kept']}/{v['before']}" for split, v in s.items()),
              "support", [round(x, 4) for x in report["support_05_95"][c]])
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
