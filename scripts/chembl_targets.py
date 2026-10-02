"""Step 1: filter ChEMBL activities to high-confidence binding data and count
distinct molecules per target.

Writes data/chembl/target_counts.csv and data/chembl/activities_filtered.parquet.
"""
import argparse, os
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.parquet as pq

ACT_TYPES = {"Ki", "Kd", "IC50", "EC50"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activities", required=True, help="ChEMBL_activities.csv")
    ap.add_argument("--out_dir", default="data/chembl")
    ap.add_argument("--min_confidence", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    keep_cols = [
        "molecule_chembl_id", "target_chembl_id", "standard_relation",
        "standard_value", "standard_units", "standard_type", "assay_type",
        "confidence_score",
    ]
    reader = pacsv.open_csv(
        args.activities,
        convert_options=pacsv.ConvertOptions(include_columns=keep_cols),
        read_options=pacsv.ReadOptions(block_size=1 << 26),
    )

    batches, n_in = [], 0
    for batch in reader:
        t = pa.Table.from_batches([batch])
        n_in += t.num_rows
        mask = pc.and_(
            pc.equal(t["assay_type"], "B"),
            pc.greater_equal(t["confidence_score"], args.min_confidence),
        )
        mask = pc.and_(mask, pc.equal(t["standard_relation"], "="))
        mask = pc.and_(mask, pc.equal(t["standard_units"], "nM"))
        mask = pc.and_(mask, pc.is_in(t["standard_type"], value_set=pa.array(sorted(ACT_TYPES))))
        mask = pc.and_(mask, pc.is_valid(t["standard_value"]))
        mask = pc.and_(mask, pc.greater(t["standard_value"], 0))
        t = t.filter(pc.fill_null(mask, False))
        if t.num_rows:
            batches.append(t)
    act = pa.concat_tables(batches)
    print(f"rows in {n_in:,} -> kept {act.num_rows:,}")

    # distinct (target, molecule) pairs
    pairs = act.select(["target_chembl_id", "molecule_chembl_id"]).group_by(
        ["target_chembl_id", "molecule_chembl_id"]).aggregate([])
    counts = pairs.group_by("target_chembl_id").aggregate(
        [("molecule_chembl_id", "count")]).sort_by(
        [("molecule_chembl_id_count", "descending")])
    counts = counts.rename_columns(["target_chembl_id", "n_molecules"])

    pq.write_table(act, os.path.join(args.out_dir, "activities_filtered.parquet"))
    pacsv.write_csv(counts, os.path.join(args.out_dir, "target_counts.csv"))
    print(f"targets: {counts.num_rows:,}")
    d = counts.to_pydict()
    for n in (10000, 5000, 2000, 1000):
        k = sum(1 for c in d["n_molecules"] if c >= n)
        print(f"  targets with >= {n:>6,} distinct molecules: {k}")
    print("\ntop 30 targets:")
    for tid, c in list(zip(d["target_chembl_id"], d["n_molecules"]))[:30]:
        print(f"  {tid:16s} {c:7,}")


if __name__ == "__main__":
    main()
