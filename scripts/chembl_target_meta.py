"""Step 2: fetch ChEMBL metadata (name, organism, protein class) for the
most-populated targets. Writes data/chembl/target_meta.csv.
"""
import argparse, csv, json, os, time, urllib.request

API = "https://www.ebi.ac.uk/chembl/api/data"


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                print(f"  ! failed {url}: {e}")
                return None
            time.sleep(2 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="data/chembl/target_counts.csv")
    ap.add_argument("--out", default="data/chembl/target_meta.csv")
    ap.add_argument("--top", type=int, default=200)
    args = ap.parse_args()

    with open(args.counts) as f:
        rows = list(csv.DictReader(f))[: args.top]
    tids = [r["target_chembl_id"] for r in rows]
    counts = {r["target_chembl_id"]: int(r["n_molecules"]) for r in rows}

    targets = {}
    for i in range(0, len(tids), 20):
        chunk = tids[i : i + 20]
        d = get(f"{API}/target.json?target_chembl_id__in={','.join(chunk)}&limit=20")
        for t in (d or {}).get("targets", []):
            targets[t["target_chembl_id"]] = t
        print(f"  targets {i + len(chunk)}/{len(tids)}")

    class_cache, comp_cache = {}, {}

    def class_of(accession):
        if not accession:
            return ""
        if accession not in comp_cache:
            d = get(f"{API}/target_component.json?accession={accession}&limit=1")
            comps = (d or {}).get("target_components", [])
            cids = [c["protein_classification_id"]
                    for c in (comps[0].get("protein_classifications", []) if comps else [])]
            comp_cache[accession] = cids[0] if cids else None
        cid = comp_cache[accession]
        if cid is None:
            return ""
        if cid not in class_cache:
            d = get(f"{API}/protein_classification/{cid}.json")
            class_cache[cid] = (d or {}).get("protein_class_desc", "")
        return class_cache[cid]

    out = []
    for tid in tids:
        t = targets.get(tid, {})
        comps = t.get("target_components", [])
        acc = comps[0].get("accession") if comps else None
        desc = class_of(acc)
        parts = desc.split()
        out.append({
            "target_chembl_id": tid,
            "n_molecules": counts[tid],
            "pref_name": t.get("pref_name", ""),
            "organism": t.get("organism", ""),
            "target_type": t.get("target_type", ""),
            "accession": acc or "",
            "protein_class_desc": desc,
            "class_l1": parts[0] if parts else "",
            "class_l2": parts[1] if len(parts) > 1 else "",
            "class_l3": parts[2] if len(parts) > 2 else "",
        })
        print(f"  {tid:16s} {counts[tid]:6,} {t.get('pref_name','?')[:46]:46s} | {desc}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {args.out} ({len(out)} targets)")


if __name__ == "__main__":
    main()
