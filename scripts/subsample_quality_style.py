"""Std of house style and summary quality by test-set subsampling (needs a GPU for AlignScore/BERTScore).

    python scripts/subsample_quality_style.py \\
        --arm ARM=runs/<run>/evals/eval_round_0100__<stamp>.json --arm ... \\
        [--seeds 0,1,2] [--frac 0.75] [--device cuda] [--json out.json]

Companion of scripts/subsample_std.py (same subsets: drawn per seed and client from the article order,
`subsample_std.draw_subsets`). Every summary is scored once (cached per eval file in
runs/_subsample_items/), then each metric is aggregated per subset exactly as the original scripts do:

  quality (scripts/score_quality.py): AlignScore vs the article, BERTScore F1 vs the reference summary,
      length, repeated-trigram rate; per client and alpha, then in-/out-of-support means over alphas,
      then the mean over clients. gap_* = generated - real summaries binned to the nearest alpha
      (the real summaries of the same subset articles of all clients).
  house style (scripts/style_eval.py, same cached density-matched classifier): attribution accuracy,
      P(own publication), style-feature gap vs the client's real training summaries within +-0.125 alpha;
      in-support alphas, mean over clients.

Ceiling: the real test summaries of the same subset articles (attribution, P(own), AlignScore, length).
The full-test value is printed next to each mean +- std as a check against the existing reports.
The std is over subsets: label it "<n> subsets x <frac> of test articles"; it reflects test sampling only.
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fedsteer.data import alpha_reference_from_json, read_jsonl  # noqa: E402
from style_eval import abin, style_features, train_classifier  # noqa: E402
from subsample_std import draw_subsets  # noqa: E402

QM = ("align", "bert_f1", "length", "rep3")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def score_items(name, path, ev, by_url, clf, classes, align, bert, cache_dir):
    """Per-summary scores [article][alpha] for every client, cached."""
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, os.path.basename(os.path.dirname(os.path.dirname(path))) + "__"
                         + os.path.basename(path))
    if os.path.exists(cpath):
        return json.load(open(cpath))
    from fedsteer.quality import surface
    items = {}
    for c, res in ev["clients"].items():
        texts, recs = res["outputs"], [by_url[u] for u in res["record_ids"]]
        flat = [(i, j) for i in range(len(texts)) for j in range(len(ev["alphas"]))]
        cands = [texts[i][j] for i, j in flat]
        al = align.score([recs[i]["article"] for i, _ in flat], cands)
        bs = bert.score(cands, [recs[i]["target"] for i, _ in flat])
        nonempty = [t if t.strip() else " " for t in cands]
        proba = clf.predict_proba(nonempty)
        k = len(ev["alphas"])
        grid = lambda vals: [vals[i * k:(i + 1) * k] for i in range(len(texts))]
        items[c] = {"align": grid([float(a) for a in al]), "bert_f1": grid([float(b["F1"]) for b in bs]),
                    "length": grid([surface(t)["length"] for t in cands]),
                    "rep3": grid([surface(t)["rep3"] for t in cands]),
                    "empty": grid([not t.strip() for t in cands]),
                    "correct": grid([classes[p.argmax()] == c for p in proba]),
                    "p_own": grid([float(p[classes.index(c)]) for p in proba]),
                    "feat": grid([style_features(t).tolist() for t in cands])}
        print(f"  scored {name}/{c}: {len(cands)} summaries", flush=True)
    json.dump(items, open(cpath, "w"))
    return items


def aggregate(ev, items, real, idx, alphas, train_feats, scale, feat_refs):
    """One subset (idx: client -> row indices): quality + style summaries, mean over clients."""
    # reference bins from the real summaries of the subset articles of all clients (score_quality)
    rb = defaultdict(lambda: defaultdict(list))
    for c, ix in idx.items():
        for i in ix:
            j = int(np.argmin([abs(real[c]["alpha"][i] - a) for a in alphas]))
            for m in ("align", "length"):
                rb[j][m].append(real[c][m][i])
    rb = {j: {m: mean(v) for m, v in d.items()} for j, d in rb.items()}
    q_cl, s_cl, ceil = {}, {}, {}
    for c, ix in idx.items():
        it, (lo, hi) = items[c], ev["clients"][c]["support"]
        ins = [lo <= a <= hi for a in alphas]
        per_alpha = [{m: mean([it[m][i][j] for i in ix]) for m in QM} for j in range(len(alphas))]
        summ = {}
        for tag, sel in (("in", ins), ("out", [not x for x in ins])):
            js = [j for j, s in enumerate(sel) if s]
            for m in QM:
                summ[f"{m}_{tag}"] = mean([per_alpha[j][m] for j in js]) if js else None
            for m in ("align", "length"):
                g = [per_alpha[j][m] - rb[j][m] for j in js if j in rb]
                summ[f"gap_{m}_{tag}"] = mean(g) if (js and g) else None
        q_cl[c] = summ
        # style (style_eval): per alpha over non-empty texts, then in-support alphas
        pa = []
        for j, a in enumerate(alphas):
            keep = [i for i in ix if not it["empty"][i][j]]
            if not keep:
                continue
            gen = np.array([it["feat"][i][j] for i in keep])
            pa.append({"in": ins[j], "acc": float(np.mean([it["correct"][i][j] for i in keep])),
                       "p_own": float(np.mean([it["p_own"][i][j] for i in keep])),
                       "feature_gap": float(np.mean(np.abs(gen.mean(0) - feat_refs[c][j]) / scale))})
        sel = [x for x in pa if x["in"]] or pa
        s_cl[c] = {k: float(np.mean([x[k] for x in sel])) for k in ("acc", "p_own", "feature_gap")}
        ceil[c] = {"acc": float(np.mean([real[c]["correct"][i] for i in ix])),
                   "p_own": float(np.mean([real[c]["p_own"][i] for i in ix])),
                   "align": float(np.mean([real[c]["align"][i] for i in ix])),
                   "length": float(np.mean([real[c]["length"][i] for i in ix]))}
    out = {f"q_{k}": mean([v[k] for v in q_cl.values()]) for k in next(iter(q_cl.values()))}
    out.update({f"style_{k}": float(np.mean([v[k] for v in s_cl.values()])) for k in ("acc", "p_own", "feature_gap")})
    out.update({f"real_{k}": float(np.mean([v[k] for v in ceil.values()])) for k in ("acc", "p_own", "align", "length")})
    out["per_client"] = {c: {"quality": q_cl[c], "style": s_cl[c], "real": ceil[c]} for c in idx}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=path/to/eval.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--frac", type=float, default=0.75)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache_dir", default=os.path.join("runs", "_subsample_items"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = (path, json.load(open(path)))
    first_run = os.path.dirname(os.path.dirname(os.path.abspath(next(iter(arms.values()))[0])))
    cfg = yaml.safe_load(open(os.path.join(first_run, "config.yaml")))
    ref_json = json.load(open(os.path.join(first_run, "alpha_reference.json")))
    ref = alpha_reference_from_json(ref_json)

    # same checks as subsample_std: identical clients, article ids and alpha reference
    base = next(iter(arms.values()))[1]
    for name, (path, ev) in arms.items():
        run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        if json.load(open(os.path.join(run, "alpha_reference.json"))) != ref_json:
            sys.exit(f"REFUSING: {name} has a different alpha reference")
        for c in base["clients"]:
            if c not in ev["clients"] or ev["clients"][c]["record_ids"] != base["clients"][c]["record_ids"]:
                sys.exit(f"REFUSING: {name}/{c}: test articles differ")
            if "outputs" not in ev["clients"][c]:
                sys.exit(f"REFUSING: {name}/{c}: no saved outputs")

    clients = cfg["clients"]
    records = [r for r in read_jsonl(cfg["data_path"]) if r["client"] in clients and r.get("split") in ("train", "test")]
    for r in records:
        r["alpha"] = ref.cdf(r["score"])
        r["bin"] = abin(r["alpha"])
    by_url = {r["url"]: r for r in records}
    clf, clf_path = train_classifier(records, clients, ref, 400, 0, os.path.join("runs", "_style"))
    classes = list(clf.classes_)
    print(f"classifier: {clf_path}", flush=True)
    train_feats = {c: [(r["alpha"], style_features(r["target"])) for r in records
                       if r["client"] == c and r["split"] == "train"] for c in clients}
    scale = np.stack([f for v in train_feats.values() for _, f in v]).std(0) + 1e-6
    alphas = base["alphas"]
    feat_refs = {c: [np.stack([f for x, f in train_feats[c] if abs(x - a) <= 0.125]
                              or [f for _, f in train_feats[c]]).mean(0) for a in alphas] for c in clients}

    from fedsteer.quality import AlignScorer, BERTScorer, surface
    align, bert = AlignScorer(device=args.device), BERTScorer(device=args.device)
    items = {name: score_items(name, path, ev, by_url, clf, classes, align, bert, args.cache_dir)
             for name, (path, ev) in arms.items()}

    # real summaries of the same articles (ceiling and reference bins)
    real = {}
    for c in base["clients"]:
        recs = [by_url[u] for u in base["clients"][c]["record_ids"]]
        proba = clf.predict_proba([r["target"] for r in recs])
        real[c] = {"alpha": [r["alpha"] for r in recs],
                   "align": [float(a) for a in align.score([r["article"] for r in recs], [r["target"] for r in recs])],
                   "length": [surface(r["target"])["length"] for r in recs],
                   "correct": [classes[p.argmax()] == c for p in proba],
                   "p_own": [float(p[classes.index(c)]) for p in proba]}

    n = {c: len(base["clients"][c]["record_ids"]) for c in base["clients"]}
    subsets = draw_subsets(n, seeds, args.frac)
    full = {c: np.arange(n[c]) for c in n}
    res = {name: {"full": aggregate(ev, items[name], real, full, alphas, train_feats, scale, feat_refs),
                  "subsets": [aggregate(ev, items[name], real, subsets[s], alphas, train_feats, scale, feat_refs)
                              for s in seeds]}
           for name, (_, ev) in arms.items()}

    label = f"{len(seeds)} subsets x {args.frac:.0%} of test articles"
    keys = [k for k in next(iter(res.values()))["full"] if k != "per_client"]
    print(f"\n== quality and house style, mean +- std over {label} (full-test value in brackets)")
    for k in keys:
        print(f"\n-- {k}")
        names = list(arms) if not k.startswith("real_") else [next(iter(arms))]
        for name in names:
            v = np.array([r[k] for r in res[name]["subsets"]], dtype=float)
            f = res[name]["full"][k]
            tag = "real summaries (ceiling)" if k.startswith("real_") else name
            print(f"   {tag:28s} {v.mean():.3f} +- {v.std(ddof=1):.3f}   [{f:.3f}]")
    if args.json:
        json.dump({"label": label, "seeds": seeds, "frac": args.frac, "classifier": clf_path,
                   "files": {k: v[0] for k, v in arms.items()}, "results": res}, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
