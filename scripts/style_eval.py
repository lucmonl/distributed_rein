"""House style of generated summaries, controlled for extractiveness (CPU only).

    python scripts/style_eval.py --run Method=runs/A --run A2=runs/B::eval_round_0070__*.json
    python scripts/style_eval.py --run Method=runs/A --split dev --rounds 10,50,100

Does a client's model write like its own publication, beyond what extractiveness explains?

1. Attribution.  A publication classifier (char 2-5 + word 1-2 tf-idf, logistic regression) is
   trained on *real* training summaries of the run's participants, **density-matched**: within
   each of 5 global-alpha bins every publication present contributes the same number of
   summaries, so extractiveness cannot identify the publication.  Applied to each client's
   generated summaries at in-support alphas: accuracy (argmax = own publication) and mean
   probability of the own publication.  Topic comes from the article and is the same for every
   model (same articles), so differences between models are the style the model adds.  Real
   test summaries give the ceiling; a shared adapter (A2) the no-personalization floor.
2. Style-feature gap.  Interpretable features (sentences, words per sentence, word length,
   quotes, ellipsis, parentheses, numerals, capitalized words, first/second person, ? and !).
   For each client and in-support alpha: |standardized mean difference| between the generated
   summaries and the client's real training summaries within +-0.125 alpha, averaged over
   features (lower = closer to the house style).

The classifier is cached in runs/_style/.  Writes <run>/evals/style_<eval file>.json.
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import random
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.data import alpha_reference_from_json, read_jsonl  # noqa: E402

BINS = np.linspace(0, 1, 6)
FEATURES = ("sentences", "words_per_sentence", "word_len", "quotes", "ellipsis", "parens", "numerals",
            "capitalized", "first_person", "second_person", "question_excl")


def abin(a):
    return min(int(np.digitize(a, BINS)) - 1, len(BINS) - 2)


def style_features(t):
    words = re.findall(r"[A-Za-z0-9']+", t)
    nw = max(len(words), 1)
    sents = max(len(re.findall(r"[.!?]+(\s|$)", t)), 1)
    return np.array([
        sents, nw / sents, np.mean([len(w) for w in words]) if words else 0.0,
        len(re.findall(r"[\"“”‘’]", t)) / nw, float("..." in t or "…" in t), t.count("(") / nw,
        sum(w.isdigit() for w in words) / nw, sum(w[:1].isupper() for w in words) / nw,
        sum(w.lower() in ("i", "we", "my", "our", "me", "us") for w in words) / nw,
        sum(w.lower() in ("you", "your") for w in words) / nw, len(re.findall(r"[?!]", t)) / nw,
    ], dtype=float)


def train_classifier(records, clients, ref, per_bin, seed, cache_dir):
    key = hashlib.md5(json.dumps([sorted(clients), per_bin, seed, len(records)]).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"style_clf_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb")), path
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    rng = random.Random(seed)
    texts, labels = [], []
    for k in range(len(BINS) - 1):
        by = {c: [r["target"] for r in records if r["client"] == c and r["split"] == "train" and r["bin"] == k]
              for c in clients}
        by = {c: v for c, v in by.items() if len(v) >= 30}
        m = min(min(len(v) for v in by.values()), per_bin)
        for c, v in by.items():
            texts += rng.sample(v, m)
            labels += [c] * m
    clf = make_pipeline(make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True),
                                   TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                        LogisticRegression(max_iter=3000, C=4))
    clf.fit(texts, labels)
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(clf, open(path, "wb"))
    return clf, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path[::eval glob]")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--rounds", default=None, help="dev only: comma list of rounds (default: all)")
    ap.add_argument("--per_bin", type=int, default=400, help="training summaries per publication per bin")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    specs = []
    for spec in args.run:
        name, path = spec.split("=", 1)
        path, _, pat = path.partition("::")
        specs.append((name, path, pat))
    cfg = yaml.safe_load(open(os.path.join(specs[0][1], "config.yaml")))
    clients = cfg["clients"]
    ref = alpha_reference_from_json(json.load(open(os.path.join(specs[0][1], "alpha_reference.json"))))
    records = [r for r in read_jsonl(cfg["data_path"]) if r["client"] in clients and r.get("split") in ("train", "test")]
    for r in records:
        r["alpha"] = ref.cdf(r["score"])
        r["bin"] = abin(r["alpha"])
    clf, clf_path = train_classifier(records, clients, ref, args.per_bin, args.seed, os.path.join("runs", "_style"))
    classes = list(clf.classes_)
    print(f"classifier: {clf_path}", flush=True)

    # house-style feature references: each client's real training summaries
    train_feats = {c: [(r["alpha"], style_features(r["target"])) for r in records
                       if r["client"] == c and r["split"] == "train"] for c in clients}
    allf = np.stack([f for v in train_feats.values() for _, f in v])
    scale = allf.std(0) + 1e-6

    def feature_gap(c, a, texts):
        refs = np.stack([f for x, f in train_feats[c] if abs(x - a) <= 0.125] or [f for _, f in train_feats[c]])
        gen = np.stack([style_features(t) for t in texts])
        return float(np.mean(np.abs(gen.mean(0) - refs.mean(0)) / scale))

    def attribution(c, texts):
        p = clf.predict_proba(texts)[:, classes.index(c)]
        pred = [classes[i] for i in clf.predict_proba(texts).argmax(1)]
        return float(np.mean([x == c for x in pred])), float(p.mean())

    # ceiling: real test summaries
    real = {}
    for c in clients:
        texts = [r["target"] for r in records if r["client"] == c and r["split"] == "test"]
        real[c] = attribution(c, texts)
    print(f"\nreal test summaries (ceiling): attribution acc {np.mean([v[0] for v in real.values()]):.3f}, "
          f"P(own) {np.mean([v[1] for v in real.values()]):.3f}  (chance {1 / len(clients):.3f})")

    rows = []
    for name, path, pat in specs:
        if args.split == "test":
            pat = pat or "eval_round_[0-9][0-9][0-9][0-9]__*.json"
            files = [max(glob.glob(os.path.join(path, "evals", pat)), key=os.path.getmtime)]
        else:
            files = sorted(glob.glob(os.path.join(path, "evals", "eval_round_*_dev__*.json")))
            if args.rounds:
                want = {int(x) for x in args.rounds.split(",")}
                files = [f for f in files if int(re.search(r"round_(\d+)", f).group(1)) in want]
        for f in files:
            ev = json.load(open(f))
            res = {}
            for c, v in ev["clients"].items():
                if c not in clients or "outputs" not in v:
                    continue
                lo, hi = v.get("support", [0.0, 1.0])
                per_alpha = {}
                for j, a in enumerate(ev["alphas"]):
                    texts = [o[j] for o in v["outputs"] if o[j].strip()]
                    if not texts:
                        continue
                    acc, pown = attribution(c, texts)
                    per_alpha[a] = {"acc": acc, "p_own": pown, "feature_gap": feature_gap(c, a, texts),
                                    "in_support": lo <= a <= hi}
                ins = [x for x in per_alpha.values() if x["in_support"]] or list(per_alpha.values())
                res[c] = {"per_alpha": per_alpha,
                          **{f"{k}_in": float(np.mean([x[k] for x in ins])) for k in ("acc", "p_own", "feature_gap")}}
            summ = {k: float(np.mean([v[k] for v in res.values()])) for k in ("acc_in", "p_own_in", "feature_gap_in")}
            out = {"eval": f, "classifier": clf_path, "real_test": real, "clients": res, "summary": summ}
            json.dump(out, open(os.path.join(os.path.dirname(f), "style_" + os.path.basename(f)), "w"), indent=1)
            rows.append((name, ev["round"], summ, res))

    print(f"\n{'run':14s} {'round':>5s} {'attr acc':>8s} {'P(own)':>7s} {'feat gap':>8s}   (in-support alphas; mean over clients)")
    for name, rnd, s, _ in rows:
        print(f"{name:14s} {rnd:5d} {s['acc_in']:8.3f} {s['p_own_in']:7.3f} {s['feature_gap_in']:8.3f}")
    if args.split == "test":
        print("\nper-client attribution accuracy (in-support)")
        print(f"{'run':14s} " + " ".join(f"{c[:10]:>10s}" for c in clients))
        print(f"{'real test':14s} " + " ".join(f"{real[c][0]:10.3f}" for c in clients))
        for name, rnd, _, res in rows:
            print(f"{name:14s} " + " ".join(f"{res[c]['acc_in']:10.3f}" if c in res else f"{'-':>10s}" for c in clients))
        print("\nattribution accuracy by target alpha (all clients; checks that it does not track extractiveness)")
        alphas = sorted({a for _, _, _, res in rows for v in res.values() for a in v["per_alpha"]})
        print(f"{'run':14s} " + " ".join(f"{a:>7.2f}" for a in alphas))
        for name, rnd, _, res in rows:
            print(f"{name:14s} " + " ".join(f"{np.mean([v['per_alpha'][a]['acc'] for v in res.values() if a in v['per_alpha']]):7.3f}"
                                           for a in alphas))


if __name__ == "__main__":
    main()
