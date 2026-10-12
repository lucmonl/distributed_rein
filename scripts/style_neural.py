"""House-style attribution with neural classifiers (NR-79, CAL-34); needs a GPU.

    python scripts/style_neural.py --ref OURS \\
        --arm OURS=runs/<run>/evals/eval_round_0100__<stamp>.json --arm ... [--classifiers roberta,qwen,tfidf] \\
        [--json out.json]

Same task and training data as scripts/style_eval.py; only the classifier changes:
  * training set: REAL training summaries of the run's participants, density-matched exactly as
    style_eval.train_classifier (5 global-alpha bins, the same publications kept per bin, <= per_bin
    summaries per publication and bin, the same random.Random(seed) draw), so extractiveness cannot
    identify the publication. A validation split (val_frac, stratified by publication x bin) is held out
    of these TRAINING summaries for early stopping and the learning rate; test is never used for tuning.
  * classifiers, all on the raw summary text only (no hand-built features, no article):
      roberta  FacebookAI/roberta-large, fully fine-tuned, 8-way sequence classification
      qwen     Qwen/Qwen3-4B-Instruct-2507 + LoRA + classification head (bf16)
      tfidf    style_eval's cached char/word tf-idf + logistic regression (reference row)
    No truncation: max_length is the longest tokenized text (training, validation, generated, real);
    the truncation rate is reported.
  * evaluation as style_eval: per client and alpha, non-empty generated summaries -> attribution accuracy
    (argmax = own publication) and mean P(own); in-support alphas; mean over clients. Ceiling: the real
    test summaries of the same articles.
  * std over 3 subsets x 75% of test articles (subsample_std.draw_subsets, the same subsets as NR-76/77);
    classifiers are trained once (fixed seed). Per-subset differences vs --ref are reported.
Models are cached in runs/_style/neural_<classifier>_<key>/.
"""
import argparse, hashlib, json, os, random, sys, time
from collections import Counter, defaultdict
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fedsteer.data import alpha_reference_from_json, read_jsonl  # noqa: E402
from style_eval import BINS, abin, train_classifier  # noqa: E402
from subsample_std import draw_subsets  # noqa: E402

MODELS = {"roberta": "FacebookAI/roberta-large", "qwen": "Qwen/Qwen3-4B-Instruct-2507"}


def density_matched(records, clients, per_bin, seed):
    """style_eval.train_classifier's training draw, with the bin of every summary."""
    rng = random.Random(seed)
    texts, labels, bins = [], [], []
    for k in range(len(BINS) - 1):
        by = {c: [r["target"] for r in records if r["client"] == c and r["split"] == "train" and r["bin"] == k]
              for c in clients}
        by = {c: v for c, v in by.items() if len(v) >= 30}
        m = min(min(len(v) for v in by.values()), per_bin)
        for c, v in by.items():
            texts += rng.sample(v, m)
            labels += [c] * m
            bins += [k] * m
    return texts, labels, bins


def split_val(n_labels_bins, val_frac, seed):
    """Stratified (publication x bin) validation indices."""
    rng = random.Random(seed + 1)
    strata = defaultdict(list)
    for i, key in enumerate(n_labels_bins):
        strata[key].append(i)
    val = []
    for key in sorted(strata):
        idx = strata[key][:]
        rng.shuffle(idx)
        val += idx[:max(1, int(round(val_frac * len(idx))))]
    return sorted(val)


def add_lora(model, targets, r=16, alpha=32, dropout=0.05):
    """Minimal LoRA (in place, no peft): freeze everything, wrap the target nn.Linear layers with
    W x + (alpha / r) * B A dropout(x), B = 0 at init; the classification head ("score") stays trainable.
    (The steer env's peft dispatches q_proj to a foreign SteerLinear class, so peft is not used.)"""
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            self.drop, self.scale = nn.Dropout(dropout), alpha / r

        def forward(self, x):
            h = self.drop(x).to(self.lora_A.dtype)
            return self.base(x) + ((h @ self.lora_A.T) @ self.lora_B.T * self.scale).to(x.dtype)

    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    for n in names:
        parent, _, child = n.rpartition(".")
        mod = model.get_submodule(parent)
        setattr(mod, child, LoRALinear(getattr(mod, child)))
    for n, p in model.named_parameters():
        if "lora_" in n or n.startswith("score"):
            p.requires_grad_(True)
            if n.startswith("score"):
                p.data = p.data.float()
    return model


class NeuralClf:
    def __init__(self, kind, classes, device, max_length):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.kind, self.classes, self.device = torch, kind, classes, device
        self.tok = AutoTokenizer.from_pretrained(MODELS[kind])
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_length = max_length
        dtype = torch.bfloat16 if kind == "qwen" else torch.float32
        m = AutoModelForSequenceClassification.from_pretrained(MODELS[kind], num_labels=len(classes), torch_dtype=dtype)
        m.config.pad_token_id = self.tok.pad_token_id
        if kind == "qwen":
            m.gradient_checkpointing_enable()
            m.enable_input_require_grads()
            m = add_lora(m, {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
        self.model = m.to(device)

    def _batches(self, texts, bs):
        order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), bs):
            idx = order[s:s + bs]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            yield idx, enc

    def proba(self, texts, bs=64):
        torch = self.torch
        self.model.eval()
        out = np.zeros((len(texts), len(self.classes)), dtype=np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for idx, enc in self._batches(texts, bs):
                out[idx] = torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy()
        return out

    def fit(self, tr_texts, tr_y, va_texts, va_y, lr, epochs, bs, accum, seed, log):
        torch = self.torch
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        steps = epochs * int(np.ceil(len(tr_texts) / (bs * accum)))
        warm = max(1, int(0.06 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(            # linear warm-up, then linear decay
            opt, lambda t: (t + 1) / warm if t < warm else max(0.0, (steps - t) / max(1, steps - warm)))
        y = torch.tensor(tr_y)
        best, best_state, bad = -1.0, None, 0
        for ep in range(epochs):
            self.model.train()
            perm = rng.permutation(len(tr_texts))
            t0, k = time.time(), 0
            for s in range(0, len(perm), bs):
                idx = perm[s:s + bs]
                enc = self.tok([tr_texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(**enc).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, y[idx].to(self.device)) / accum
                loss.backward()
                k += 1
                if k % accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            p = self.proba(va_texts)
            acc = float((p.argmax(1) == np.array(va_y)).mean())
            log(f"   {self.kind} lr={lr:g} epoch {ep + 1}/{epochs}: val acc {acc:.4f} ({time.time() - t0:.0f}s)")
            if acc > best:
                best, bad = acc, 0
                best_state = {n: v.detach().cpu().clone() for n, v in self.model.state_dict().items()
                              if self.kind == "roberta" or "lora_" in n or "score" in n}
            else:
                bad += 1
                if bad >= 1:                                   # early stopping: patience 1
                    break
        self.model.load_state_dict(best_state, strict=False)
        return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=path/to/eval.json")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--classifiers", default="roberta,qwen,tfidf")
    ap.add_argument("--per_bin", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--lrs_roberta", default="1e-5,2e-5")
    ap.add_argument("--lrs_qwen", default="5e-5,1e-4")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--frac", type=float, default=0.75)
    ap.add_argument("--cache_dir", default=os.path.join("runs", "_style"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = (path, json.load(open(path)))
    first = os.path.dirname(os.path.dirname(os.path.abspath(next(iter(arms.values()))[0])))
    cfg = yaml.safe_load(open(os.path.join(first, "config.yaml")))
    ref_json = json.load(open(os.path.join(first, "alpha_reference.json")))
    ref = alpha_reference_from_json(ref_json)
    clients = cfg["clients"]
    base = arms[args.ref][1]
    for name, (path, ev) in arms.items():
        run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        if json.load(open(os.path.join(run, "alpha_reference.json"))) != ref_json:
            sys.exit(f"REFUSING: {name} has a different alpha reference")
        for c in clients:
            if ev["clients"].get(c, {}).get("record_ids") != base["clients"][c]["record_ids"]:
                sys.exit(f"REFUSING: {name}/{c}: test articles differ")
    records = [r for r in read_jsonl(cfg["data_path"]) if r["client"] in clients and r.get("split") in ("train", "test")]
    for r in records:
        r["alpha"] = ref.cdf(r["score"])
        r["bin"] = abin(r["alpha"])
    by_url = {r["url"]: r for r in records}
    classes = sorted(clients)
    cidx = {c: i for i, c in enumerate(classes)}

    texts, labels, bins = density_matched(records, clients, args.per_bin, args.seed)
    val = set(split_val(list(zip(labels, bins)), args.val_frac, args.seed))
    tr = [i for i in range(len(texts)) if i not in val]
    va = sorted(val)
    log(f"density-matched training summaries: {len(texts)} ({len(tr)} train / {len(va)} validation); "
        f"per publication: {dict(Counter(labels))}")
    real = {c: [by_url[u]["target"] for u in base["clients"][c]["record_ids"]] for c in clients}
    gen_texts = sorted({t for _, ev in arms.values() for c in clients for row in ev["clients"][c]["outputs"] for t in row})

    probs, report = {}, {"classifiers": {}}
    for kind in args.classifiers.split(","):
        log(f"\n=== classifier {kind}")
        if kind == "tfidf":
            clf, path = train_classifier(records, clients, ref, args.per_bin, args.seed, args.cache_dir)
            order = [list(clf.classes_).index(c) for c in classes]
            pred = lambda xs: clf.predict_proba(xs)[:, order]
            report["classifiers"][kind] = {"model": path, "note": "style_eval's cached classifier (trained on all density-matched summaries)"}
        else:
            import torch
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODELS[kind])
            lens = lambda xs: [len(tok(x, add_special_tokens=True)["input_ids"]) for x in xs]
            all_lens = lens(texts) + lens(gen_texts) + lens([t for c in clients for t in real[c]])
            max_len = int(max(all_lens))
            limit = 512 if kind == "roberta" else 32768
            trunc = sum(l > limit for l in all_lens)
            max_length = min(max_len, limit)
            log(f"   token lengths: max {max_len}, 99.9th pct {np.percentile(all_lens, 99.9):.0f}; max_length {max_length}; "
                f"truncated {trunc} of {len(all_lens)} texts")
            key = hashlib.md5(json.dumps([classes, args.per_bin, args.seed, args.val_frac, len(texts), kind,
                                          args.lrs_roberta if kind == "roberta" else args.lrs_qwen,
                                          args.epochs]).encode()).hexdigest()[:10]
            cdir = os.path.join(args.cache_dir, f"neural_{kind}_{key}")
            info_path = os.path.join(cdir, "info.json")
            if os.path.exists(info_path):
                info = json.load(open(info_path))
                clf = NeuralClf(kind, classes, "cuda", max_length)
                clf.model.load_state_dict(torch.load(os.path.join(cdir, "state.pt")), strict=False)
                log(f"   loaded cached {cdir} (val acc {info['val_acc']:.4f}, lr {info['lr']})")
            else:
                tr_t, tr_y = [texts[i] for i in tr], [cidx[labels[i]] for i in tr]
                va_t, va_y = [texts[i] for i in va], [cidx[labels[i]] for i in va]
                lrs = [float(x) for x in (args.lrs_roberta if kind == "roberta" else args.lrs_qwen).split(",")]
                bs, accum = (32, 1) if kind == "roberta" else (16, 2)
                best_acc, best_lr, best_state = -1.0, None, None
                for lr in lrs:
                    clf = NeuralClf(kind, classes, "cuda", max_length)
                    acc = clf.fit(tr_t, tr_y, va_t, va_y, lr, args.epochs, bs, accum, args.seed, log)
                    if acc > best_acc:
                        best_acc, best_lr = acc, lr
                        best_state = {n: v.detach().cpu().clone() for n, v in clf.model.state_dict().items()
                                      if kind == "roberta" or "lora_" in n or "score" in n}
                    del clf
                    torch.cuda.empty_cache()
                acc, lr = best_acc, best_lr
                clf = NeuralClf(kind, classes, "cuda", max_length)
                clf.model.load_state_dict(best_state, strict=False)
                os.makedirs(cdir, exist_ok=True)
                state = {n: v.detach().cpu() for n, v in clf.model.state_dict().items()
                         if kind == "roberta" or "lora_" in n or "score" in n}
                torch.save(state, os.path.join(cdir, "state.pt"))
                info = {"val_acc": acc, "lr": lr, "max_length": max_length}
                json.dump(info, open(info_path, "w"))
                log(f"   selected lr {lr:g} (val acc {acc:.4f}); saved {cdir}")
            pred = lambda xs, clf=clf: clf.proba(xs)
            report["classifiers"][kind] = {"model": MODELS[kind], "cache": cdir, **info,
                                           "max_token_length": max_len, "truncated_texts": int(trunc)}
        # validation diagnostics (density-matched real TRAINING summaries held out; tf-idf saw them in training)
        vp = pred([texts[i] for i in va])
        vy = np.array([cidx[labels[i]] for i in va])
        vb = np.array([bins[i] for i in va])
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(vy, vp.argmax(1)):
            cm[t, p] += 1
        report["classifiers"][kind].update({
            "val_acc_overall": float((vp.argmax(1) == vy).mean()),
            "val_acc_by_bin": {int(b): float((vp.argmax(1)[vb == b] == vy[vb == b]).mean()) for b in sorted(set(vb))},
            "confusion_matrix": {"rows_true_cols_pred": classes, "counts": cm.tolist()}})
        log(f"   validation acc {report['classifiers'][kind]['val_acc_overall']:.4f}; by alpha bin "
            + ", ".join(f"{b}: {a:.3f}" for b, a in report["classifiers"][kind]["val_acc_by_bin"].items()))
        # scores for every generated and real summary
        P = {}
        for name, (_, ev) in arms.items():
            flat = [(c, i, j) for c in clients for i, row in enumerate(ev["clients"][c]["outputs"]) for j in range(len(row))]
            tx = [ev["clients"][c]["outputs"][i][j] for c, i, j in flat]
            pp = pred([t if t.strip() else " " for t in tx])
            g = {c: [[None] * len(ev["alphas"]) for _ in ev["clients"][c]["outputs"]] for c in clients}
            for (c, i, j), t, p in zip(flat, tx, pp):
                g[c][i][j] = None if not t.strip() else (bool(p.argmax() == cidx[c]), float(p[cidx[c]]))
            P[name] = g
        rp = {c: [(bool(p.argmax() == cidx[c]), float(p[cidx[c]])) for p in pred(real[c])] for c in clients}
        probs[kind] = (P, rp)

    n = {c: len(base["clients"][c]["record_ids"]) for c in clients}
    subsets = draw_subsets(n, seeds, args.frac)
    full = {c: np.arange(n[c]) for c in clients}

    def agg(kind, name, idx):
        P, _ = probs[kind]
        ev = arms[name][1]
        per = {}
        for c in clients:
            lo, hi = ev["clients"][c]["support"]
            pa = []
            for j, a in enumerate(ev["alphas"]):
                v = [P[name][c][i][j] for i in idx[c] if P[name][c][i][j] is not None]
                if v:
                    pa.append((lo <= a <= hi, np.mean([x[0] for x in v]), np.mean([x[1] for x in v])))
            sel = [x for x in pa if x[0]] or pa
            per[c] = {"acc": float(np.mean([x[1] for x in sel])), "p_own": float(np.mean([x[2] for x in sel]))}
        return {"acc": float(np.mean([v["acc"] for v in per.values()])),
                "p_own": float(np.mean([v["p_own"] for v in per.values()])), "per_client": per}

    def agg_real(kind, idx):
        _, rp = probs[kind]
        per = {c: {"acc": float(np.mean([rp[c][i][0] for i in idx[c]])), "p_own": float(np.mean([rp[c][i][1] for i in idx[c]]))}
               for c in clients}
        return {"acc": float(np.mean([v["acc"] for v in per.values()])),
                "p_own": float(np.mean([v["p_own"] for v in per.values()])), "per_client": per}

    label = f"{len(seeds)} subsets x {args.frac:.0%} of test articles"
    res = {}
    for kind in probs:
        res[kind] = {name: {"full": agg(kind, name, full), "subsets": [agg(kind, name, subsets[s]) for s in seeds]}
                     for name in arms}
        res[kind]["real summaries"] = {"full": agg_real(kind, full), "subsets": [agg_real(kind, subsets[s]) for s in seeds]}
    ms = lambda R, k: (np.mean([s[k] for s in R["subsets"]]), np.std([s[k] for s in R["subsets"]], ddof=1), R["full"][k])
    for kind in res:
        log(f"\n== {kind}: attribution in-support, mean +- std over {label} [full test]")
        for name, R in res[kind].items():
            a, p = ms(R, "acc"), ms(R, "p_own")
            log(f"   {name:28s} acc {a[0]:.3f} +- {a[1]:.3f} [{a[2]:.3f}]   P(own) {p[0]:.3f} +- {p[1]:.3f} [{p[2]:.3f}]")
        log(f"   per-subset difference vs {args.ref} (negative = arm lower):")
        for name in list(arms) + ["real summaries"]:
            if name == args.ref:
                continue
            for k in ("acc", "p_own"):
                d = np.array([s[k] for s in res[kind][name]["subsets"]]) - np.array([s[k] for s in res[kind][args.ref]["subsets"]])
                log(f"     {name:28s} {k:6s} {d.mean():+.4f} +- {d.std(ddof=1):.4f}")
        log(f"   per client (acc mean +- std):")
        for c in clients:
            log(f"     {c:16s} " + "  ".join(
                f"{name[:14]}: {np.mean([s['per_client'][c]['acc'] for s in R['subsets']]):.3f}+-{np.std([s['per_client'][c]['acc'] for s in R['subsets']], ddof=1):.3f}"
                for name, R in res[kind].items()))
    if args.json:
        json.dump({"label": label, "ref": args.ref, "files": {k: v[0] for k, v in arms.items()}, "report": report,
                   "results": res}, open(args.json, "w"), indent=1)
        log(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
