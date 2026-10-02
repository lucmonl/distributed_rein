"""LLM-as-judge quality evaluation on a subsample of saved outputs.

    python scripts/judge_quality.py --run runs/X --eval runs/X/evals/eval_round_0060__<stamp>.json

Judge: Qwen2.5-7B-Instruct (local, greedy).  Per summary it returns
  * faithfulness: FineSurE-style (Song et al., ACL 2024) sentence-level fact check,
    reported as the fraction of summary sentences fully supported by the article;
  * relevance and coherence: 1-5 ratings in the style of G-Eval / SummEval.
The judge is never told the target alpha or the method.  The same judge scores real
reference summaries of the same split, binned by global alpha, as a baseline.

Writes <run>/evals/judge_<eval file name>.json.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedsteer.data import ClientQuantiles, alpha_reference_from_json, read_jsonl  # noqa: E402
from fedsteer.quality import split_sentences  # noqa: E402

SYSTEM = "You are a careful news editor who evaluates article summaries."
TEMPLATE = """Article:
{article}

Summary (one sentence per line):
{sentences}

Evaluate the summary against the article.
1. For each summary sentence, answer "yes" if every claim in it is supported by the article, otherwise "no".
2. Relevance (1-5): how well the summary captures the most important information of the article.
3. Coherence (1-5): how clear, well-formed and non-repetitive the summary is.

Reply with JSON only, in this format:
{{"faithful": ["yes", "no", ...], "relevance": <1-5>, "coherence": <1-5>}}"""


def parse(text: str, n_sent: int):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        f = [str(x).strip().lower().startswith("y") for x in d["faithful"]][:n_sent]
        return {"faithful": sum(f) / len(f) if f else None,
                "relevance": float(d["relevance"]), "coherence": float(d["coherence"])}
    except Exception:  # noqa: BLE001
        return None


@torch.no_grad()
def judge(model, tok, items, batch_size=16, max_new_tokens=160):
    prompts = []
    for art, summ in items:
        sents = split_sentences(summ) or [summ]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TEMPLATE.format(article=art, sentences="\n".join(sents))}]
        prompts.append((tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), len(sents)))
    out = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok([p for p, _ in batch], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out.extend(parse(t, n) for t, (_, n) in zip(texts, batch))
        print(f"judged {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out


def agg(rows):
    keys = ("faithful", "relevance", "coherence")
    ok = [r for r in rows if r is not None]
    return {**{k: (float(np.mean([r[k] for r in ok if r[k] is not None])) if ok else None) for k in keys},
            "n": len(rows), "parse_fail": len(rows) - len(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", nargs="+", required=True, help="one or more eval files of the run")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--per_client", type=int, default=25, help="articles per client (all alphas each)")
    ap.add_argument("--refs_per_bin", type=int, default=30, help="reference summaries per alpha bin")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    local_q = {c: ClientQuantiles(v) for c, v in json.load(open(os.path.join(args.run, "client_quantiles.json"))).items()}
    ref_path = os.path.join(args.run, "alpha_reference.json")
    gref = alpha_reference_from_json(json.load(open(ref_path))) if os.path.exists(ref_path) else None
    records = read_jsonl(cfg["data_path"])
    by_url = {r["url"]: r for r in records}   # all clients: E2 evaluates held-out (non-participant) clients

    tok = AutoTokenizer.from_pretrained(args.judge)
    model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    ref_cache = {}                            # (split, alphas) -> reference scores by alpha bin
    for eval_path in args.eval:
        judge_file(eval_path, args, cfg, local_q, gref, records, by_url, tok, model, ref_cache)


def judge_file(eval_path, args, cfg, local_q, gref, records, by_url, tok, model, ref_cache):
    ev = json.load(open(eval_path))
    alphas, split = ev["alphas"], ev["split"]
    items, keys = [], []
    for c, res in ev["clients"].items():
        for i in range(min(args.per_client, len(res["outputs"]))):
            art = by_url[res["record_ids"][i]]["article"]
            for j in range(len(alphas)):
                items.append((art, res["outputs"][i][j]))
                keys.append(("gen", c, j))
    # reference baseline: participants' real summaries of the split, by alpha bin (scored once per split)
    ck = (split, tuple(alphas))
    if ck not in ref_cache:
        rng = random.Random(args.seed)
        bins = defaultdict(list)
        for r in records:
            if r.get("split") == split and r["client"] in cfg["clients"]:
                q = gref if gref is not None else local_q[r["client"]]
                bins[int(np.argmin([abs(q.cdf(r["score"]) - a) for a in alphas]))].append(r)
        for j, rs in bins.items():
            for r in rng.sample(rs, min(args.refs_per_bin, len(rs))):
                items.append((r["article"], r["target"]))
                keys.append(("ref", None, j))

    scores = judge(model, tok, items, args.batch_size)
    gen = defaultdict(lambda: defaultdict(list))
    ref = defaultdict(list)
    for (kind, c, j), s in zip(keys, scores):
        (gen[c][j] if kind == "gen" else ref[j]).append(s)
    if ck in ref_cache:
        ref = ref_cache[ck]
    else:
        ref_cache[ck] = ref
    out = {"eval": eval_path, "judge": args.judge, "alphas": alphas,
           "reference_by_alpha": {j: agg(ref[j]) for j in sorted(ref)}, "clients": {}}
    for c, res in ev["clients"].items():
        lo, hi = res.get("support", [0.0, 1.0])
        per_alpha = [agg(gen[c][j]) for j in range(len(alphas))]
        summ = {}
        for tag, js in (("in", [j for j, a in enumerate(alphas) if lo <= a <= hi]),
                        ("out", [j for j, a in enumerate(alphas) if not lo <= a <= hi])):
            for k in ("faithful", "relevance", "coherence"):
                vals = [per_alpha[j][k] for j in js if per_alpha[j][k] is not None]
                summ[f"{k}_{tag}"] = float(np.mean(vals)) if vals else None
                gaps = [per_alpha[j][k] - out["reference_by_alpha"][j][k] for j in js
                        if j in out["reference_by_alpha"] and per_alpha[j][k] is not None
                        and out["reference_by_alpha"][j][k] is not None]
                summ[f"gap_{k}_{tag}"] = float(np.mean(gaps)) if gaps else None
        out["clients"][c] = {"support": [lo, hi], "per_alpha": per_alpha, "summary": summ}
        print(c, json.dumps(summ), flush=True)
    path = os.path.join(os.path.dirname(eval_path), "judge_" + os.path.basename(eval_path))
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
