"""Does B1's k-shot prompting COPY its in-context decoration sets? (CAL-26)

    python scripts/b1_copy_report.py --run runs/<fed run> --eval <b1 k=3 eval json> \
        --data data/chembl_deco_skew_pruned/data.jsonl [--shots 3]

Uses clients[c]["outputs"] (EVERY prompt x alpha generation) with clients[c]["record_ids"] for the
prompt order, not the "samples" block -- `fedsteer.evaluate.assemble` puts only the FIRST prompt's
outputs in `samples`, so a report built on it would cover 5 generations per client instead of all
750. record_ids also give each prompt's own record, which is what `pick_shots` excludes, so the
shot reconstruction matches the evaluation exactly rather than approximately.

The shot pool is rebuilt with `build_clients` using the RUN'S OWN config -- same `max_train`, seed,
`tie_break` and `alpha_mode` as `eval_baselines.py` used. The raw jsonl records cannot be used
directly: `pick_shots` ranks by `r["alpha"]`, which does not exist on disk and is assigned by
`build_clients` from the alpha protocol. Reading the jsonl straight gives KeyError 'alpha'.

MOL-26a's smoke showed k = 3 scoring WORSE than k = 0 on the client model (21% vs 0% unscorable)
even though both produce in-format output. Two explanations with different implications:

  copying           the model reproduces a shot's decoration set verbatim, so it is retrieving
                    rather than decorating, and its error is whatever that shot's level happened
                    to be;
  distribution shift the shots pull the model off the distribution it was fine-tuned on, so its
                    output is novel but worse.

This counts the first so the second can be inferred. Shots are not stored in the eval json, so they
are reproduced with `fedsteer.baselines.pick_shots` on the same training pool, alpha and exclusion
rule the evaluation used -- the function is deterministic, so the reconstruction is exact as long as
--shots matches the run.

Reported per client: the share of generated outputs equal to one of that prompt's own shot targets
(exact, and canonicalised by sorting the '.'-joined fragments so attachment-point order does not
hide a match), plus the share equal to ANY training target of that client, which catches retrieval
from the wider pool rather than from the shots.

"= any train target" needs two companions to be interpretable, so they are printed too:
  distinct   how many distinct outputs the model produced, over how many generations. Mode collapse
             onto a handful of generic decoration sets looks like retrieval on the "any" column
             without being retrieval at all.
  in top10   the share of outputs that are among the client's ten most common training targets.
             The target space is NOT degenerate (292-1691 distinct targets per client, top-1 share
             only 3-12%), so a high "any" share WITH a high "in top10" share and few distinct
             outputs means collapse onto common simple decorations; a high "any" with many distinct
             outputs would be genuine retrieval.
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from fedsteer.baselines import pick_shots
from fedsteer.data import build_clients, read_jsonl


def canon(s: str) -> str:
    """Decoration sets are '.'-joined [n*] fragments; sort them so order is not a false difference."""
    return ".".join(sorted(p.strip() for p in (s or "").strip().split(".") if p.strip()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", required=True, help="the k>0 B1 eval json")
    ap.add_argument("--data", required=True)
    ap.add_argument("--shots", type=int, default=3)
    args = ap.parse_args()

    o = json.load(open(args.eval))
    alphas = [float(a) for a in o["alphas"]]
    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    recs = read_jsonl(args.data)
    by_id = {f"{r['client']}/{r.get('molecule_chembl_id')}": r for r in recs}
    # the pool pick_shots saw: same cap, seed, tie_break and alpha protocol as eval_baselines.py
    train, _ = build_clients(recs, cfg["clients"], tie_break=cfg.get("tie_break", "average"),
                             max_train=cfg.get("max_train_per_client"), seed=cfg["fed"]["seed"],
                             alpha_mode=cfg.get("alpha_mode", "local"))
    all_targets = {c: {canon(r["target"]) for r in rs} for c, rs in train.items()}

    top10 = {c: {t for t, _ in Counter(canon(r["target"]) for r in rs).most_common(10)}
             for c, rs in train.items()}

    print(f"== B1 verbatim copying  ({os.path.basename(args.eval)}, k={args.shots})")
    print(f"{'client':12s} {'n':>5s} {'=a shot':>8s} {'=any':>7s} {'in top10':>9s} "
          f"{'distinct':>9s} {'unscorable':>11s}")
    tot = {"n": 0, "shot": 0, "any": 0, "t10": 0}
    for c in sorted(o["clients"]):
        m = o["clients"][c]
        outs, ids = m.get("outputs"), m.get("record_ids") or []
        if not outs:
            print(f"{c:12s} {'-':>5s}   (no stored outputs)")
            continue
        pool = list(train.get(c, []))
        n = shot_hits = any_hits = t10_hits = 0
        seen = set()
        for i, row in enumerate(outs):                     # outputs[prompt][alpha]
            rec = by_id.get(ids[i]) if i < len(ids) else None
            for j, text in enumerate(row):
                out = canon(text)
                if not out:
                    continue
                n += 1
                seen.add(out)
                shots = pick_shots(pool, alphas[j] if j < len(alphas) else 0.0,
                                   args.shots, (rec or {}).get("url"))
                if out in {canon(t["target"]) for t in shots}:
                    shot_hits += 1
                if out in all_targets.get(c, set()):
                    any_hits += 1
                if out in top10.get(c, set()):
                    t10_hits += 1
        u = m.get("unscorable_row_rate")
        tot["n"] += n; tot["shot"] += shot_hits; tot["any"] += any_hits; tot["t10"] += t10_hits
        print(f"{c:12s} {n:5d} {shot_hits/max(n,1):8.3f} {any_hits/max(n,1):7.3f} "
              f"{t10_hits/max(n,1):9.3f} {f'{len(seen)}/{n}':>9s} "
              f"{(u if u is not None else float('nan')):11.3f}")
    print(f"{'MEAN':12s} {tot['n']:5d} {tot['shot']/max(tot['n'],1):8.3f} "
          f"{tot['any']/max(tot['n'],1):7.3f} {tot['t10']/max(tot['n'],1):9.3f}")
    print("\nReading: a high '=a shot' share means k-shot B1 is retrieving its own examples, so its\n"
          "error reflects the shots' levels rather than any calibration. A low share with a raised\n"
          "unscorable rate means the shots shifted the model off its fine-tuned distribution instead.")


if __name__ == "__main__":
    main()
