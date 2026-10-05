"""Backup variant of the math-CoT data (user, 10-04): the gold answer is stated in the prompt, so the
task is "write a solution of the requested length that reaches this answer". Accuracy then stops
being a confound and the experiment is purely about the length of the CoT.

    python scripts/make_answer_prompt_data.py --src data/math_fed --dst data/math_fed_ans

Same records, splits, targets and scores as --src (rewritten prompts only), so results on the two
variants are paired problem by problem. clients.json is copied with a note.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_math_fed import INSTRUCTION  # noqa: E402

ANS_INSTRUCTION = ("The final answer is ${answer}$. Please reason step by step to reach it, "
                   "and put your final answer within \\boxed{{}}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/math_fed")
    ap.add_argument("--dst", default="data/math_fed_ans")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    n = 0
    with open(os.path.join(args.src, "data.jsonl")) as f, open(os.path.join(args.dst, "data.jsonl"), "w") as g:
        for line in f:
            r = json.loads(line)
            assert r["prompt"].endswith(INSTRUCTION), r["url"]
            r["prompt"] = r["prompt"][: -len(INSTRUCTION)] + ANS_INSTRUCTION.format(answer=r["answer"])
            g.write(json.dumps(r) + "\n")
            n += 1
    meta = json.load(open(os.path.join(args.src, "clients.json")))
    meta["instruction"] = ANS_INSTRUCTION
    meta["variant"] = "answer_in_prompt"
    json.dump(meta, open(os.path.join(args.dst, "clients.json"), "w"), indent=1)
    print(f"wrote {n:,} records to {args.dst}/data.jsonl")


if __name__ == "__main__":
    main()
