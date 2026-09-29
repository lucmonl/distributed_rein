"""Toy federated steering data: the attribute is response length (words).

Every client writes short notes on the same kinds of topics, but clients have
different length ranges, e.g. client_0 writes 1-3 sentences and client_3 writes
4-10.  A good shared direction means "longer"; each client's gain and private
adapter must map alpha onto its own range.  Used for smoke tests only.
"""

import argparse
import json
import random

TOPICS = ["the ocean", "a busy city", "a mountain hike", "an old library", "a rainy morning", "a train journey",
          "a garden in spring", "a winter night", "a local market", "a quiet village", "a science museum",
          "a football match", "a coffee shop", "a desert road", "a school trip", "a harbor at dawn"]
SENTENCES = [
    "It was calmer than I expected.", "The light kept changing through the afternoon.",
    "People moved slowly and nobody seemed in a hurry.", "There was a faint smell of salt and rain.",
    "Small details stood out, like the color of the doors.", "I stayed longer than I had planned.",
    "The sounds faded as the evening came.", "Everything felt a little older than it looked.",
    "A few children were laughing somewhere nearby.", "The path curved and disappeared behind the trees.",
    "I kept thinking about how quiet it was.", "It is the kind of place you remember later.",
    "The air was cool and clear.", "Someone had left a bicycle by the wall.",
    "I wrote a few lines in my notebook.", "The whole scene looked like an old photograph.",
]
# (min_sentences, max_sentences) per client: overlapping but shifted ranges
RANGES = [(1, 3), (2, 5), (3, 7), (4, 10), (1, 5), (2, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with open(args.out, "w") as f:
        for c in range(args.clients):
            lo, hi = RANGES[c % len(RANGES)]
            for split, n in (("train", args.train), ("test", args.test)):
                for _ in range(n):
                    topic = rng.choice(TOPICS)
                    k = rng.randint(lo, hi)
                    target = f"A note about {topic}. " + " ".join(rng.sample(SENTENCES, k))
                    rec = {"client": f"client_{c}", "split": split,
                           "prompt": f"Write a short note about {topic}.",
                           "target": target, "score": len(target.split())}
                    f.write(json.dumps(rec) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
