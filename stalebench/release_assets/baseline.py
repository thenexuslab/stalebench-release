"""Reference baselines over the released files --- nothing else needed.

Reproduces the paper's headline store-all rows from the *public* statement and
query files: no fact keys, no versions, and no currency labels are read, so
whatever this scores is achievable by any submission.

    pip install sentence-transformers
    python baseline.py --config anchored --split test
    python evaluate.py --key key_anchored_test.jsonl \
                       --submission sub_anchored_test_cosine.jsonl

The paper's true-key oracle is deliberately absent: it needs the answer key, so
it is an upper bound rather than a baseline anyone could submit.
"""

import argparse
import json

import numpy as np


def load(path):
    return [json.loads(line) for line in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="anchored")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=2, help="time-rerank depth")
    ap.add_argument("--topn", type=int, default=20)
    a = ap.parse_args()

    st = load("statements_%s_%s.jsonl" % (a.config, a.split))
    qs = load("queries_%s_%s.jsonl" % (a.config, a.split))
    st.sort(key=lambda s: s["t"])                    # consume in time order

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model)
    emb = m.encode([s["text"] for s in st], batch_size=128,
                   convert_to_numpy=True, normalize_embeddings=True)
    qemb = m.encode([q["question"] for q in qs], batch_size=128,
                    convert_to_numpy=True, normalize_embeddings=True)
    ids = [s["id"] for s in st]
    ts = np.array([s["t"] for s in st], dtype=float)

    subs = {"cosine": [], "time_rerank": []}
    for qi in range(len(qs)):
        order = list(np.argsort(-(emb @ qemb[qi]), kind="stable"))
        subs["cosine"].append([ids[j] for j in order[:a.topn]])
        top = sorted(order[:a.k], key=lambda j: -ts[j])
        subs["time_rerank"].append([ids[j] for j in (top + order[a.k:])[:a.topn]])

    for name, rankings in subs.items():
        path = "sub_%s_%s_%s.jsonl" % (a.config, a.split, name)
        with open(path, "w") as fh:
            for q, r in zip(qs, rankings):
                fh.write(json.dumps({"query_id": q["query_id"],
                                     "ranking": r}) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
