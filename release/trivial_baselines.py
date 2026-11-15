"""Reference *trivial* baselines: what a rule with no model scores.

A benchmark should publish the score of the cheapest thing that could possibly
work, so a submission knows what it has to beat. Two rules, both reading only
the public statement and query files --- never the key file:

``verbatim``  recover the subject from the question with the released question
              templates, keep every statement containing that string, answer
              most-recent first. This near-saturates the *anchored*
              configurations, which is the anchor axis doing its job rather than
              a defect: templated text that repeats the subject is not a hard
              same-fact problem.
``trigram``   the same, but link statements to the recovered subject inventory
              by character-trigram Jaccard, so short forms resolve too. This is
              the read-path analogue of the paper's fuzzy keying rung.

Report contamination@5 and slots alongside current@1: both rules keep the whole
store and are maximally contaminated, so current@1 alone flatters them.

    python trivial_baselines.py --config anchor_free --split test
    python evaluate.py --key key_anchor_free_test.jsonl \
                       --submission sub_anchor_free_test_trivial_trigram.jsonl
"""

import argparse
import collections
import json
import re

_PATTERNS = [
    r"Which team does (.+?) currently play for\?", r"What is (.+?)'s current club\?",
    r"Who does (.+?) play for now\?", r"What position does (.+?) currently hold\?",
    r"What office does (.+?) hold now\?", r"What is (.+?)'s current position\?",
    r"Who does (.+?) currently work for\?", r"What is (.+?)'s current employer\?",
    r"Where does (.+?) work now\?", r"Which party is (.+?) currently a member of\?",
    r"What is (.+?)'s current political party\?",
    r"Which party does (.+?) belong to now\?",
    r"Who is currently the head coach of (.+?)\?", r"Who coaches (.+?) now\?",
    r"What is the name of (.+?)'s current head coach\?",
    r"Who is currently the chair of (.+?)\?", r"Who chairs (.+?) now\?",
    r"Who currently leads the board of (.+?)\?",
    r"Who is currently the head of the government of (.+?)\?",
    r"Who leads (.+?)'s government now\?",
    r"Who currently heads the government of (.+?)\?", r"Who currently owns (.+?)\?",
    r"What is the current owner of (.+?)\?", r"Who is (.+?) owned by now\?",
    r"Which institution did (.+?) attend most recently\?",
    r"Where did (.+?) study most recently\?",
    r"What is the latest school (.+?) attended\?",
]


def subject_of(question):
    for pattern in _PATTERNS:
        m = re.fullmatch(pattern, question)
        if m:
            return m.group(1)
    return None


def trigrams(text):
    padded = "  %s " % text.lower()
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def run(statements, queries, mode, topn=20):
    """Return one {query_id, ranking} record per query."""
    subjects = [subject_of(q["question"]) for q in queries]
    inventory = sorted({s for s in subjects if s})
    assign = collections.defaultdict(list)
    if mode == "verbatim":
        for s in statements:
            for subj in inventory:
                if subj in s["text"]:
                    assign[subj].append(s)
    else:
        grams = [trigrams(s) for s in inventory]
        index = collections.defaultdict(list)
        for i, g in enumerate(grams):
            for t in g:
                index[t].append(i)
        for s in statements:
            q = trigrams(s["text"])
            cand = collections.Counter()
            for t in sorted(q):                      # sorted: hash-seed safe
                for i in index.get(t, ()):
                    cand[i] += 1
            best, best_sim = None, 0.0
            for i in sorted(cand):
                sim = cand[i] / (len(q) + len(grams[i]) - cand[i])
                if sim > best_sim:
                    best, best_sim = i, sim
            if best is not None:
                assign[inventory[best]].append(s)
    out = []
    for q, subj in zip(queries, subjects):
        items = sorted(assign.get(subj, []), key=lambda s: -s["t"])
        out.append({"query_id": q["query_id"],
                    "ranking": [s["id"] for s in items[:topn]]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="anchor_free")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="both",
                    choices=["verbatim", "trigram", "both"])
    a = ap.parse_args()
    st = [json.loads(l) for l in
          open("statements_%s_%s.jsonl" % (a.config, a.split))]
    qs = [json.loads(l) for l in
          open("queries_%s_%s.jsonl" % (a.config, a.split))]
    st.sort(key=lambda s: s["t"])
    modes = ["verbatim", "trigram"] if a.mode == "both" else [a.mode]
    for mode in modes:
        path = "sub_%s_%s_trivial_%s.jsonl" % (a.config, a.split, mode)
        with open(path, "w") as fh:
            for r in run(st, qs, mode):
                fh.write(json.dumps(r) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
