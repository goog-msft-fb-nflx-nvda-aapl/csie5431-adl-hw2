#!/usr/bin/env python3
"""Run the official judge (evaluate.LLMJudge, gold as reference) over the top-N cached candidates of
a split. Saves cache/judge_{split}.npy: int8 [nq, N], 1 = judge accepts, 0 = rejects, 2 = is gold."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "provided"))
import evaluate as ev  # noqa: E402
from eval_utils import build_eval_query, load_eval_queries, load_products  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--cands", default="")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), "..", "data")
qs = load_eval_queries(f"{root}/{a.split}/dialogues.jsonl")
prods = load_products(f"{root}/{a.split}/products.jsonl")
c = dict(np.load(f"cache/cands_{a.split}{a.cands}.npz"))
pids, top, gold = c["pids"], c["top"][:, : a.n], c["gold"]
nq = a.limit or len(qs)
items, where = [], []
for i in range(nq):
    g = str(pids[gold[i]])
    for r in range(a.n):
        p = str(pids[top[i, r]])
        if p == g:
            continue
        items.append(ev.JudgeItem(qs[i]["qid"], r + 1, g, p, qs[i]["query"], prods[g], prods[p]))
        where.append((i, r))
order = np.argsort([len(it.candidate_product) + len(it.gold_product) for it in items])
judge = ev.LLMJudge(ev.DEFAULT_MODEL, "cuda")
out = np.zeros((nq, a.n), np.int8)
for i in range(nq):
    out[i][top[i] == gold[i]] = 2
bad = 0
for s in range(0, len(order), a.bs):
    batch = [items[j] for j in order[s: s + a.bs]]
    for j, res in zip(order[s: s + a.bs], judge.judge_batch(batch)):
        i, r = where[j]
        out[i, r] = 1 if res["relevant"] else 0
        bad += 0 if res["parse_ok"] else 1
    if (s // a.bs) % 100 == 0:
        print(f"{s}/{len(items)} parse_fail={bad}", flush=True)
np.save(f"cache/judge_{a.split}{a.cands}.npy", out)
non_gold = out != 2
print("accept rate among non-gold:", float((out == 1).sum() / non_gold.sum()),
      "queries with >=1 accepted alt in top-n:", float(((out == 1).any(1)).mean()), "parse_fail", bad)
