#!/usr/bin/env python3
"""Cache BM25 + dense (zero-shot Qwen3-0.6B) scores and top-N RRF hybrid candidates per split.
Saves cache/cands_{split}.npz with pids, qids, gold idx, top-N candidate indices, rrf scores."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from zs_eval import bm25_scores, dense_scores, rrf  # noqa: E402
from zs_eval import load_eval_queries, load_products  # noqa: E402
import vram  # noqa: E402

vram.apply_cap()

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="public")
ap.add_argument("--model", default="qwen3-0.6b")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--tag", default="")
a = ap.parse_args()
root = os.environ.get("HW2_DATA", "data")
qs = load_eval_queries(f"{root}/{a.split}/dialogues.jsonl")
prods = load_products(f"{root}/{a.split}/products.jsonl")
pids = list(prods)
texts = [prods[p] for p in pids]
queries = [q["query"] for q in qs]
os.makedirs("cache", exist_ok=True)
sb = bm25_scores(queries, pids, texts)
sd = dense_scores(queries, texts, a.model, 512, f"cache/{a.split}_{a.model.replace(chr(47), chr(95))}_L512.npy")
S = rrf([sb, sd])
top = np.argsort(-S, axis=1)[:, : a.n]
pid2i = {p: i for i, p in enumerate(pids)}
gold = np.array([pid2i[q["gold_pid"]] for q in qs])
np.savez(f"cache/cands_{a.split}{a.tag}.npz", pids=np.array(pids), qids=np.array([q["qid"] for q in qs]),
         gold=gold, top=top, rrf=np.take_along_axis(S, top, 1),
         dense_top=np.argsort(-sd, axis=1)[:, : a.n], bm25_top=np.argsort(-sb, axis=1)[:, : a.n])
vram.report("cands")
hit = (top == gold[:, None])
print(a.split, "hybrid recall@{10,30,50,100,200}:",
      [round(float(hit[:, :k].any(1).mean()), 3) for k in (10, 30, 50, 100, 200)], flush=True)
