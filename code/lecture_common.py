"""Shared helpers for the lecture-method ablations (query likelihood, HyDE / rewriting, pairwise reranking,
reranker-to-retriever distillation). All operate on cached shortlists cache/cands_public{tag}.npz."""
import json
import os
import sys

import numpy as np

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "provided"]
from eval_utils import load_eval_queries, load_products  # noqa: E402


def load_public(tag="_lite"):
    root = os.environ.get("HW2_DATA", "data")
    qs = load_eval_queries(f"{root}/public/dialogues.jsonl")
    prods = load_products(f"{root}/public/products.jsonl")
    train = set(load_products(f"{root}/train/products.jsonl"))
    c = dict(np.load(f"cache/cands_public{tag}.npz"))
    seen = np.array([q["gold_pid"] in train for q in qs])
    return qs, prods, c, seen


def metrics(top, gold, seen, k=10):
    hit = top[:, :k] == gold[:, None]
    ranks = np.where(hit.any(1), hit.argmax(1) + 1, 0)
    out = {}
    for nm, m in (("all", np.ones(len(gold), bool)), ("seen", seen), ("unseen", ~seen)):
        r = ranks[m]
        out[nm] = {"R@10": round(float((r > 0).mean()), 4),
                   "MRR@10": round(float(np.where(r > 0, 1.0 / np.maximum(r, 1), 0).mean()), 4)}
    return out


def reorder(top, S, d):
    """Re-sort the first d candidates of each query by score S[:, :d] (descending)."""
    order = np.argsort(-S[:, :d], axis=1, kind="stable")
    return np.concatenate([np.take_along_axis(top[:, :d], order, 1), top[:, d:]], 1)


def zscore(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)


def emit(top, c, path, qs):
    res = [{"qid": q["qid"], "ranked_pids": [str(c["pids"][j]) for j in top[i, :10]]} for i, q in enumerate(qs)]
    json.dump({"top_k": 10, "results": res}, open(path, "w"))


def log_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
