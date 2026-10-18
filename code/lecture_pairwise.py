#!/usr/bin/env python3
"""Pairwise reranking (lecture slides 28-30): Qwen2.5-1.5B-Instruct compares the current top-K products two at a time
(both orders, to cancel position bias); a product's score is its mean win probability. Combined with the fine-tuned
reranker's scores by z-score sum."""
import argparse
import itertools

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lecture_common as lc

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_lite")
ap.add_argument("--rr", default="lite_rr")
ap.add_argument("--depth", type=int, default=30)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--doc-chars", type=int, default=800)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--out", default="results/lecture.jsonl")
a = ap.parse_args()

qs, prods, c, seen = lc.load_public(a.tag)
pids, gold = c["pids"], c["gold"]
R = np.load(f"cache/rr_public_{a.rr}.npy")[:, : a.depth]
top = lc.reorder(c["top"][:, : a.depth], R, a.depth)  # fine-tuned reranker order
K = a.k
SYS = ("You compare two candidate products for a shopper. Decide which product better satisfies the shopper's request in "
       "the conversation: first the product type and core function, then the firm constraints they state or confirm. "
       "Answer with a single letter, A or B.")
cache = f"cache/pw_public{a.tag}_k{K}.npy"
try:
    W = np.load(cache)
except FileNotFoundError:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    ida, idb = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    items = [(i, x, y) for i in range(len(qs)) for x, y in itertools.permutations(range(K), 2)]
    P = {}
    order = sorted(range(len(items)), key=lambda k: len(qs[items[k][0]]["query"]))
    for b in range(0, len(order), a.bs):
        batch = [items[k] for k in order[b: b + a.bs]]
        texts = []
        for i, x, y in batch:
            da = prods[str(pids[top[i, x]])][: a.doc_chars]; db = prods[str(pids[top[i, y]])][: a.doc_chars]
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": f"[CONVERSATION]\n{qs[i]['query']}\n\n[PRODUCT A]\n{da}\n\n[PRODUCT B]\n{db}\n\nWhich product is better, A or B?"}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            lg = m(**enc).logits[:, -1, :].float()
        pa = torch.sigmoid(lg[:, ida] - lg[:, idb]).cpu().numpy()
        for (i, x, y), p in zip(batch, pa):
            P[(i, x, y)] = float(p)
    W = np.zeros((len(qs), K), np.float32)  # mean probability that candidate x beats the others
    for i in range(len(qs)):
        for x in range(K):
            ws = [0.5 * (P[(i, x, y)] + 1 - P[(i, y, x)]) for y in range(K) if y != x]
            W[i, x] = np.mean(ws)
    np.save(cache, W)
rows = {"reranker_only": lc.metrics(top, gold, seen)["all"]}
Rk = np.take_along_axis(R, np.argsort(-R, 1, kind="stable"), 1)[:, :K]
for alpha in (0.0, 0.25, 0.5, 1.0):
    comb = lc.zscore(Rk) + alpha * lc.zscore(W) if alpha else Rk
    if alpha == 0.0:
        pass
    sc = np.concatenate([comb, np.full((len(qs), a.depth - K), -1e9)], 1)
    r = lc.metrics(lc.reorder(top, sc, a.depth), gold, seen)
    rows[f"rr+{alpha}*pairwise"] = r["all"]
pair_only = np.concatenate([W, np.full((len(qs), a.depth - K), -1e9)], 1)
rows["pairwise_only_top%d" % K] = lc.metrics(lc.reorder(top, pair_only, a.depth), gold, seen)["all"]
lc.log_row(a.out, {"exp": "pairwise_rerank", "k": K, "rows": rows})
for alpha in (0.5, 1.0):
    comb = lc.zscore(Rk) + alpha * lc.zscore(W)
    sc = np.concatenate([comb, np.full((len(qs), a.depth - K), -1e9)], 1)
    lc.emit(lc.reorder(top, sc, a.depth), c, f"results/rr_pw_a{alpha}_k{K}.json", qs)
lc.emit(lc.reorder(top, pair_only, a.depth), c, f"results/rr_pw_only_k{K}.json", qs)
