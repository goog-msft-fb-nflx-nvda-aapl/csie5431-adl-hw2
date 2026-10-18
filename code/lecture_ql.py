#!/usr/bin/env python3
"""Query-likelihood reranking (lecture slides 19-21, UPR-style): score each shortlisted product by the mean log
P(conversation text | product, instruction) under Qwen2.5-1.5B-Instruct, then compare / combine with the fine-tuned
reranker scores (cache/rr_public_{--rr}.npy, produced by rerank_eval.py on the same shortlist)."""
import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lecture_common as lc

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_lite")
ap.add_argument("--rr", default="lite_rr")
ap.add_argument("--depth", type=int, default=30)
ap.add_argument("--doc-chars", type=int, default=1400)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--out", default="results/lecture.jsonl")
a = ap.parse_args()

qs, prods, c, seen = lc.load_public(a.tag)
pids, top, gold = c["pids"], c["top"][:, : a.depth], c["gold"]
cache = f"cache/ql_public{a.tag}_d{a.depth}.npy"
HEAD = "Product listing:\n{doc}\n\nThe conversation below is between a shopper and a sales assistant, and it ended with this product being the right recommendation.\n\n"
if os.path.exists(cache):
    S = np.load(cache)
else:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    q_ids = [tok.encode(q["query"], add_special_tokens=False) for q in qs]
    items = [(i, j) for i in range(len(qs)) for j in range(a.depth)]
    p_ids = {}
    S = np.zeros((len(qs), a.depth), np.float32)
    order = sorted(range(len(items)), key=lambda k: len(prods[str(pids[top[items[k][0], items[k][1]]])][: a.doc_chars]))
    for b in range(0, len(order), a.bs):
        batch = [items[k] for k in order[b: b + a.bs]]
        seqs, spans = [], []
        for i, j in batch:
            doc = prods[str(pids[top[i, j]])][: a.doc_chars]
            pi = tok.encode(HEAD.format(doc=doc), add_special_tokens=False)
            seqs.append(pi + q_ids[i]); spans.append((len(pi), len(q_ids[i])))
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, : len(s)] = torch.tensor(s); att[r, : len(s)] = 1
        ids, att = ids.cuda(), att.cuda()
        with torch.inference_mode():
            h = m.model(input_ids=ids, attention_mask=att).last_hidden_state
            for r, (lp, lq) in enumerate(spans):
                logits = m.lm_head(h[r, lp - 1: lp + lq - 1]).float()
                lps = torch.log_softmax(logits, -1).gather(1, ids[r, lp: lp + lq].unsqueeze(1)).squeeze(1)
                i, j = batch[r]
                S[i, j] = lps.mean().item()
    np.save(cache, S)
print("ql scores", S.shape, flush=True)

R = np.load(f"cache/rr_public_{a.rr}.npy")[:, : a.depth]
rows = {"first": lc.metrics(top, gold, seen)["all"]}
for name, sc in (("rr", R), ("ql", S)):
    rows[name] = lc.metrics(lc.reorder(top, sc, a.depth), gold, seen)["all"]
best = None
for w in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
    comb = lc.zscore(R) + w * lc.zscore(S)
    r = lc.metrics(lc.reorder(top, comb, a.depth), gold, seen)
    rows[f"rr+{w}*ql"] = r["all"]
    if best is None or r["all"]["MRR@10"] > best[1]:
        best = (w, r["all"]["MRR@10"])
# cheaper variants: query likelihood only on the top-K candidates of the reranker's order (others keep that order)
idx = np.argsort(-R, axis=1, kind="stable")
Rs, Ss, tops = (np.take_along_axis(x, idx, 1) for x in (R, S, top))
for K in (5, 10, 15, 20):
    for w in (1.0, 2.0):
        comb = np.full_like(Rs, -1e9)
        comb[:, :K] = lc.zscore(Rs[:, :K]) + w * lc.zscore(Ss[:, :K])
        comb[:, K:] = -1e8 - np.arange(a.depth - K)[None, :]
        rows[f"ql_top{K}_w{w}"] = lc.metrics(lc.reorder(tops, comb, a.depth), gold, seen)["all"]
lc.log_row(a.out, {"exp": "query_likelihood", "depth": a.depth, "rows": rows, "best_w": best[0]})
comb = lc.zscore(R) + best[0] * lc.zscore(S)
lc.emit(lc.reorder(top, comb, a.depth), c, f"results/rr_ql_comb_d{a.depth}.json", qs)
lc.emit(lc.reorder(top, S, a.depth), c, f"results/rr_ql_alone_d{a.depth}.json", qs)
