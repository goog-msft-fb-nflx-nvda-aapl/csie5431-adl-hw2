#!/usr/bin/env python3
"""Fine-tune Qwen3-Reranker-0.6B (yes/no logit score) with listwise softmax CE on train dialogues.
Candidates from cache/cands_train.npz top-N, labels from cache/judge_train.npy (judge accepts /
rejects, gold = 2). Positive = gold (or, with --alt-prob, a judge-accepted alternative); negatives =
judge-rejected candidates. Holds out the same product-disjoint dev split as ft_biencoder (seed 0)."""
import argparse
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "provided"))
from eval_utils import load_eval_queries, load_products  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3-Reranker-0.6B")
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--group", type=int, default=8)  # 1 pos + group-1 negs
ap.add_argument("--qbs", type=int, default=4)    # queries per step
ap.add_argument("--doc-chars", type=int, default=1800)
ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--alt-prob", type=float, default=0.0)
ap.add_argument("--use-all-train", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

INSTR = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
          "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

root = os.path.join(os.path.dirname(__file__), "..", "data")
qs = load_eval_queries(f"{root}/train/dialogues.jsonl")
prods = load_products(f"{root}/train/products.jsonl")
c = dict(np.load("cache/cands_train.npz"))
pids = c["pids"]
J = np.load("cache/judge_train.npy")
N = J.shape[1]
top = c["top"][:, :N]
rng = random.Random(a.seed)
targets = sorted({q["gold_pid"] for q in qs})
rng.shuffle(targets)
dev_t = set(targets[: int(len(targets) * 0.1)])
idx = [i for i, q in enumerate(qs) if a.use_all_train or q["gold_pid"] not in dev_t]
print("train queries", len(idx), flush=True)

tok = AutoTokenizer.from_pretrained(a.base, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable()
model.config.use_cache = False
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
steps_total = a.epochs * (len(idx) // a.qbs)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / (0.1 * steps_total)) * max(0.0, 1 - s / steps_total))


def fmt(q, p):
    return PREFIX + f"<Instruct>: {INSTR}\n<Query>: {q}\n<Document>: {prods[p][: a.doc_chars]}" + SUFFIX


def make_group(i):
    g = str(pids[c["gold"][i]])
    rej = [str(pids[top[i, r]]) for r in range(N) if J[i, r] == 0]
    acc = [str(pids[top[i, r]]) for r in range(N) if J[i, r] == 1]
    pos = rng.choice(acc) if (acc and rng.random() < a.alt_prob) else g
    if len(rej) < a.group - 1:
        return None
    return [pos] + rng.sample(rej, a.group - 1)


model.train()
step = 0
for ep in range(a.epochs):
    order = idx[:]
    rng.shuffle(order)
    for s in range(0, len(order) - a.qbs + 1, a.qbs):
        groups = [(i, make_group(i)) for i in order[s: s + a.qbs]]
        groups = [(i, g) for i, g in groups if g]
        if not groups:
            continue
        texts = [fmt(qs[i]["query"], p) for i, g in groups for p in g]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to("cuda")
        h = model.model(**enc).last_hidden_state[:, -1, :]
        lg = (h @ model.lm_head.weight[[yes, no]].T).float()
        sc = (lg[:, 0] - lg[:, 1]).view(len(groups), a.group)
        loss = torch.nn.functional.cross_entropy(sc, torch.zeros(len(groups), dtype=torch.long, device="cuda"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0:
            print(f"ep {ep} step {step}/{steps_total} loss {loss.item():.4f}", flush=True)
model.config.use_cache = True
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
print("saved", a.out, flush=True)
