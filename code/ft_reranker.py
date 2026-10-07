#!/usr/bin/env python3
"""Fine-tune Qwen3-Reranker-0.6B (yes/no logit score) with listwise softmax CE on train dialogues.
Candidates: cache/cands_train.npz top-N; labels: external/judge_labels_train.json ({qid: {pid: 1 accepted /
0 rejected}}, from judge_labels.py). Positive = gold or (with prob --alt-prob) a judge-accepted alternative;
negatives = judge-rejected candidates. One dialogue per micro-step (gradient checkpointing keeps memory small),
--qbs dialogues per optimizer step. Holds out the same product-disjoint dev split as ft_biencoder (seed 0)."""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "provided"]
from eval_utils import load_eval_queries, load_products  # noqa: E402
import vram  # noqa: E402

vram.apply_cap()

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3-Reranker-0.6B")
ap.add_argument("--out", required=True)
ap.add_argument("--labels", default="external/judge_labels_train.json")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--group", type=int, default=16)  # 1 pos + group-1 negs
ap.add_argument("--qbs", type=int, default=2)     # dialogues per optimizer step
ap.add_argument("--doc-chars", type=int, default=1800)
ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--alt-prob", type=float, default=0.5)
ap.add_argument("--n-cands", type=int, default=20)
ap.add_argument("--use-all-train", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

INSTR = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
          "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

root = os.environ.get("HW2_DATA", "data")
qs = load_eval_queries(f"{root}/train/dialogues.jsonl")
prods = load_products(f"{root}/train/products.jsonl")
c = dict(np.load("cache/cands_train.npz"))
pids = c["pids"]
labels = json.load(open(a.labels))
top = c["top"][:, : a.n_cands]
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
    lab = labels.get(qs[i]["qid"], {})
    cand = [str(pids[j]) for j in top[i]]
    rej = [p for p in cand if lab.get(p) == 0]
    acc = [p for p in cand if lab.get(p) == 1]
    pos = rng.choice(acc) if (acc and rng.random() < a.alt_prob) else g
    if len(rej) < a.group - 1:
        return None
    return [pos] + rng.sample(rej, a.group - 1)


def micro_loss(i, group):
    txt = [fmt(qs[i]["query"], p) for p in group]
    enc = tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to("cuda")
    h = model.model(**enc).last_hidden_state[:, -1, :]
    lg = (h @ model.lm_head.weight[[yes, no]].T).float()
    sc = (lg[:, 0] - lg[:, 1]).unsqueeze(0)
    return torch.nn.functional.cross_entropy(sc, torch.zeros(1, dtype=torch.long, device="cuda"))


model.train()
step = 0
for ep in range(a.epochs):
    order = idx[:]
    rng.shuffle(order)
    for s in range(0, len(order) - a.qbs + 1, a.qbs):
        n_ok, last = 0, 0.0
        for i in order[s: s + a.qbs]:
            g = make_group(i)
            if not g:
                continue
            loss = micro_loss(i, g) / a.qbs
            loss.backward()
            n_ok += 1
            last = loss.item() * a.qbs
        if not n_ok:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0:
            print(f"ep {ep} step {step}/{steps_total} loss {last:.4f}", flush=True)
vram.report("ft_reranker")
model.config.use_cache = True
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
print("saved", a.out, flush=True)
