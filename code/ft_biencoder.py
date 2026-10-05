#!/usr/bin/env python3
"""Fine-tune a sentence-transformers bi-encoder on train dialogues with CachedMNRL (+ optional
mined hard negatives from cache/cands_train.npz). Holds out a product-disjoint dev split."""
import argparse
import os
import random
import sys

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers import SentenceTransformerTrainingArguments as Args
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "provided"))
from eval_utils import load_eval_queries, load_products  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3-Embedding-0.6B")
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--scale", type=float, default=20.0)
ap.add_argument("--n-neg", type=int, default=0)
ap.add_argument("--neg-lo", type=int, default=5)
ap.add_argument("--neg-hi", type=int, default=50)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--dev-frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--synth", default="")  # optional jsonl of {"query","pid"} extra positives
ap.add_argument("--judge-filter", default="")  # npy bool [nq, depth] True = judge accepts (drop)
a = ap.parse_args()

root = os.path.join(os.path.dirname(__file__), "..", "data")
qs = load_eval_queries(f"{root}/train/dialogues.jsonl")
prods = load_products(f"{root}/train/products.jsonl")
c = dict(np.load("cache/cands_train.npz"))
pids = list(c["pids"])
pid2i = {p: i for i, p in enumerate(pids)}
rng = random.Random(a.seed)
targets = sorted({q["gold_pid"] for q in qs})
rng.shuffle(targets)
dev_t = set(targets[: int(len(targets) * a.dev_frac)])
train_idx = [i for i, q in enumerate(qs) if q["gold_pid"] not in dev_t]
dev_idx = [i for i, q in enumerate(qs) if q["gold_pid"] in dev_t]
np.save(f"{a.out}_dev_idx.npy", np.array(dev_idx))
title = {p: prods[p].split("\n", 1)[0] for p in prods}
jf = np.load(a.judge_filter) if a.judge_filter else None

rows = {"query": [], "positive": []}
for k in range(a.n_neg):
    rows[f"negative{k}"] = []
for i in train_idx:
    g = qs[i]["gold_pid"]
    cand = []
    for r in range(a.neg_lo, min(a.neg_hi, c["top"].shape[1])):
        if jf is not None and r < jf.shape[1] and jf[i, r]:
            continue
        p = pids[c["top"][i, r]]
        if p != g and title[p] != title[g]:
            cand.append(p)
    if len(cand) < a.n_neg:
        continue
    rows["query"].append(qs[i]["query"])
    rows["positive"].append(prods[g])
    for k, p in enumerate(rng.sample(cand, a.n_neg)):
        rows[f"negative{k}"].append(prods[p])
print("train rows", len(rows["query"]), "dev queries", len(dev_idx), flush=True)
ds = Dataset.from_dict(rows).shuffle(seed=a.seed)

model = SentenceTransformer(a.base, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
model.max_seq_length = a.max_len
qprompt = model.prompts.get("query", "") if hasattr(model, "prompts") else ""
loss = CachedMultipleNegativesRankingLoss(model, scale=a.scale, mini_batch_size=32)
args = Args(output_dir=a.out + "_tmp", num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
            learning_rate=a.lr, warmup_ratio=0.1, bf16=True, logging_steps=5, save_strategy="no",
            batch_sampler=BatchSamplers.NO_DUPLICATES, prompts={"query": qprompt},
            report_to="none", seed=a.seed, lr_scheduler_type="linear")
SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
model.save_pretrained(a.out)
print("saved", a.out, flush=True)
