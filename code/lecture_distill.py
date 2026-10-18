#!/usr/bin/env python3
"""Reranker-to-retriever distillation (lecture slide 31, PairDistill idea, our own listwise variant).
--stage teacher : score each train dialogue's top-20 zero-shot candidates (+gold) with the fine-tuned reranker.
--stage train   : fine-tune Qwen3-Embedding-0.6B with KL(softmax(teacher scores / T) || softmax(scale * cos)) over the
                  candidate list of one dialogue (one dialogue per micro-step, gradient checkpointing, accumulate)."""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "provided"]
from eval_utils import load_eval_queries, load_products  # noqa: E402
import vram  # noqa: E402

vram.apply_cap()
ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
ap.add_argument("--teacher", default="models/rr_g8")
ap.add_argument("--out", default="models/dense_pd")
ap.add_argument("--n", type=int, default=16)         # candidates per list used in training (incl. gold)
ap.add_argument("--T", type=float, default=2.0)      # teacher temperature
ap.add_argument("--scale", type=float, default=20.0)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--max-len", type=int, default=384)
ap.add_argument("--gold-ce", type=float, default=0.0)  # weight of an extra CE on the gold item
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

root = os.environ.get("HW2_DATA", "data")
qs = load_eval_queries(f"{root}/train/dialogues.jsonl")
prods = load_products(f"{root}/train/products.jsonl")
c = dict(np.load("cache/cands_train.npz"))
pids = [str(p) for p in c["pids"]]
top20 = c["top"][:, :20]
gold = c["gold"]
T_PATH = "cache/teacher_train.npy"

if a.stage == "teacher":
    from rerank_eval import score_qwen3rr
    # list per dialogue = top-20 candidates, gold appended if absent (index -1 slot filled with gold always)
    lists = []
    for i in range(len(qs)):
        lst = list(top20[i])
        if gold[i] not in lst:
            lst = lst[:19] + [gold[i]]
        lists.append(lst)
    lists = np.array(lists)
    qq = [qs[i]["query"] for i in range(len(qs)) for _ in range(20)]
    dd = [prods[pids[j]][:1400] for i in range(len(qs)) for j in lists[i]]
    flat = score_qwen3rr(qq, dd, 16, 1024, model_id=a.teacher)
    np.save(T_PATH, flat.reshape(len(qs), 20))
    np.save("cache/teacher_train_lists.npy", lists)
    vram.report("teacher")
    sys.exit(0)

from sentence_transformers import SentenceTransformer
T = np.load(T_PATH)
lists = np.load("cache/teacher_train_lists.npy")
rng = random.Random(a.seed)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
model.max_seq_length = a.max_len
model[0].auto_model.gradient_checkpointing_enable()
qprompt = model.prompts.get("query", "")
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
idx = list(range(len(qs)))
steps_total = int(a.epochs * len(idx) / a.accum)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / (0.1 * steps_total)) * max(0.0, 1 - s / steps_total))


def embed(texts):
    feats = model.tokenize(texts)
    feats = {k: v.cuda() for k, v in feats.items()}
    return model(feats)["sentence_embedding"].float()


model.train()
step, micro = 0, 0
done = False
while not done:
    rng.shuffle(idx)
    for i in idx:
        lst, t = lists[i], T[i]
        # keep the gold plus the best-scored (teacher) / highest-ranked candidates up to n
        keep = [k for k in range(20) if lst[k] != gold[i]]
        keep = keep[: a.n - 1]
        sel = [k for k in range(20) if lst[k] == gold[i]] + keep
        docs = [prods[pids[lst[k]]] for k in sel]
        eq = embed([qprompt + qs[i]["query"]])
        ed = embed(docs)
        stu = a.scale * (eq @ ed.T)[0]
        tt = torch.tensor(t[sel], device="cuda", dtype=torch.float32)
        loss = torch.nn.functional.kl_div(torch.log_softmax(stu, -1), torch.softmax(tt / a.T, -1), reduction="sum")
        if a.gold_ce:
            loss = loss + a.gold_ce * torch.nn.functional.cross_entropy(stu.unsqueeze(0), torch.zeros(1, dtype=torch.long, device="cuda"))
        (loss / a.accum).backward()
        micro += 1
        if micro % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                print(f"step {step}/{steps_total} loss {loss.item():.4f}", flush=True)
            if step >= steps_total:
                done = True
                break
vram.report("distill")
model.save_pretrained(a.out)
print("saved", a.out, flush=True)
