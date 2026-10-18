#!/usr/bin/env python3
"""Query rewriting / HyDE (lecture slides 27, 32, 41-42 neighbourhood): Qwen2.5-1.5B-Instruct (a) rewrites the
conversation into one standalone product-search query, (b) writes a hypothetical product listing (HyDE). First-stage
retrieval variants with the fine-tuned dense model (cached pool embeddings) and BM25, exact-ID metrics at k = 10/30/100.
Saves the best shortlist as cache/cands_public_hyde.npz for rerank_eval.py --cands _hyde."""
import argparse
import json
import os

import numpy as np
import torch

import lecture_common as lc
from zs_eval import bm25_scores

ap = argparse.ArgumentParser()
ap.add_argument("--dense", default="models/dense_lite")
ap.add_argument("--pool", default="cache/public_models_dense_lite_L384.npy")
ap.add_argument("--len", type=int, default=384)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--out", default="results/lecture.jsonl")
a = ap.parse_args()

qs, prods, c, seen = lc.load_public("_lite")
pids = [str(p) for p in c["pids"]]
texts = [prods[p] for p in pids]
gold = c["gold"]
queries = [q["query"] for q in qs]

gen_path = "cache/hyde_gen_public.json"
if os.path.exists(gen_path):
    G = json.load(open(gen_path))
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
    PROMPTS = {
        "rewrite": ("Rewrite the shopper's needs from this conversation as one standalone product search query "
                    "(under 40 words). Keep the product type and every firm constraint. Output only the query.", 60),
        "hyde": ("Write a short product listing (title, then a 2-3 sentence description with key specifications) for the "
                 "single product that best satisfies this shopper's needs. Output only the listing.", 140),
    }
    G = {}
    for name, (inst, mx) in PROMPTS.items():
        outs = []
        for b in range(0, len(queries), a.bs):
            batch = queries[b: b + a.bs]
            txt = [tok.apply_chat_template([{"role": "system", "content": inst},
                                            {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                   for q in batch]
            enc = tok(txt, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                g = m.generate(**enc, max_new_tokens=mx, do_sample=False)
            outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        G[name] = [o.strip() for o in outs]
    json.dump(G, open(gen_path, "w"))
    del m
    torch.cuda.empty_cache()
print("example rewrite:", G["rewrite"][0][:200].replace("\n", " "), flush=True)
print("example hyde:", G["hyde"][0][:300].replace("\n", " "), flush=True)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(a.dense, device="cuda", model_kwargs={"torch_dtype": torch.float16})
model.max_seq_length = a.len
kw = dict(batch_size=64, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
P = torch.tensor(np.load(a.pool)).cuda().half()
Eq = model.encode(queries, prompt_name="query", **kw)
Er = model.encode(G["rewrite"], prompt_name="query", **kw)
Eh = model.encode(G["hyde"], **kw)  # hypothetical document, encoded as a document
Em = torch.nn.functional.normalize(Eq + Eh, dim=-1)
Eqh = model.encode([q + "\n" + h for q, h in zip(queries, G["hyde"])], prompt_name="query", **kw)
sc = {n: (E @ P.T).float().cpu().numpy() for n, E in
      (("dense:orig", Eq), ("dense:rewrite", Er), ("dense:hyde", Eh), ("dense:orig+hyde(avg emb)", Em),
       ("dense:orig+hyde(concat text)", Eqh))}
sc["bm25:orig"] = bm25_scores(queries, pids, texts)
sc["bm25:orig+hyde"] = bm25_scores([q + "\n" + h for q, h in zip(queries, G["hyde"])], pids, texts)
sc["bm25:rewrite"] = bm25_scores(G["rewrite"], pids, texts)


def rrf(names, k=60):
    out = np.zeros_like(sc[names[0]])
    for n in names:
        out += 1.0 / (k + np.argsort(np.argsort(-sc[n], axis=1), axis=1) + 1)
    return out


combos = {n: sc[n] for n in sc}
combos["RRF dense:orig+bm25:orig (submitted first stage)"] = rrf(["dense:orig", "bm25:orig"])
combos["RRF dense:orig+bm25:orig+dense:hyde"] = rrf(["dense:orig", "bm25:orig", "dense:hyde"])
combos["RRF dense:orig+bm25:orig+dense:rewrite"] = rrf(["dense:orig", "bm25:orig", "dense:rewrite"])
combos["RRF dense:avg+bm25:orig+hyde"] = rrf(["dense:orig+hyde(avg emb)", "bm25:orig+hyde"])
combos["RRF dense:orig+dense:hyde+bm25:orig+bm25:rewrite"] = rrf(["dense:orig", "dense:hyde", "bm25:orig", "bm25:rewrite"])
rows, best = {}, None
for n, S in combos.items():
    t = np.argsort(-S, axis=1)[:, :200]
    hit = t == gold[:, None]
    rk = np.where(hit.any(1), hit.argmax(1) + 1, 0)
    row = {f"R@{k}": round(float(((rk > 0) & (rk <= k)).mean()), 4) for k in (1, 10, 30, 100, 200)}
    row["MRR@10"] = round(float(np.where((rk > 0) & (rk <= 10), 1.0 / np.maximum(rk, 1), 0).mean()), 4)
    rows[n] = row
    if n.startswith("RRF") and (best is None or row["R@30"] > best[1]):
        best = (n, row["R@30"], t)
lc.log_row(a.out, {"exp": "hyde_rewrite_first_stage", "rows": rows, "best_rrf_by_R@30": best[0]})
np.savez("cache/cands_public_hyde.npz", pids=c["pids"], qids=c["qids"], gold=gold, top=best[2])
