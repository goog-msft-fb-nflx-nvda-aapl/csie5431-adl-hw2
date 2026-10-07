#!/usr/bin/env python3
"""Zero-shot retrieval eval: exact-gold-ID Recall@k / MRR@10, split by seen/unseen gold.

No LLM judge here (fast iteration); judge-graded numbers come from provided/evaluate.py.
Usage: python zs_eval.py --method bm25|dense|hybrid [--model NAME] [--max-len N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "provided"]
from eval_utils import load_eval_queries, load_products  # noqa: E402

MODELS = {
    # name: (hf id, query prefix, passage prefix, ST prompt_name for query)
    "bge-base": ("BAAI/bge-base-en-v1.5",
                 "Represent this sentence for searching relevant passages: ", "", None),
    "bge-m3": ("BAAI/bge-m3", "", "", None),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "", "", "query"),
    # BLaIR: CLS pooling + L2 norm per its model card (not a sentence-transformers repo)
    "blair-base": ("hyp1231/blair-roberta-base", "", "", None),
    "blair-large": ("hyp1231/blair-roberta-large", "", "", None),
}


def bm25_scores(queries, pids, texts, k1=1.2, b=0.75):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w+\b")
    tf = cv.fit_transform(texts).tocsr().astype(np.float32)
    n = tf.shape[0]
    dl = np.asarray(tf.sum(1)).ravel()
    df = np.asarray((tf > 0).sum(0)).ravel()
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    tf = tf.tocoo()
    denom = tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean())
    from scipy.sparse import csr_matrix
    w = csr_matrix((tf.data * (k1 + 1) / denom * idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
    q = cv.transform(queries)
    q.data[:] = 1
    return (q @ w.T).toarray()  # [nq, n]


def dense_scores(queries, texts, key, max_len, cache):
    import torch
    from sentence_transformers import SentenceTransformer
    hf, qp, pp, pname = MODELS.get(key, (key, "", "", "query"))
    key_s = key.replace("/", "_")
    if key.startswith("blair"):
        from sentence_transformers import models
        tr = models.Transformer(hf, model_args={"torch_dtype": torch.float16})
        model = SentenceTransformer(
            modules=[tr, models.Pooling(tr.get_word_embedding_dimension(), pooling_mode="cls")],
            device="cuda")
    else:
        model = SentenceTransformer(hf, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = max_len
    t0 = time.time()
    if cache and os.path.exists(cache):
        P = np.load(cache)
    else:
        P = model.encode([pp + t for t in texts], batch_size=32, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
        if cache:
            np.save(cache, P)
    t_pool = time.time() - t0
    t0 = time.time()
    kw = {"prompt_name": pname} if pname else {}
    Q = model.encode([qp + q for q in queries], batch_size=64, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False, **kw)
    print(f"pool encode {t_pool:.0f}s (0 if cached), query encode {time.time()-t0:.1f}s, "
          f"peak vram {torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    return Q @ P.T


def rrf(score_list, k=60):
    out = np.zeros_like(score_list[0])
    for s in score_list:
        ranks = np.argsort(np.argsort(-s, axis=1), axis=1) + 1
        out += 1.0 / (k + ranks)
    return out


def report(name, S, gold_idx, seen_mask):
    gs = S[np.arange(len(S)), gold_idx]
    rank = (S > gs[:, None]).sum(1) + 1
    row = {"name": name}
    for sub, m in (("all", np.ones(len(S), bool)), ("seen", seen_mask), ("unseen", ~seen_mask)):
        r = rank[m]
        row[sub] = {f"R@{k}": float((r <= k).mean()) for k in (1, 10, 50, 100)}
        row[sub]["MRR@10"] = float(np.where(r <= 10, 1.0 / r, 0).mean())
        row[sub]["n"] = int(m.sum())
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="bm25")
    ap.add_argument("--model", default="bge-base")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--split", default="public")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results/zs.jsonl")
    a = ap.parse_args()

    root = os.environ.get("HW2_DATA", "data")
    qs = load_eval_queries(f"{root}/{a.split}/dialogues.jsonl")
    prods = load_products(f"{root}/{a.split}/products.jsonl")
    train_pids = set(load_products(f"{root}/train/products.jsonl"))
    pids = list(prods)
    texts = [prods[p] for p in pids]
    pid2i = {p: i for i, p in enumerate(pids)}
    gold = np.array([pid2i[q["gold_pid"]] for q in qs])
    seen = np.array([q["gold_pid"] in train_pids for q in qs])
    queries = [q["query"] for q in qs]
    os.makedirs(a.cache_dir, exist_ok=True)

    tag = a.method
    scores = []
    if a.method in ("bm25", "hybrid"):
        scores.append(bm25_scores(queries, pids, texts))
        if a.method == "bm25":
            S = scores[0]
    if a.method in ("dense", "hybrid"):
        tag = f"{a.method}-{a.model}-L{a.max_len}"
        cache = f"{a.cache_dir}/{a.split}_{a.model.replace(chr(47), chr(95))}_L{a.max_len}.npy"
        scores.append(dense_scores(queries, texts, a.model, a.max_len, cache))
        S = scores[-1] if a.method == "dense" else rrf(scores)
    row = report(tag, S, gold, seen)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
