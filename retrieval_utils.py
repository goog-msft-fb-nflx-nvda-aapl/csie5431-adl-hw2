#!/usr/bin/env python3
"""Student-editable retrieval and reranking implementation."""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_TOP_K = 10
DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DENSE_MAX_LEN = 512


def _bm25(queries: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75):
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w+\b")
    tf = cv.fit_transform(texts).tocsr().astype(np.float32)
    n = tf.shape[0]
    dl = np.asarray(tf.sum(1)).ravel()
    df = np.asarray((tf > 0).sum(0)).ravel()
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    tf = tf.tocoo()
    denom = tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean())
    w = csr_matrix((tf.data * (k1 + 1) / denom * idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
    q = cv.transform(queries)
    q.data[:] = 1
    return (q @ w.T).toarray()


def _dense(queries: list[str], texts: list[str]):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DENSE_MODEL, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = DENSE_MAX_LEN
    kw = dict(batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    p = model.encode(texts, **kw)
    q = model.encode(queries, prompt_name="query", **kw)
    del model
    torch.cuda.empty_cache()
    return q @ p.T


def _rrf(score_list, k: int = 60):
    out = np.zeros_like(score_list[0])
    for s in score_list:
        out += 1.0 / (k + np.argsort(np.argsort(-s, axis=1), axis=1) + 1)
    return out


def retrieve_and_rerank(
    queries: list[dict[str, Any]],
    passages: dict[str, str],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Return one final ranked PID list per query."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(passages) < top_k:
        raise ValueError(
            f"product collection has {len(passages)} items, fewer than top_k={top_k}"
        )

    # ==================== STUDENT CODE START ====================
    # Baseline: BM25 + zero-shot Qwen3-Embedding-0.6B, fused with RRF (k=60).
    pids = list(passages)
    texts = [passages[p] for p in pids]
    qtexts = [q["query"] for q in queries]
    fused = _rrf([_bm25(qtexts, texts), _dense(qtexts, texts)])
    top = np.argsort(-fused, axis=1)[:, :top_k]
    results: list[dict[str, Any]] = [
        {"qid": q["qid"], "ranked_pids": [pids[j] for j in top[i]]}
        for i, q in enumerate(queries)
    ]
    # ===================== STUDENT CODE END =====================

    return results
