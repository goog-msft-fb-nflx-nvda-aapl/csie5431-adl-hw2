#!/usr/bin/env python3
"""Student-editable retrieval and reranking implementation.

Pipeline: fine-tuned Qwen3-Embedding-0.6B dense retrieval (optionally fused with BM25 via RRF) ->
top-N shortlist -> fine-tuned Qwen3-Reranker-0.6B (yes/no logit) on the top RERANK_DEPTH -> top_k.
Models are read from ./models/{dense,reranker} next to this file (put there by download.sh); if a
directory is missing the corresponding public base model is loaded from the Hugging Face hub.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np


DEFAULT_TOP_K = 10

_HERE = os.path.dirname(os.path.abspath(__file__))
DENSE_BASE = "Qwen/Qwen3-Embedding-0.6B"
RERANK_BASE = "Qwen/Qwen3-Reranker-0.6B"
DENSE_MODEL = os.environ.get("HW2_DENSE", os.path.join(_HERE, "models", "dense"))
RERANK_MODEL = os.environ.get("HW2_RERANK", os.path.join(_HERE, "models", "reranker"))
USE_BM25 = os.environ.get("HW2_BM25", "1") == "1"
DENSE_MAX_LEN = int(os.environ.get("HW2_DENSE_LEN", "512"))  # upper bound; may be lowered by the time guard
BUDGET_S = float(os.environ.get("HW2_BUDGET", "3000"))  # wall-clock guard for the whole call (limit is 3600s)
DENSE_SHARE = 0.45  # at most this share of the budget goes to pool encoding
SAFETY_S = 300.0
DENSE_BS = int(os.environ.get("HW2_DENSE_BS", "32"))
SHORTLIST = int(os.environ.get("HW2_SHORTLIST", "200"))
RERANK_DEPTH = int(os.environ.get("HW2_DEPTH", "30"))  # 0 disables reranking
RERANK_DOC_CHARS = int(os.environ.get("HW2_DOC_CHARS", "1800"))
RERANK_MAX_LEN = 1024
RERANK_BS = int(os.environ.get("HW2_RERANK_BS", "16"))
RRF_K = 60

_INSTR = "Given a web search query, retrieve relevant passages that answer the query"
_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
           "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _log(msg: str) -> None:
    print(f"[retrieval_utils] {msg}", file=sys.stderr, flush=True)


def _peak() -> str:
    try:
        import torch
        return f"peak vram {torch.cuda.max_memory_allocated() / 2**30:.2f}GB"
    except Exception:
        return ""


def _model_path(path: str, base: str) -> str:
    return path if os.path.isdir(path) else base


def _bm25_topn(queries: list[str], texts: list[str], n: int, k1: float = 1.2, b: float = 0.75):
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w+\b")
    tf = cv.fit_transform(texts).tocsr().astype(np.float32)
    n_docs = tf.shape[0]
    dl = np.asarray(tf.sum(1)).ravel()
    df = np.asarray((tf > 0).sum(0)).ravel()
    idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    coo = tf.tocoo()
    denom = coo.data + k1 * (1 - b + b * dl[coo.row] / dl.mean())
    w = csr_matrix((coo.data * (k1 + 1) / denom * idf[coo.col], (coo.row, coo.col)), shape=coo.shape)
    q = cv.transform(queries)
    q.data[:] = 1
    out = np.zeros((len(queries), n), dtype=np.int64)
    for s in range(0, len(queries), 200):
        sc = (q[s:s + 200] @ w.T).toarray()
        part = np.argpartition(-sc, n, axis=1)[:, :n]
        order = np.argsort(-np.take_along_axis(sc, part, 1), axis=1, kind="stable")
        out[s:s + 200] = np.take_along_axis(part, order, 1)
    return out


def _dense_topn(queries: list[str], texts: list[str], n: int, t_start: float):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_model_path(DENSE_MODEL, DENSE_BASE), device="cuda",
                                model_kwargs={"torch_dtype": torch.float16})
    kw = dict(batch_size=DENSE_BS, normalize_embeddings=True, convert_to_tensor=True,
              show_progress_bar=False)
    # time guard: time a sample at the longest length, project the whole pool, shrink max length if needed
    lens = np.array([len(x) for x in model.tokenizer(texts, add_special_tokens=True,
                                                     truncation=True, max_length=1024)["input_ids"]])
    rng = np.random.RandomState(0)
    sidx = rng.choice(len(texts), min(len(texts), 1000), replace=False)
    sample = [texts[i] for i in sidx]
    model.max_seq_length = DENSE_MAX_LEN
    model.encode(sample[:64], **kw)  # warm-up
    torch.cuda.synchronize()
    t = time.time()
    model.encode(sample, **kw)
    torch.cuda.synchronize()
    rate = float(np.minimum(lens[sidx], DENSE_MAX_LEN).sum()) / (time.time() - t)  # tokens/s at the longest length
    allowed = DENSE_SHARE * BUDGET_S - (time.time() - t_start)
    max_len = DENSE_MAX_LEN
    for cand_len in sorted({DENSE_MAX_LEN, 384, 256, 192, 128}, reverse=True):
        if cand_len > DENSE_MAX_LEN:
            continue
        max_len = cand_len
        if float(np.minimum(lens, cand_len).sum()) / rate <= allowed:
            break
    _log(f"dense max_len {max_len} (sample rate {rate:.0f} tok/s, allowed {allowed:.0f}s)")
    model.max_seq_length = max_len
    p = model.encode(texts, **kw)
    qe = model.encode(queries, prompt_name="query", **kw)
    out = np.zeros((len(queries), n), dtype=np.int64)
    for s in range(0, len(queries), 256):
        sc = qe[s:s + 256] @ p.T
        out[s:s + 256] = sc.topk(n, dim=1).indices.cpu().numpy()
    del model, p, qe
    torch.cuda.empty_cache()
    return out


def _rrf_topn(lists: list[np.ndarray], n: int) -> np.ndarray:
    out = np.zeros((lists[0].shape[0], n), dtype=np.int64)
    for i in range(lists[0].shape[0]):
        score: dict[int, float] = {}
        for lst in lists:
            for r, j in enumerate(lst[i]):
                score[j] = score.get(j, 0.0) + 1.0 / (RRF_K + r + 1)
        best = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        out[i] = [j for j, _ in best]
    return out


def _rerank(qtexts: list[str], texts: list[str], cand: np.ndarray, depth_max: int, t_start: float):
    """Rerank the top-d of each query's shortlist; d shrinks if the projected time exceeds the budget."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = _model_path(RERANK_MODEL, RERANK_BASE)
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    w = model.lm_head.weight[[yes, no]]
    out = cand.copy()
    nq, chunk = len(qtexts), 16
    pairs_done, t_scoring, depths = 0, 0.0, []
    for s in range(0, nq, chunk):
        qs = range(s, min(s + chunk, nq))
        avail = BUDGET_S - (time.time() - t_start) - SAFETY_S
        if pairs_done:
            d = int(min(depth_max, 0.9 * avail * (pairs_done / t_scoring) / (nq - s)))
        else:
            d = depth_max
        if d < 2:
            break  # out of time: remaining queries keep the first-stage order
        t = time.time()
        items = [(i, j) for i in qs for j in range(d)]
        txt = [_PREFIX + f"<Instruct>: {_INSTR}\n<Query>: {qtexts[i]}\n<Document>: "
               f"{texts[cand[i, j]][:RERANK_DOC_CHARS]}" + _SUFFIX for i, j in items]
        order = np.argsort([len(x) for x in txt])
        sc = np.zeros(len(txt), dtype=np.float32)
        for b in range(0, len(order), RERANK_BS):
            idx = order[b:b + RERANK_BS]
            enc = tok([txt[k] for k in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=RERANK_MAX_LEN).to("cuda")
            with torch.inference_mode():
                h = model.model(**enc).last_hidden_state[:, -1, :]
                lg = (h @ w.T).float()
            sc[idx] = (lg[:, 0] - lg[:, 1]).cpu().numpy()
        sc = sc.reshape(len(qs), d)
        for a, i in enumerate(qs):
            out[i, :d] = cand[i, :d][np.argsort(-sc[a], kind="stable")]
        torch.cuda.synchronize()
        t_scoring += time.time() - t
        pairs_done += len(items)
        depths.append(d)
    _log(f"rerank depth min/mean {min(depths) if depths else 0}/{np.mean(depths) if depths else 0:.1f}, "
         f"{pairs_done} pairs in {t_scoring:.0f}s")
    del model
    torch.cuda.empty_cache()
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
    # Only modify the code inside this block.
    t0 = time.time()
    pids = list(passages)
    texts = [passages[p] for p in pids]
    qtexts = [q["query"] for q in queries]
    n = min(max(SHORTLIST, top_k, RERANK_DEPTH), len(pids))

    lists = [_dense_topn(qtexts, texts, n, t0)]
    _log(f"dense done {time.time() - t0:.0f}s {_peak()}")
    if USE_BM25:
        lists.append(_bm25_topn(qtexts, texts, n))
        _log(f"bm25 done {time.time() - t0:.0f}s")
    cand = _rrf_topn(lists, n) if len(lists) > 1 else lists[0]

    depth = min(RERANK_DEPTH, n)
    if depth > 0:
        cand = _rerank(qtexts, texts, cand, depth, t0)
        _log(f"rerank done {time.time() - t0:.0f}s {_peak()}")

    results: list[dict[str, Any]] = [
        {"qid": q["qid"], "ranked_pids": [pids[j] for j in cand[i, :top_k]]}
        for i, q in enumerate(queries)
    ]
    # ===================== STUDENT CODE END =====================

    return results
