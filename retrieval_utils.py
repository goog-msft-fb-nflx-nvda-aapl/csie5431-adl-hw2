#!/usr/bin/env python3
"""Student-editable retrieval and reranking implementation."""

from __future__ import annotations

from typing import Any


DEFAULT_TOP_K = 10


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
    #
    # Implement retrieval and optional reranking here. Return one dictionary
    # per query in the following form:
    # {
    #     "qid": query["qid"],
    #     "ranked_pids": [pid_1, pid_2, ..., pid_top_k],
    # }
    results: list[dict[str, Any]] = []
    raise NotImplementedError("Implement retrieval inside the student code block")
    # ===================== STUDENT CODE END =====================

    return results
