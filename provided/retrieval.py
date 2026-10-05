#!/usr/bin/env python3
"""Produce the ranked product-ID JSON consumed by ``evaluate.py``."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from eval_utils import load_products, load_retrieval_queries
from retrieval_utils import DEFAULT_TOP_K, retrieve_and_rerank


def save_submission(
    path: str | os.PathLike[str],
    results: list[dict[str, Any]],
    *,
    top_k: int,
) -> None:
    payload = {
        "top_k": top_k,
        "results": results,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", default="data/public/dialogues.jsonl")
    parser.add_argument("--products", default="data/public/products.jsonl")
    parser.add_argument("--output", default="results/retrieval_results.json")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_retrieval_queries(args.dialogues)
    passages = load_products(args.products)

    results = retrieve_and_rerank(
        queries,
        passages,
        top_k=args.top_k,
    )
    save_submission(
        args.output,
        results,
        top_k=args.top_k,
    )
    print(f"Saved {len(results)} ranked lists -> {args.output}")


if __name__ == "__main__":
    main()
