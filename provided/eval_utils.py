#!/usr/bin/env python3

import json


def build_eval_query(conversation: list[dict]) -> str:
    """Convert all pre-recommendation conversation turns into one query."""
    lines = [
        f"{turn.get('role')}: {(turn.get('content') or '').strip()}"
        for turn in conversation or []
    ]
    return "\n".join(line for line in lines if line.split(": ", 1)[-1]).strip()


def _load_queries(
    dialogues_path: str,
    *,
    include_gold: bool,
) -> list[dict[str, str]]:
    queries = []
    seen_qids = set()
    with open(dialogues_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("dialogue_id")
            query = build_eval_query(row.get("conversation", []))
            if not isinstance(qid, str) or not qid:
                raise ValueError(
                    f"{dialogues_path}:{line_number}: missing dialogue_id"
                )
            if qid in seen_qids:
                raise ValueError(f"{dialogues_path}:{line_number}: duplicate {qid}")
            if not query:
                raise ValueError(f"{dialogues_path}:{line_number}: empty conversation")
            seen_qids.add(qid)
            query_row = {"qid": qid, "query": query}
            if include_gold:
                target = row.get("target")
                if not isinstance(target, str) or not target:
                    raise ValueError(
                        f"{dialogues_path}:{line_number}: missing target"
                    )
                query_row["gold_pid"] = target
            queries.append(query_row)
    return queries


def load_retrieval_queries(dialogues_path: str) -> list[dict[str, str]]:
    """Read query IDs and conversations without exposing gold product IDs."""
    return _load_queries(dialogues_path, include_gold=False)


def load_eval_queries(dialogues_path: str) -> list[dict[str, str]]:
    """Read query IDs, conversations, and gold product IDs for evaluation."""
    return _load_queries(dialogues_path, include_gold=True)


def load_products(products_path: str, field: str = "text") -> dict[str, str]:
    """Load product data as ``{parent_asin: product text}``.

    ``text`` combines the title and description. ``title`` and ``description``
    can also be selected individually.
    """
    if field not in {"text", "title", "description"}:
        raise ValueError("product field must be 'text', 'title', or 'description'")

    products = {}
    with open(products_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row.get("parent_asin")
            title = (row.get("title") or "").strip()
            description = (row.get("description") or "").strip()
            if not isinstance(pid, str) or not pid:
                raise ValueError(
                    f"{products_path}:{line_number}: missing parent_asin"
                )
            if pid in products:
                raise ValueError(f"{products_path}:{line_number}: duplicate {pid}")
            if field == "text":
                value = "\n".join(part for part in (title, description) if part)
            else:
                value = title if field == "title" else description
            if not value:
                raise ValueError(
                    f"{products_path}:{line_number}: product {pid!r} has no {field}"
                )
            products[pid] = value
    return products
