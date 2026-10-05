#!/usr/bin/env python3
"""Q2: generate sales responses with Qwen2.5-1.5B-Instruct from retrieved specs, 3 prompt variants.
Usage: python q2_generate.py --results results/rr_xxx.json --n 8 --out results/q2.jsonl"""
import argparse
import json
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "provided"]
from eval_utils import load_eval_queries, load_products  # noqa: E402

PROMPTS = {
    # P1: bare instruction, all retrieved specs pasted in rank order
    "plain": ("You are a helpful shopping assistant.",
              "Conversation so far:\n{conv}\n\nProduct specs:\n{specs}\n\nWrite the assistant's next reply recommending a product."),
    # P2: grounded sales-assistant with explicit constraints
    "grounded": ("You are a sales assistant. Recommend exactly one product from the candidate list. Use only facts that appear "
                 "in the spec text; do not invent features, prices or numbers. Mention the product title and 2-3 specific "
                 "features that match what the user asked for. Keep it under 80 words.",
                 "Conversation so far:\n{conv}\n\nCandidate products (best match first):\n{specs}\n\nReply to the user."),
    # P3: reason-then-answer, with a check against the user's stated constraints
    "checklist": ("You are a careful sales assistant. First list the user's firm requirements as short bullets, then pick "
                  "the single candidate that satisfies the most of them, quoting the matching phrase from its spec, and "
                  "say if any requirement is not confirmed by the spec. Use only the given specs.",
                  "Conversation so far:\n{conv}\n\nCandidate products (best match first):\n{specs}\n\nFormat:\nRequirements:\n- ...\nRecommendation: <title> because ...\nNot confirmed: ..."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--dialogues", default="data/public/dialogues.jsonl")
    ap.add_argument("--products", default="data/public/products.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--spec-chars", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/q2.jsonl")
    a = ap.parse_args()
    qs = {q["qid"]: q for q in load_eval_queries(a.dialogues)}
    prods = load_products(a.products)
    ranked = {r["qid"]: r["ranked_pids"] for r in json.load(open(a.results))["results"]}
    rng = random.Random(a.seed)
    qids = rng.sample(sorted(qs), a.n)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.bfloat16).cuda().eval()
    out = open(a.out, "w")
    for qid in qids:
        q, pids = qs[qid], ranked[qid][: a.k]
        specs = "\n\n".join(f"[{j + 1}] {prods[p][: a.spec_chars]}" for j, p in enumerate(pids))
        for name, (sys_p, user_t) in PROMPTS.items():
            msgs = [{"role": "system", "content": sys_p},
                    {"role": "user", "content": user_t.format(conv=q["query"], specs=specs)}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                g = model.generate(ids, max_new_tokens=220, do_sample=False)
            text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
            spec_nums = set(re.findall(r"\d+(?:\.\d+)?", " ".join(prods[p] for p in pids)))
            out_nums = set(re.findall(r"\d+(?:\.\d+)?", text))
            row = {"qid": qid, "prompt": name, "gold_pid": q["gold_pid"], "gold_rank": (pids.index(q["gold_pid"]) + 1
                   if q["gold_pid"] in pids else None), "retrieved": pids, "query": q["query"],
                   "gold_title": prods[q["gold_pid"]].split("\n")[0], "top1_title": prods[pids[0]].split("\n")[0],
                   "response": text, "n_words": len(text.split()),
                   "numbers_not_in_specs": sorted(out_nums - spec_nums)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    print("done", a.n * len(PROMPTS))


if __name__ == "__main__":
    main()
