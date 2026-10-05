#!/usr/bin/env python3
"""Report figures from saved result files (results/*.jsonl, results/eval_*.json, ranked-list jsons)."""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "provided"]
from eval_utils import load_eval_queries  # noqa: E402

os.makedirs("figures", exist_ok=True)


def rows(path):
    return {json.loads(l)["name"]: json.loads(l) for l in open(path)}


zs, ft = rows("results/zs.jsonl"), rows("results/ft.jsonl")

# 1. first-stage exact recall@k
ks = [1, 10, 50, 100]
series = {
    "BM25": zs["bm25"], "zero-shot Qwen3-0.6B": zs["dense-qwen3-0.6b-L512"],
    "zero-shot + BM25 (RRF)": zs["hybrid-qwen3-0.6b-L512"],
    "ft in-batch (ftA)": ft["dense-models/ftA_inbatch-L512"],
    "ft 1 hard neg (ftB)": ft["dense-models/ftB_hn1-L512"],
    "ft 3 hard neg, lr2e-5 3ep (ftF)": ft["dense-models/ftF_hn3_lr2e5_e3-L512"],
}
plt.figure(figsize=(6, 4))
for name, r in series.items():
    plt.plot(range(len(ks)), [r["all"][f"R@{k}"] for k in ks], marker="o", label=name)
plt.xticks(range(len(ks)), [f"R@{k}" for k in ks]); plt.ylabel("exact-ID recall (public)")
plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("figures/recall_at_k.png", dpi=160)

# 2. graded metrics per pipeline stage vs tier thresholds
stages = json.load(open("results/stages.json"))  # [{"name", "R@10", "MRR@10"}]
fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(stages))
ax.bar([i - .2 for i in x], [s["R@10"] for s in stages], .4, label="graded R@10")
ax.bar([i + .2 for i in x], [s["MRR@10"] for s in stages], .4, label="graded MRR@10")
for y, lab in ((0.86, "Simple R"), (0.885, "Strong R"), (0.95, "Boss R")):
    ax.axhline(y, ls="--", lw=.7, color="gray"); ax.text(len(stages) - .5, y, lab, fontsize=6, va="bottom")
for y, lab in ((0.65, "Simple M"), (0.68, "Strong M"), (0.80, "Boss M")):
    ax.axhline(y, ls=":", lw=.7, color="gray"); ax.text(len(stages) - .5, y, lab, fontsize=6, va="bottom")
ax.set_xticks(list(x)); ax.set_xticklabels([s["name"] for s in stages], fontsize=6, rotation=15)
ax.set_ylim(0.4, 1.0); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig("figures/stages_graded.png", dpi=160)

# 3. per-category + 4. seen/unseen from the final ranked list
final = sys.argv[1] if len(sys.argv) > 1 else "results/e2e_final.json"
qs = {q["qid"]: q for q in load_eval_queries("data/public/dialogues.jsonl")}
train = {json.loads(l)["parent_asin"] for l in open("data/train/products.jsonl")}
cat, seen = defaultdict(list), defaultdict(list)
for r in json.load(open(final))["results"]:
    g = qs[r["qid"]]["gold_pid"]
    rank = r["ranked_pids"].index(g) + 1 if g in r["ranked_pids"] else 0
    c = r["qid"].replace("public-test-", "").rsplit("-", 1)[0]
    for d, k in ((cat, c), (seen, "seen" if g in train else "unseen")):
        d[k].append((rank > 0, 1 / rank if rank else 0))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, d, t in ((axs[0], cat, "per category"), (axs[1], seen, "seen vs unseen gold")):
    names = sorted(d)
    ax.bar([i - .2 for i in range(len(names))], [sum(a for a, _ in d[n]) / len(d[n]) for n in names], .4, label="exact R@10")
    ax.bar([i + .2 for i in range(len(names))], [sum(b for _, b in d[n]) / len(d[n]) for n in names], .4, label="exact MRR@10")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([f"{n}\n(n={len(d[n])})" for n in names], fontsize=6)
    ax.set_title(t, fontsize=9); ax.set_ylim(0, 1.15); ax.legend(fontsize=7, loc="upper right", ncol=2)
plt.tight_layout(); plt.savefig("figures/category_seen.png", dpi=160)
print("figures written")
