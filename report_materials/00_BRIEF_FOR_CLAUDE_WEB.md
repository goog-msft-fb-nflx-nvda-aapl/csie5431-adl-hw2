# Brief: write report.pdf for ADL 2026 HW2 (product retrieval for a sales chatbot)

Please write the final report (PDF-ready, concise, about 4-6 pages) from the materials in this folder.
Use only numbers that appear in these files; do not invent results. Where a number is single-run, the
files say so; keep that wording. Say plainly that no RTX 3070 was available, so 3070 speed was not measured
(only H200 timings and a FLOP-based estimate, labelled as an estimate).

## What the report must contain (from the spec, see spec.md "Report")
- **Q1 Retrieval & reranking strategy (4%)**: the strategy in detail; how training data was constructed
  (anchor / positive / negative sampling, external data types used: none, only the provided judge model
  generated labels); loss function specification + rationale; hyperparameters; how many strategies were
  tried and why the submitted one is best.
- **Q2 Generation analysis (2%)**: Qwen2.5-1.5B-Instruct generation from retrieved specs, at least 5
  generation results from different samples with qualitative analysis, at least 3 different prompts
  compared. Scored on richness and depth of analysis. Raw outputs: results/q2_final.jsonl (10 dialogues x 3
  prompts, final retrieval lists) and results/q2.jsonl (8 dialogues, earlier lists). The prompts are in
  code/q2_generate.py (PROMPTS dict). Quote some actual outputs; the worklog has the qualitative notes.

## Files
- `WORKLOG.md`: chronological record of every experiment with settings and measured numbers (source of truth).
- `report.md`: my own draft of the report (a starting point; rewrite freely, check its numbers against WORKLOG).
- `results/`: raw metric rows (zs.jsonl zero-shot baselines, ft.jsonl fine-tuned dense variants, rerank.jsonl
  rerank sweeps, eval_*.json judge-graded metrics + judge timing, stages.json, q2 outputs).
- `figures/`: recall_at_k.png, stages_graded.png, category_seen.png (more can be requested).
- `retrieval_utils.py` (final pipeline), `run.sh` (training recipe), `code/` (all scripts).
- `EDA.md` (data facts), `spec.md` (task spec and grading).

## Final submitted system (one paragraph)
Qwen3-Embedding-0.6B fine-tuned (CachedMNRL, 7 mined hard negatives, lr 3e-5, 4 epochs, batch 128) + BM25,
fused with RRF (k=60) over top-200 lists, top 30 reranked by Qwen3-Reranker-0.6B fine-tuned with listwise
softmax CE on judge-labelled groups (1 positive = gold or judge-accepted alternative with p=0.5, 15
judge-rejected negatives; lr 2e-5), top 10 returned. Judge-graded public R@10 0.986 / MRR@10 0.908
(eval_e2e_final2_default.json). A wall-clock guard lowers the dense max length and rerank depth if the
projected runtime exceeds a 3000 s budget.

## Honesty points to keep in the report
- Hyperparameters were selected using public-set metrics (no separate dev scoring), so public numbers are
  optimistic for the final configuration; private metrics are unknowable here.
- Seed spread measured only for two settings (about 0.004 R@10, 0.008 MRR@10); other comparisons are single runs.
- BLaIR was run zero-shot only as a disclosed ablation (possible overlap with the benchmark's source data); not used.
- Zero-shot MiniLM / bge-reranker / Qwen2.5 rerankers made results worse than no reranking; say so.
- Q2 counts are surface features of 10 samples, not a hallucination rate; no human/judge faithfulness rating.
