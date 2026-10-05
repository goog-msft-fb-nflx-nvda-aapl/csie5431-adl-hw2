# ADL HW2 — experiment log

## 2026-10-05 — Setup: spec intake, data download, EDA

- Read `ADL2026-HW2.pptx.pdf` (ground truth) + cross-checked the Qwen-extracted
  `.pptx.md` against it page by page — no discrepancies found. Consolidated into
  `docs/spec/spec.md`.
- Pulled the actual provided files from the TA's Google Drive folder (link embedded in
  the PDF, page 3) via `download_hw2_data.sh` / `gdown --folder`. Matches the slides'
  stated sizes exactly: train 4,830 dialogues / 50,000 products, public 1,000 dialogues
  / 75,000 products.
- Ran EDA directly on the real files (not just the slide numbers) — see `docs/EDA.md`.
  Headline finding: the product pool is general-domain (6 roughly-even top-level
  categories: Sports & Outdoors, Pet Supplies, Toys & Games, Home & Kitchen, Beauty,
  Beauty & Personal Care), not beauty-specific despite the `beauty-*` dialogue-ID
  prefix, and 500/1000 public-test gold targets are products never seen in the train
  pool at all — a retriever/reranker needs to generalize from product *text*, not
  memorize train-pool identities.
- Set up this repo (`goog-msft-fb-nflx-nvda-aapl/csie5431-adl-hw2`, private) and
  `CLAUDE.md` with project conventions (GPU server gotchas, GitHub rules, working
  style) carried over from the HW1 project, so a fresh session has full context without
  re-deriving it.
- 2026-10-05 later: Claude + Gemini survey responses saved in `docs/survey/`, synthesized
  into a numbered method backlog in `docs/TODO.md`. Perplexity not run yet. No numbers
  from the surveys are measured on our setup.
- Next (as of setup): write the deep-research survey prompt (incorporating the EDA findings above),
  run it externally (Claude/Gemini/Perplexity), then start the candidate-method backlog
  the same way HW1 did — see `docs/TODO.md`.

## 2026-10-05 — Zero-shot baselines (exp 1)

Setup: `code/zs_eval.py`, public split (1,000 queries, 75k pool), exact gold-ID match (no
LLM judge), queries = all turns via `build_eval_query`, passages = title+description,
dense models fp16, max_seq_len 512, no fine-tuning. BM25 = sklearn CountVectorizer
(english stopwords, k1=1.2, b=0.75). Hybrid = RRF k=60 of BM25 + dense over full ranking.
gsm-gpu2 GPU2 (H200), env `adl_hw2_env`. Raw rows: `/home/jtan/adl_hw2/results/zs.jsonl`.

| method | R@1 | R@10 | R@50 | R@100 | MRR@10 | R@10 seen / unseen |
|---|---|---|---|---|---|---|
| bm25 | 0.513 | 0.765 | 0.882 | 0.915 | 0.595 | 0.778 / 0.752 |
| bge-base dense | 0.246 | 0.556 | 0.748 | 0.830 | 0.335 | 0.550 / 0.562 |
| bge-base + bm25 | 0.392 | 0.724 | 0.903 | 0.936 | 0.498 | 0.724 / 0.724 |
| qwen3-emb-0.6b dense | 0.281 | 0.621 | 0.814 | 0.870 | 0.381 | 0.616 / 0.626 |
| qwen3-emb-0.6b + bm25 | 0.439 | 0.762 | 0.917 | 0.948 | 0.537 | 0.764 / 0.760 |
| bge-m3 dense | 0.195 | 0.473 | 0.688 | 0.768 | 0.279 | 0.482 / 0.464 |
| bge-m3 + bm25 | 0.362 | 0.694 | 0.887 | 0.925 | 0.462 | 0.704 / 0.684 |

Pool encode time / peak VRAM on H200 (not the 3070): bge-base 59s / 0.84GB, qwen3-0.6b
261s / 5.53GB, bge-m3 102s / 1.87GB. Peak VRAM is torch max_memory_allocated within each process (the cached hybrid runs report lower peaks since they skip pool encoding).

Observations (measured): BM25 beats every zero-shot dense model at R@1/R@10/MRR@10.
Hybrid beats BM25 only at R@50/R@100 (qwen3 hybrid R@100 0.948 vs 0.915) and ties it at
R@10 for qwen3. Seen vs unseen gold R@10 differ by <=2 points for every method. Not yet
measured: judge-graded metrics, fine-tuned models, any reranker, 3070 timing, train split.

## 2026-10-05 — Perplexity survey intake

`docs/survey/perplexity_survey_1.md` saved (pre-signed attachment URLs in its citations
stripped). Folded into `docs/TODO.md` as backlog items 14-18. Nothing in it is measured
by us. Points of agreement with the Claude/Gemini surveys: BM25 + dense RRF first stage,
fine-tuned Qwen3-Embedding-0.6B as the main dense candidate, cross-encoder rerank on
top ~50, product-disjoint dev split, full-dialogue queries, cached MNRL, conservative
hard-negative filtering. Points of difference: it recommends cross-encoder/ms-marco-
MiniLM-L6-v2 as the primary reranker and mxbai-embed-large-v1 as the safe dense
fallback; it advises against Qwen2.5-1.5B as an online reranker (offline teacher only)
while the Claude survey recommends judge-prompt pointwise rerank on the top 10-20. Its
claim that the Qwen judge is nondeterministic does not match `evaluate.py` (greedy
decode, `CUBLAS_WORKSPACE_CONFIG` set); the rerank-vs-teacher question is open until
tested. Model sizes/context lengths and the cited arXiv/HF pages are unverified by us.
