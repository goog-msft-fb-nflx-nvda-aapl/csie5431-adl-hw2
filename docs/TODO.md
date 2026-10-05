# ADL HW2 — TODO / backlog

## Current state (2026-10-05)

- [x] Spec read + consolidated (`docs/spec/spec.md`), PDF-verified against the Qwen OCR.
- [x] Data downloaded (`data/train`, `data/public`), EDA done (`docs/EDA.md`).
- [x] Repo + `CLAUDE.md` set up.
- [x] Deep-research survey prompt, see `docs/survey/`.
- [x] Survey run on Claude + Gemini + Perplexity, synthesized into the method backlog
      below.
- [x] Zero-shot baselines on public (BM25, bge-base, qwen3-0.6b, bge-m3, hybrids), see
      `docs/WORKLOG.md`. Script `code/zs_eval.py` is an analysis script, not yet a
      `retrieve_and_rerank()`.
- [~] Baseline `retrieve_and_rerank()` (BM25 + zero-shot Qwen3-0.6B, RRF) written in
      `retrieval_utils.py`; judge-graded `evaluate.py` run in progress (tmux `eval1` on
      gsm-gpu2), results go in `docs/WORKLOG.md`.
- [ ] Training-data construction strategy for a bi-encoder/reranker (anchor = query,
      positive = gold + judge-acceptable alternatives if obtainable, negative sampling
      strategy — in-batch, hard negatives from BM25/dense-retrieval misses, etc.) —
      report Q1 requires this to be explained in detail either way.
- [ ] At least 3 distinct retrieval/reranking strategies tried, compared, and written
      up (report Q1 explicitly asks "how many did you try, why is the submitted one best").
- [ ] Q2 (generation analysis): Qwen2.5-1.5B-Instruct generation over retrieved specs,
      >=5 examples, >=3 different prompts compared.
- [ ] Package for submission: `download.sh` (model + external data only, <=30min @
      10MB/s, no pip install), `run.sh` (reproduces training), `README.md`
      (step-by-step, must match `run.sh`), `report.pdf`.
- [ ] Verify `run.sh`/`evaluate.py` finishes public-set retrieval within 1hr on RTX 3070
      8GB VRAM equivalent constraints (tighter than HW1's 3080Ti 10GB — mind model size).

## Method backlog (from survey: `docs/survey/claude_survey_1.md`, `geimini_survey_1.md`)

All throughput/latency/benchmark numbers in the surveys are unmeasured estimates or
unverified citations; nothing below is adopted until measured here. Perplexity response
run (Claude, Gemini, Perplexity responses all in `docs/survey/`).

Measure first:
1. Benchmark encode throughput + rerank pairs/s on the 8GB target (or a proxy) for
   bge-base, Qwen3-Embedding-0.6B, bge-m3 with plain transformers/sentence-transformers,
   fp16, length-sorted batching. Surveys disagree by ~2x on encode time; this sets
   passage max length and rerank depth.
2. [done 2026-10-06, see worklog] Test env packages: spec lets us add packages as long as
   they are listed in `requirements.txt`; no vLLM/bm25s in the recommended set. Our BM25
   uses scikit-learn + scipy (added to `requirements.txt`).
3. Zero-shot baselines through `retrieval.py` + `evaluate.py`: BM25, bge-base,
   Qwen3-Embedding-0.6B, bge-m3, gte-Qwen2-1.5B, plus jina-v5 / others if they load.
   Report exact-ID recall and judge-graded recall side by side.
4. Build a product-disjoint internal dev split (gold products excluded from training as
   positives and negatives) to measure cold-start without tuning on public labels.

Candidate methods:
5. Dense + BM25 hybrid (RRF, k=60), top-50..100 shortlist.
6. Passage re-templating (category path / brand / title first, strip low-signal fields),
   max length sweep 128/256/384/512; instruction-prefixed queries; query construction
   ablation (all turns / user-only / last turn).
7. Bi-encoder fine-tune: MNRL (CachedMNRL for large batch), low LR, 1-3 epochs, optional
   LoRA / weight-averaging with base. Negatives: in-batch only vs BM25 hard vs dense hard.
8. False-negative handling: positive-aware mining (`mine_hard_negatives` with
   relative_margin / range_min), judge-filtered negatives (run Qwen2.5-1.5B with the eval
   prompt over mined negatives, drop accepted), GISTEmbed, MarginMSE distillation from a
   cross-encoder teacher. The survey's "skip ranks 1-4" rule is a heuristic, treat as one
   sweep point, not a default.
9. Synthetic train-style conversations for train-pool products only (never the 25k
   public-only products), generated on H200s; must be described in report.
10. Cross-encoder rerank on top-20..50: bge-reranker-v2-m3 and Qwen3-Reranker-0.6B,
    zero-shot vs fine-tuned (surveys cite conflicting independent results for these two).
11. Qwen2.5-1.5B pointwise P(yes) rerank with the judge prompt on top-10..20; also as a
    label source for cross-encoder training.
12. Query-side: HyDE / conversation rewrite with Qwen2.5-1.5B (about 1k generations).
13. ESCI external data (Substitute labels as graded positives) for reranker pretraining.

From the Perplexity survey (not in the other two):
14. Extra baselines: `mixedbread-ai/mxbai-embed-large-v1` (dense, 512 tokens) and
    `cross-encoder/ms-marco-MiniLM-L6-v2` (22.7M) as the cheap reranker, fine-tuned;
    first rerank-depth sweep on MiniLM before trying 0.6B rerankers.
15. Staged negative schedule: in-batch only first, then mine BM25 + base-dense top-50,
    filter with a reranker margin / duplicate-title check, then 1-2 hard negatives per
    query; re-mine at most once. Prefer same-category negatives.
16. Qwen2.5-1.5B as offline teacher only (keep high-scoring candidates out of the
    negatives, distill medium scores into the reranker). Conflicts with item 11 (online
    judge-prompt rerank); test both, the surveys disagree.
17. Selective chunking only for overlength passages with max-over-chunks aggregation;
    late chunking / ColBERT (`lightonai/colbertv2.0`, `jinaai/jina-colbert-v2`) as
    ablations only.
18. Report extras: rank-transition table (where reranking moves the first relevant item),
    error taxonomy on 10-20 sampled public errors, per-category plots.
Selection rule from that survey, adopt unless contradicted: choose by public judge-graded
MRR@10, then Recall@10, unseen-target MRR@10, and measured 3070 latency; if reranking does
not improve graded MRR@10 over RRF, ship RRF only.

Needs a ruling before use:
- BLaIR (`hyp1231/blair-*`) and Amazon-C4 are pretrained/built on Amazon Reviews 2023,
  likely the same corpus as the banned `Abner0803/Amazon-Beauty-S2`. Ask the TA before
  using; if used, disclose and keep it as an ablation, not the submission.
- Shipping precomputed passage embeddings in `download.sh`: private pool has 25k items
  we can't precompute anyway, so design for runtime encoding regardless.

Unverified in surveys (check on the HF model card / arXiv page before relying on them):
jina-embeddings-v5 (license CC BY-NC), Nemotron-3-Embed-1B, Querit-Reranker-A0.4B,
KaLM-Reranker, Wizard of Shopping data availability, arXiv 2601.16492, 2411.11767.
Gemini's "late chunking" description is actually chunk-and-mean-pool, not late chunking.

## Open questions / risks

- `Abner0803/Amazon-Beauty-S2` ban — likely the source dataset this benchmark was built
  from. Don't use it directly; if a strategy wants external data, needs to be clearly
  something else and clearly described per the rules.
- 8GB VRAM execution ceiling is tighter than HW1 — any reranker/embedding model choice
  needs to fit comfortably alongside the fixed `Qwen2.5-1.5B-Instruct` judge if both run
  in the same pass (check whether `evaluate.py`'s judge and our own retrieval model need
  to coexist in VRAM, or run as separate processes/stages).
- Public-set retrieval 1hr budget — need to benchmark early whether a reranking stage
  (cross-encoder over candidates) is affordable at 75k pool / 1k queries, or whether it
  has to be restricted to a bi-encoder top-N shortlist before any reranking.
