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

## 2026-10-06 — BLaIR zero-shot (exp 2, ablation only)

Same setup as exp 1 (public, exact gold-ID, fp16, max_seq_len 512, passages = title\ndescription,
all-turn query). BLaIR loaded via sentence-transformers Transformer + CLS pooling, L2-normalized,
per its model card. `hyp1231/blair-roberta-{base,large}`. Contamination caveat: BLaIR is pretrained
on Amazon Reviews 2023, which very likely overlaps this benchmark's products (unverified); treat
as a disclosed ablation, not a submission candidate. No TA ruling requested.

| method | R@1 | R@10 | R@50 | R@100 | MRR@10 | R@10 seen / unseen |
|---|---|---|---|---|---|---|
| blair-base dense | 0.186 | 0.487 | 0.706 | 0.792 | 0.273 | 0.474 / 0.500 |
| blair-base + bm25 | 0.394 | 0.745 | 0.920 | 0.947 | 0.499 | 0.752 / 0.738 |
| blair-large dense | 0.083 | 0.229 | 0.382 | 0.435 | 0.127 | 0.202 / 0.256 |
| blair-large + bm25 | 0.202 | 0.544 | 0.875 | 0.918 | 0.290 | 0.548 / 0.540 |

Pool encode on H200: base 77s / 0.87GB, large 125s / 1.48GB peak.

Measured: BLaIR-base dense (R@10 0.487) is below zero-shot bge-base (0.556) and qwen3-0.6b
(0.621) and close to bge-m3 (0.473); BLaIR-large dense is far lower (0.229) and its hybrid is
below BM25 alone (0.765). No advantage from Amazon-domain pretraining shows up in this
zero-shot setting. The large-model result being worse than base is unexplained; fp16 or
pooling/format mismatch not ruled out, not investigated. Decision: drop BLaIR from the
candidate list unless fine-tuning comparisons later give a reason to revisit.

## 2026-10-06 — Test-env packages (backlog item 2) + first end-to-end baseline

Item 2: `docs/spec/spec.md` says Python 3.12.9, recommended packages (transformers 4.56.1,
torch 2.8.0, faiss-gpu-cu12 1.12.0, sentence-transformers 5.1.0, accelerate, huggingface_hub),
and that other packages may be used if listed in `requirements.txt`. No vLLM or bm25s in the
recommended set, so I did not plan around them. The BM25 here uses scikit-learn + scipy (scikit-learn
added to `requirements.txt`; scipy comes with sentence-transformers). Whether the graders
actually `pip install` extras from our `requirements.txt` is not stated in the spec, unverified.
GPU env `adl_hw2_env` is Python 3.10, not 3.12.9.

Baseline v0: `retrieve_and_rerank()` in `retrieval_utils.py` = BM25 + zero-shot
Qwen3-Embedding-0.6B (fp16, 512 tokens, query prompt), RRF k=60, top 10. Running through
`provided/retrieval.py` then `provided/evaluate.py --batch-size 8` on public (gsm-gpu2 GPU2,
tmux `eval1`, log `/home/jtan/adl_hw2/eval1.log`).

Results (public, 1,000 queries, 75k pool, official `evaluate.py`, judge Qwen2.5-1.5B bf16):
graded Recall@10 0.877, graded MRR@10 0.646 (9,238 judge comparisons, 0 parse failures).
Same ranked lists scored by exact gold ID (exp 1, hybrid-qwen3-0.6b): R@10 0.762, MRR@10 0.537.
Timing on H200 (not the 3070): `retrieval.py` end to end 523s (BM25 + model load + pool encode
+ query encode + fusion, as one process); `evaluate.py` judge pass 328s at batch 8, peak CUDA
allocated 4.65GiB / reserved 8.39GiB for the judge alone. Results JSON:
`/home/jtan/adl_hw2/results/eval_bm25_qwen3.json`. Against the spec's public thresholds
(Simple 0.860 / 0.650, Strong 0.885 / 0.680, Boss 0.950 / 0.800 for R@10 / MRR@10): R@10 0.877 is
above Simple and below Strong; MRR@10 0.646 is below Simple (0.650) by 0.004. All four numbers
(public and private) are gated together per tier, so no tier is cleared yet on public. Private
metrics unknown. Not yet checked: where the 3070 lands vs the 1hr budget.

## 2026-10-06 — Reranking over the zero-shot hybrid shortlist (exp 3)

Setup: `code/cands.py` caches the top-200 BM25+Qwen3-0.6B RRF candidates per split
(public hybrid exact recall@10/30/50/100/200 = 0.762/0.892/0.917/0.948/0.967; train pool
0.632/0.747/0.792/0.839/0.879). `code/rerank_eval.py` scores the top-N candidates with a
reranker, re-sorts them, keeps the rest in RRF order. Docs truncated to 1800 chars, all-turn
query. Exact-ID = our metric, graded = official `evaluate.py` (judge Qwen2.5-1.5B). H200 GPU1.

| reranker (zero-shot) | depth | exact R@10 | exact MRR@10 | graded R@10 | graded MRR@10 | score time (pairs) |
|---|---|---|---|---|---|---|
| none (RRF hybrid) | - | 0.762 | 0.530 | 0.877 | 0.646 | - |
| ms-marco-MiniLM-L6-v2 | 30 | 0.574 | 0.296 | 0.740 | 0.410 | 40s (50k, depth 50) |
| bge-reranker-v2-m3 | 30 | 0.653 | 0.383 | 0.791 | 0.494 | 81s (50k) |
| Qwen2.5-1.5B judge-style P(true), no reference | 20 | 0.701 | 0.380 | 0.867 | 0.562 | 158s (20k) |
| Qwen3-Reranker-0.6B (yes/no logit) | 30 | 0.873 | 0.717 | 0.938 | 0.803 | 385s (50k, depth 50) |
| Qwen3-Reranker-0.6B | 50 | 0.886 | 0.723 | not run | not run | 385s |

Qwen3-Reranker-0.6B exact-ID by depth: d10 0.762/0.662, d20 0.837/0.700, d30 0.873/0.717,
d50 0.886/0.723 (R@10/MRR@10). Seen vs unseen gold at d30: R@10 0.884 / 0.862, MRR@10 0.715 / 0.720.
MiniLM, bge-reranker-v2-m3 and the Qwen2.5 scorer all lowered both metrics versus no reranking
(those lists were re-sorted from depth 30/30/20, per-depth numbers in
`/home/jtan/adl_hw2/results/rerank.jsonl`). The Qwen2.5 prompt here is mine, not `evaluate.py`'s
(no reference product available at inference), so it is one prompt variant, not the final word on
Qwen2.5 as a reranker. Loading Qwen3-Reranker-0.6B through sentence-transformers 5.1.0
`CrossEncoder` gave a randomly initialized score head (warning at load), so the numbers above use
the HF `AutoModelForCausalLM` yes/no-logit recipe from the model card; an earlier
sentence-transformers attempt crashed and was discarded. Timing is H200, not 3070: 385s for
50k pairs at max_len 1024; the 3070 figure is unmeasured and is the main open risk for this
option.

Graded depth-30 Qwen3-Reranker-0.6B result is above the Boss public MRR@10 threshold (0.800) and
below its R@10 threshold (0.950); above Strong (0.885 / 0.680) and Simple on both public numbers.
Private metrics unknown.

## 2026-10-06 — Bi-encoder fine-tune, in-batch negatives (exp 4a)

`code/ft_biencoder.py`: Qwen3-Embedding-0.6B, CachedMNRL (scale 20, batch 128, mini-batch 32),
lr 1e-5, 2 epochs (68 steps), bf16, NO_DUPLICATES sampler, 10% of train targets held out as a
product-disjoint dev split (index file saved, dev numbers not computed yet), train rows 4,347
dialogues (some held-out) with the gold as the only positive, no hard negatives. Public exact-ID,
512 tokens, fine-tuned weights `models/ftA_inbatch` on gsm-gpu2:

| method | R@1 | R@10 | R@50 | R@100 | MRR@10 | R@10 seen / unseen |
|---|---|---|---|---|---|---|
| zero-shot qwen3-0.6b dense (exp 1) | 0.281 | 0.621 | 0.814 | 0.870 | 0.381 | 0.616 / 0.626 |
| ftA dense | 0.563 | 0.859 | 0.946 | 0.968 | 0.666 | 0.848 / 0.870 |
| ftA + bm25 RRF | 0.623 | 0.875 | 0.963 | 0.973 | 0.702 | 0.882 / 0.868 |

Dense R@10 0.621 -> 0.859 from fine-tuning alone. Seen/unseen gold R@10 differ by 1-3 points.

## 2026-10-06 — Bi-encoder fine-tune with 1 hard negative (exp 4b), Q2 generation draft (exp 5)

4b: same recipe as 4a plus 1 mined hard negative per query (hybrid zero-shot rank 5..50, excluding
gold and same-title products, no judge filtering), weights `models/ftB_hn1`. Public exact-ID:
dense R@10 0.878 / MRR@10 0.694 (was 0.859 / 0.666 for 4a), hybrid with BM25 R@10 0.891 / MRR@10
0.725 (4a hybrid 0.875 / 0.702); hybrid R@100 0.981. R@10 seen / unseen (hybrid) 0.898 / 0.884.
Single seed each, so the 4a -> 4b difference (about 1.6 points R@10) is not yet checked for noise.
ftC (3 negatives), ftD (lr 2e-5, 3 epochs), ftE (negatives from ranks 0..30) are queued.

Q2 (`code/q2_generate.py`, Qwen2.5-1.5B-Instruct bf16 greedy, max 220 new tokens, top-3 retrieved
specs of the Qwen3-Reranker d30 list, 700 chars/spec, 8 random public dialogues x 3 prompts):
plain / grounded (one product, facts only, <80 words) / checklist (requirements bullets then
recommendation, "not confirmed" line). Outputs `results/q2.jsonl` on gsm-gpu2. Crude counts over 8
samples each: mean words 112 / 56 / 83; response names the top-1 title 8 / 7 / 5 of 8; contains a URL
1 / 0 / 0; markdown bold 5 / 3 / 1; a number not in the specs 0 / 0 / 1. Gold was retrieved at rank 1
for 6 of 8 dialogues (rank 3 and rank 2 for the others, so the response recommends the rank-1
non-gold product, e.g. a Wild Kratts suit for a hand-puppet request). Qualitative: the plain prompt
invented a link and claimed a "soft case design" the spec does not state; grounded produced short
copy that stayed on the title and on spec-stated features; checklist listed requirements usefully
but sometimes dropped the title ("[1] because ...") and wrote "None of the requirements are
explicitly stated" under Not confirmed, i.e. the model did not do the verification the prompt asked
for. These are 8 samples and counts of surface features, not a hallucination rate; no human or judge
scoring has been done yet.

## 2026-10-06 — Qwen3-Reranker-0.6B cost knobs (exp 6), zero-shot candidates, exact-ID

Depth 30 over the zero-shot hybrid shortlist, `--doc-chars` truncation (R@10 / MRR@10, H200, bs 32,
before the memory fix below): 600 chars 0.822 / 0.622 (127s, 30k pairs); 1000 chars 0.859 / 0.697
(164s); 1400 chars 0.867 / 0.717 (201s); 1800 chars 0.873 / 0.717 (earlier run). Depth by 1800 chars:
d10 0.762 / 0.662, d20 0.837 / 0.700, d30 0.873 / 0.717, d50 0.886 / 0.723.

Memory: my first scorer ran the full LM head over every position, peak 9.4-13.4GB allocated on the
H200 at bs 32 (would not fit 8GB). Changed `score_qwen3rr` (and the training script) to take the last
hidden state and multiply by only the yes/no rows of `lm_head`: peak 3.33GB allocated at bs 16,
30k pairs / 1800 chars in 166s on the H200 (scores match the old path to ~3 decimals, depth-10
MRR@10 0.663 vs 0.662). 3070 time still unmeasured.

## 2026-10-06 — Fine-tuned bi-encoder + zero-shot Qwen3-Reranker, graded (exp 7)

Pipeline: BM25 + `ftB_hn1` Qwen3-Embedding-0.6B (RRF, k=60) top-100 -> zero-shot Qwen3-Reranker-0.6B on
top 30 (1800 chars/doc, bs 16) -> top 10. Public, H200 GPU0.

| stage | exact R@10 | exact MRR@10 | graded R@10 | graded MRR@10 |
|---|---|---|---|---|
| BM25 + zero-shot qwen3 RRF (exp "baseline v0") | 0.762 | 0.530 | 0.877 | 0.646 |
| BM25 + zero-shot qwen3 RRF -> qwen3 reranker d30 | 0.873 | 0.717 | 0.938 | 0.803 |
| BM25 + ftB RRF | 0.891 | 0.719 | not run | not run |
| BM25 + ftB RRF -> qwen3 reranker d30 | 0.920 | 0.745 | 0.960 | 0.824 |

ftB hybrid exact recall@10/30/50/100/200: 0.891 / 0.953 / 0.965 / 0.981 / 0.981. Reranker scoring 170s
for 30k pairs, peak 3.34GB (H200). The graded 0.960 / 0.824 is above the spec's public Boss
thresholds (0.950 / 0.800) on both public numbers; private numbers are not measurable. Exact R@10 by
seen / unseen gold at d30: 0.912 / 0.928; MRR@10 0.733 / 0.758. These are single runs, no seed
variance measured, and the judge-graded number was measured once.

## 2026-10-06 — Judge labels on train candidates (exp 8, supervision for the reranker)

`code/judge_labels.py`: the official `evaluate.LLMJudge` (gold as reference) over the top-20 zero-shot
hybrid candidates of all 4,830 train dialogues, 93,197 judgments, 0 parse failures. Judge accepted
8.1% of non-gold candidates; 54.3% of dialogues have at least one judge-accepted non-gold product in
their top 20. Saved `cache/judge_train.npy` (1 accept, 0 reject, 2 gold). Reranker fine-tuning
(`code/ft_reranker.py`: listwise softmax CE, 1 positive + 7 judge-rejected negatives, lr 1e-5,
1 epoch, 4,344 train dialogues, product-disjoint 10% dev held out) is running for gold-only
positives (R1) and gold-or-accepted-alternative positives at probability 0.5 (R2).

## 2026-10-06 — Reranker fine-tune R1 (gold positive, judge-rejected negatives) and more bi-encoder variants (exp 9)

Bi-encoder variants (public exact-ID; Qwen3-Embedding-0.6B, CachedMNRL, batch 128; all single seed):

| run | settings | dense R@10 / MRR@10 | + BM25 RRF R@10 / MRR@10 |
|---|---|---|---|
| ftA | in-batch only, lr 1e-5, 2 ep | 0.859 / 0.666 | 0.875 / 0.702 |
| ftB | 1 hard neg (rank 5-50), lr 1e-5, 2 ep | 0.878 / 0.694 | 0.891 / 0.725 |
| ftC | 3 hard negs (rank 5-50), lr 1e-5, 2 ep | 0.885 / 0.713 | 0.892 / 0.737 |
| ftD | 1 hard neg, lr 2e-5, 3 ep | 0.912 / 0.760 | 0.907 / 0.750 |

For ftD the BM25 fusion is below dense alone at R@10 and MRR@10 (R@100 0.985 dense, 0.987 hybrid).
Hyperparameters are being chosen by these public numbers (no separate dev numbers computed yet), so
treat the public metrics as optimistic for the chosen configs; the product-disjoint dev split from
`ft_biencoder.py` has not been scored.

Reranker R1 (`code/ft_reranker.py`, Qwen3-Reranker-0.6B, full fine-tune bf16, listwise softmax CE over
1 gold + 7 judge-rejected candidates from train zero-shot top-20, lr 1e-5, 1 epoch, 1,086 steps of 4
dialogues, ~15 min on a shared H200): public exact-ID with depth-30 rerank:

| first stage | zero-shot reranker R@10 / MRR@10 | R1 reranker R@10 / MRR@10 |
|---|---|---|
| zero-shot hybrid | 0.873 / 0.717 | 0.886 / 0.805 |
| ftB hybrid | 0.920 / 0.745 | 0.938 / 0.835 |

R1 over ftB hybrid depth 20 / 10: 0.924 / 0.828, 0.891 / 0.818. Judge-graded (official `evaluate.py`)
for ftB hybrid + R1 d30: graded R@10 0.975, graded MRR@10 0.883.

## 2026-10-06 — Reranker R2, ftE (exp 10)

ftE (negatives from hybrid ranks 0..30, 1 per query, lr 1e-5, 2 ep): dense R@10 0.876 / MRR@10 0.699,
hybrid 0.895 / 0.732 (ftB with ranks 5..50: 0.878 / 0.694, 0.891 / 0.725); no clear difference at this
single-seed resolution.

Reranker R2 (as R1 but the positive in each group is a random judge-accepted alternative with
probability 0.5, otherwise the gold; everything else identical). Public exact-ID, depth 30:
zero-shot hybrid shortlist 0.887 / 0.813 (R1 0.886 / 0.805); ftB shortlist 0.938 / 0.843 (R1 0.938 /
0.835), depth 20 0.922 / 0.840, depth 10 0.891 / 0.822. Judge-graded on ftB shortlist d30: R@10 0.972,
MRR@10 0.890 (R1: 0.975 / 0.883). R1 vs R2 differ by under 1 point on every number, single seed each.

## 2026-10-06 — End-to-end through provided/retrieval.py + evaluate.py (exp 11)

Final-style `retrieval_utils.py` (dense top-200, optional BM25 top-200 RRF k=60, rerank top 30 with
the fine-tuned reranker, 1800 chars/doc, dense bs 32, rerank bs 16, fp16). Dense = `ftD_hn1_lr2e5_e3`,
reranker = `rrR2_alt50`. Public 1,000 queries / 75k pool, official judge. H200 shared with other jobs:

| BM25 in fusion | retrieval.py wall time | graded R@10 | graded MRR@10 |
|---|---|---|---|
| no (dense only) | 391s (dense stage 235s, rerank stage 155s) | 0.972 | 0.881 |
| yes | 412s | 0.977 | 0.891 |

ftF (3 hard negatives, lr 2e-5, 3 ep): dense R@10 0.920 / MRR@10 0.772 (ftD 0.912 / 0.760).
Reranker R3 (accepted alternative as the positive in every group): ftB shortlist d30 exact 0.936 /
0.829, zero-shot shortlist 0.885 / 0.799, i.e. below R1 and R2 on MRR@10.

## 2026-10-06 — Length / depth cost knobs end to end, more variants (exp 12)

ftD dense at inference length (public exact-ID R@10 / MRR@10 / R@100; pool encode seconds on a shared
H200): 256 tokens 0.894 / 0.731 / 0.981 (153s); 384: 0.907 / 0.756 / 0.985 (210s); 512: 0.912 / 0.760 /
0.985. Token stats of public passages with the Qwen3 tokenizer (3,000 sampled): mean 433, median 427,
p90 682; mean tokens after truncation to 512/384/256 = 391/333/240.

End to end via `retrieval.py` + `evaluate.py` (ftD dense + BM25 RRF + R2 reranker; graded):
dense 256 tokens, rerank depth 20, 1200 chars/doc: R@10 0.970, MRR@10 0.880, retrieval 226s
(dense stage 126s, bm25 +8s, rerank 90s). For comparison full setting (512 / depth 30 / 1800 chars):
0.977 / 0.891 in 412s. Costs were measured on a shared H200, not on the 3070.

More variants: ftG (3 hard negatives from ranks 0..20 with judge-accepted ones removed, lr 2e-5, 3 ep)
dense R@10 0.910 / MRR@10 0.762, no better than ftD / ftF (0.912 / 0.760, 0.920 / 0.772).
Reranker R4 (R2 with 2 epochs): ftB shortlist d30 exact 0.936 / 0.847 (R2 0.938 / 0.843).

## 2026-10-06 — Wall-clock guard in retrieval_utils.py (exp 13)

Risk: the 3070 is slower than the H200 by an unmeasured factor and the 1 hour limit is hard. Added
to `retrieve_and_rerank()`: (1) before encoding the pool it times a 1,000-passage sample at the
longest length, projects the whole pool and lowers the max token length (512 -> 384 -> 256 -> 192 ->
128) until the projection fits `DENSE_SHARE` (45%) of `HW2_BUDGET` (default 3000s); (2) the reranker
works 16 queries at a time and recomputes, from its measured pairs/s and the time left (budget minus
300s safety), how many shortlist items per query it can still afford, falling back to the first-stage
order when out of time. On the H200 with the default budget the guard changes nothing.

Tests (ftD dense + BM25 + R2 reranker, public, official judge, shared H200):
| setting | dense max_len chosen | rerank depth | retrieval time | peak VRAM | graded R@10 | graded MRR@10 |
|---|---|---|---|---|---|---|
| HW2_BUDGET=500 (simulated slow machine) | 256 | 3-5 per query (4,376 pairs) | 202s | 4.07GB | 0.950 | 0.855 |
| budget 3000, length cap 384 | 384 | 30 (30,000 pairs) | 402s | 4.44GB | 0.979 | 0.895 |

The 500s run shows the guard produces a valid ranking inside the budget with reduced depth; it does
not predict what a real 3070 would pick. The 3070 itself is still unmeasured.
FLOP-based estimate (interpretation, not measured): dense 0.44B non-embedding params ~0.9 GFLOP/token,
75k passages at 384 tokens (mean 333 after truncation) ~22 PFLOP; rerank 30k pairs of ~650 tokens ~18
PFLOP; at 15-25 TFLOPS achieved on a 3070 this is 25-45 minutes total, so the guard matters.

## 2026-10-06 — Q2 generation analysis on the final retrieval lists (exp 14)

`code/q2_generate.py` on `results/e2e_final_default.json` (final pipeline), 10 public dialogues
(seed 1) x 3 prompts, top-3 retrieved specs, greedy, Qwen2.5-1.5B-Instruct. Outputs
`results/q2_final.jsonl` (earlier 8-sample run on an older list: `results/q2.jsonl`). Gold was the
rank-1 retrieved item in 9 of 10 dialogues (rank 2 for the Old Spice deodorant request, where rank 1 was
Speed Stick). Surface counts over 10 samples per prompt (plain / grounded / checklist): mean words
96 / 56 / 69; names the rank-1 title 8 / 8 / 7; invents a URL 1 / 1 / 0; "Not confirmed: None ..." non-answer
0 / 0 / 4. The checklist "numbers not in specs" count (4) is inflated by its own numbered requirement
lists, not by hallucinated facts, so it is not a hallucination rate.

Observed cases (all from `q2_final.jsonl`):
- beauty-000138 (no-lamp French nails): plain ignored the rank-1 acrylic powder and recommended a
  different candidate (Makartt nail tips) and described its sizes, i.e. it did not follow the best-first
  order; grounded named the right product but wrote "last up to two weeks", a duration the user asked
  for and the spec was not checked to support; checklist recommended the right product.
- beauty-000065 (Merkur razor, deodorant stick; gold at rank 2): all three prompts recommended the
  rank-1 Speed Stick; grounded fabricated a link and a product variant ("Regular 3 oz (Pack of 2)",
  not in the specs); checklist listed requirements the user never stated ("Cleanser", "Aluminum-free").
- toys-000068 (shaped St. Patrick's puzzle): checklist listed "Standard rectangular shape" although the
  user asked for a shaped puzzle, then still recommended the shaped puzzle; plain and grounded were
  correct and grounded added the spec-stated "ages 15 and up".
- pet-supplies-000183 and home-and-kitchen-000062 (floppy dog toy; tablecloth clips): all three prompts
  produced accurate, spec-supported answers; plain was longest and added unsupported colour/size
  assurances ("medium-sized dog", "smooth edges protect scratches" from the spec) in some sentences.
- sports-000180 (hidden handgun mount): checklist's "Not confirmed" line claimed the spec does not
  mention discreet mounting or fast access although the user's discreet requirement is exactly what
  the product title says ("Concealed Holder"), i.e. the verification step was wrong.
- pet-supplies-000108 (whitefish cat food): plain added a fabricated product URL and "trusted since 2017".

Interpretation (not measured beyond these 10 samples): the 1.5B model follows the "recommend the first
candidate and use only the spec" instruction best under the short grounded prompt; the explicit
checklist prompt produced structured output but the "not confirmed" step was either empty or wrong in
most samples, and added requirements the user did not state; the plain prompt gave the longest answers
and the most unsupported details. No human or judge rating of faithfulness was done.

BM25 ablation in the final configuration (dense_final + reranker_final, both trained on all 4,830 train
dialogues, `retrieval.py` + `evaluate.py`, public): with BM25 fusion graded R@10 0.977 / MRR@10 0.887
(448s); dense only 0.969 / 0.873 (448s). dense_final = ftF recipe (3 hard negatives, lr 2e-5, 3 ep),
reranker_final = R2 recipe (alt-prob 0.5, lr 1e-5, 1 ep). Peak VRAM 3.45GB.

## 2026-10-06 — Seeds and larger sweeps (exp 15)

Seed noise (public exact-ID, seed also changes the 10% held-out dev split and negative sampling):
- ftF recipe (3 hard negatives, lr 2e-5, 3 ep), dense R@10 / MRR@10, seeds 0/1/2: 0.920 / 0.772,
  0.919 / 0.772, 0.916 / 0.774. Hybrid with BM25: 0.912 / 0.763, 0.913 / 0.765, 0.912 / 0.761.
- R2 reranker recipe over the ftB shortlist at depth 30, R@10 / MRR@10, seeds 0/1/2: 0.938 / 0.843,
  0.940 / 0.835, 0.936 / 0.836 (mean 0.938 / 0.838).
So on these runs the seed-to-seed spread is about 0.004 R@10 and 0.008 MRR@10.

More bi-encoder settings (all 1 seed): ftH (3 negs, lr 3e-5, 4 ep) dense 0.928 / 0.792, hybrid 0.920 /
0.776; ftI (3 negs, lr 5e-5, 5 ep) dense 0.919 / 0.780, hybrid 0.922 / 0.780; ftJ (7 negs, lr 3e-5,
4 ep) dense 0.932 / 0.797, hybrid 0.920 / 0.780. In these three the BM25 fusion is not above dense alone
at R@10 / MRR@10 but is at R@100 (0.993 vs 0.987-0.991).

More reranker settings, ftB shortlist depth 30 exact (zero-shot-shortlist numbers in parentheses):
R4 (2 epochs) 0.936 / 0.847 (0.886 / 0.820); R5 (lr 2e-5, group 16 = 1 + 15 negatives, 2 queries per step,
1 epoch) 0.940 / 0.857 (0.887 / 0.827); R6 (lr 5e-6) 0.936 / 0.820 (0.885 / 0.794). R5 is the best exact
MRR@10 so far (+0.019 over the R2 mean, about two seed spreads).

Next: retrain with the ftJ recipe (dense) and R5 recipe (reranker) on all train dialogues
(`models/dense_final2`, `models/reranker_final2`), run end to end through the official scripts.
