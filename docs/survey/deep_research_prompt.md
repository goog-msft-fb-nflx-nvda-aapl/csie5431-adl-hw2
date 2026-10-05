# Deep research prompt — ADL HW2 (RAG product retrieval) survey

Paste this whole file into Claude (Opus, for a research task like this), Gemini Deep
Research, or Perplexity. Goal: a grounded candidate list of methods, models, and
datasets to try, plus a concrete ablation/analysis/visualization plan — not a generic
RAG tutorial. Everything needed to understand the task is below; don't assume outside
knowledge of "ADL HW2" itself.

## Terminology (so answers map cleanly onto what we actually have)

- **Query**: a conversation's turns (user+assistant, always ending on a user turn,
  3 or 5 turns) concatenated into one text block. This is what gets embedded/searched —
  there is no separate "rewritten query" step provided; if query rewriting/expansion is
  worth doing, that's part of what we're asking about.
- **Product pool / passages**: `{parent_asin: "title\ndescription"}`. Description is
  free text (title, category path, dimensions, brand, etc. all flattened into prose),
  mean ~1,720 chars, up to ~6,500.
- **Gold target**: the one product ID each conversation is labeled with.
- **Graded relevance / "acceptable alternative"**: at eval time, every non-gold product
  in a retrieved top-10 gets judged by a fixed small LLM (`Qwen2.5-1.5B-Instruct`,
  deterministic/greedy decode) for whether it's *also* a valid recommendation for that
  conversation (binary accept/reject, gain 1 or 0; gold itself is always gain 1). This
  means **retrieval doesn't have to hit the exact labeled ID** — near-duplicate/
  equivalent products count too, if the judge agrees.
- **Recall@10 / MRR@10**: computed over the graded gains above (best gain in top 10 for
  Recall; best gain/rank for MRR), not plain exact-ID-match recall/MRR — the usual IR
  definitions still apply, just with "relevant" redefined by the judge rather than being
  purely the gold label.
- **Train / public test / private test**: train (4,830 queries, 50,000-item pool) is the
  only split with labels we can train on. Public test (1,000 queries, 75,000-item pool =
  train's 50,000 + 25,000 new) has labels too (for our own dev use) but is also what the
  TA grades us on directly. Private test (2,000 queries, 100,000-item pool = public's
  75,000 + another 25,000 new) is never seen by us at all.

## Complete problem statement

Build `retrieve_and_rerank(queries, passages, top_k=10) -> list[{"qid", "ranked_pids"}]`
— the only function we're allowed to implement. Given it's one function with full
freedom inside, "the system" can be anything: BM25, a dense bi-encoder, a cross-encoder
reranker on top of a first-stage retriever, hybrid fusion, an LLM reranker, multi-stage
pipelines, etc. Constraints:

- **Deploy hardware**: RTX 3070, **8GB VRAM**, 32GB RAM, 20GB disk, Ubuntu 22.04. (We
  can train on much bigger GPUs — 4x H200 NVL available — but the *final* packaged
  model/pipeline must run inference within this budget.)
- **Time budget**: public-set retrieval (1,000 queries against a 75,000-item pool) must
  finish within **1 hour** on that hardware.
- **`download.sh`** fetches our trained model + any external training data, must finish
  in 30 min at 10MB/s (so bounds total download size), and cannot `pip install` anything
  — the test environment's packages are fixed to whatever's declared, no runtime setup.
- **Cannot** use `Abner0803/Amazon-Beauty-S2` (HF dataset) — almost certainly the source
  this benchmark's splits were built from, so likely especially easy to "cheat" with if
  used directly; any other external data is allowed but must be described in the report.
- **Cannot** fine-tune on test data or seek private test data.
- `Qwen2.5-1.5B-Instruct` is already fixed/available as the eval-time judge — it's a
  legitimate, already-"blessed" model if useful as a reranker or generation component too
  (the homework's second report question is a separate generation-analysis task using
  this exact model over our retrieved specs).

### Data facts that should shape method choice (full detail: see attached/linked EDA)

- **Product pool is general-domain, not beauty-specific**: despite a `beauty-*`
  dialogue-ID prefix, the pool splits roughly evenly across 6 top-level Amazon
  categories (Sports & Outdoors, Pet Supplies, Toys & Games, Home & Kitchen, Beauty,
  Beauty & Personal Care, each ~6k-10k items in the 50k train pool), and gold targets
  are spread across all 6, not concentrated in Beauty. Public-test gold targets are
  almost exactly stratified ~200 each across the 5 main categories.
- **Half of public-test gold targets are products never in the train pool at all**
  (500/1,000 — the 25,000 "new" items added for public test). This is the single
  biggest thing to optimize for: whatever retrieval/reranking approach is chosen has to
  generalize from product *text content*, not learn fixed per-ID embeddings/identities.
  Private test adds another 25,000 unseen-at-public-test-time items on top.
- Product descriptions embed structured info (dimensions, brand, category path, best-
  sellers rank, etc.) as unstructured prose within the free-text description, not as
  separate fields — worth asking whether light structure-extraction/templating before
  embedding helps over raw concatenation.
- Train has 4,830 queries but only 4,547 unique gold targets (283 products targeted by
  exactly 2 queries, small effect, probably not worth special handling).
- Queries are short (154-842 chars, mean 411) relative to product text (24-6,516 chars,
  mean 1,720) — an asymmetric-length retrieval setup (short query, long passage).

## What to research (please be concrete — named papers, named HF model IDs/repos, named
GitHub repos, not just technique categories)

1. **SOTA retrieval architectures for short-conversational-query -> long-product-passage
   retrieval at this scale (50k-100k candidates).** Bi-encoder (dense dual-encoder) vs.
   late-interaction (ColBERT-style) vs. sparse (BM25/SPLADE) vs. hybrid fusion — what's
   actually state of the art for *product search specifically* (not generic open-domain
   QA retrieval), and what's realistic to fine-tune/run within an 8GB-VRAM inference
   budget? Include specific benchmarks if relevant (Amazon ESCI, Amazon-C4/product
   search leaderboards, MS MARCO passage, BEIR) and what architectures win there.

2. **Named embedding models on HuggingFace** worth starting from for this task —
   general-purpose (not beauty-specific, per the EDA finding above) sentence/text
   embedding models in a size range that's fine-tunable and fits comfortably alongside
   inference-time memory for reranking/judging. Concretely name candidates (e.g. the
   BGE/E5/GTE/Qwen3-Embedding families, Sentence-Transformers model zoo, etc. — but
   verify actual current recommendations, don't assume a particular one is still SOTA)
   with their embedding dims, max sequence length, and approximate VRAM footprint.

3. **Reranking strategies that fit the budget.** Cross-encoder reranking over a
   first-stage shortlist (how deep a shortlist is affordable within the 1hr/1,000-query
   budget at 8GB VRAM?), vs. using `Qwen2.5-1.5B-Instruct` itself as an LLM reranker
   (pointwise/listwise), vs. skipping reranking and tuning the bi-encoder harder. What
   does recent literature say about where the recall gain from reranking actually comes
   from, and when it's not worth the latency cost at this candidate-pool scale.

4. **Training-data construction for the bi-encoder/reranker, given we only have one gold
   label per query (no labeled negatives, no labeled "also acceptable" alternatives at
   train time).** Standard contrastive-learning setups: in-batch negatives, hard-negative
   mining (via BM25 miss-but-topically-close, or via a first untrained embedding model's
   false positives), and whether synthetic/LLM-generated hard negatives are worth it
   given the "acceptable alternative" scoring means some apparent negatives might
   actually be correct. How do other systems handle label noise from this specific
   "there may be multiple valid products" structure?

5. **Generalizing to never-seen-at-train products specifically** (the 50%-of-public-
   test-targets-are-new fact above) — this is close to a zero-shot/cold-start dense
   retrieval problem. What does the literature recommend for retrieval systems that must
   generalize to catalog items added after training, as opposed to closed-set retrieval?

6. **Loss functions** typically used for this kind of asymmetric dense retrieval
   fine-tuning (InfoNCE/contrastive, triplet, multiple-negatives-ranking-loss as used in
   Sentence-Transformers, etc.) and any reported sensitivity to negative-sampling
   strategy interacting with loss choice.

7. **Chunking/handling long product descriptions** (up to 6,500 chars) within an
   embedding model's max sequence length — truncation vs. chunk-and-pool vs. late
   chunking vs. summarization-before-embedding, and what's known to actually help vs.
   just adding latency.

8. **What ablations are standard/expected for a report like this** (we need to show "how
   many strategies did you try, why is the submitted one best" with hyperparameters and
   loss specified) — e.g. context-turns-included-in-query vs. last-turn-only, embedding
   model choice, negative-mining strategy, rerank-or-not, rerank-depth, chunking
   strategy, seen-vs-unseen-product generalization gap as its own reported ablation.

9. **What visualizations/analyses would make the report's quantitative story clearest**
   given the above — e.g. Recall@k across k (not just k=10), performance split by
   seen-vs-never-seen-at-train gold target, performance split by product category,
   latency/VRAM budget breakdown per pipeline stage, embedding similarity distribution
   for gold vs. hard-negative vs. random-negative pairs.

## Constraints on the search itself

- Prefer concrete, named, currently-available resources (HF model/dataset IDs, GitHub
  repos, paper titles+arXiv IDs) over generic advice. We'll verify licensing, VRAM
  footprint, and real-world results ourselves before adopting anything — flag what's
  unverified.
- Don't restrict suggestions to "what a typical ADL course lecture would cover" — as
  long as a technique isn't prohibited by the rules above (no banned dataset, no test-
  data fine-tuning, no private-test access), it's in scope to explore, including
  anything genuinely state-of-the-art that goes beyond lecture material.
- If multiple reasonable architectures exist with real tradeoffs (e.g. bi-encoder-only
  vs. bi-encoder+cross-encoder-rerank), present the tradeoff explicitly (expected
  recall/MRR gain vs. latency/VRAM cost at our specific scale) rather than picking one
  without justification.
