I'll verify the current state of the key model families before writing this up, since embedding/reranker recommendations go stale quickly.I ran a handful of searches to check that the model and paper names below are still current. This is a working survey with a recommended pipeline and an experiment plan. Items I couldn't verify are marked **[unverified]**.

## The two facts that drive everything

**1. Passage encoding time, not VRAM, is the real limit.** 75k passages × ~1,720 chars is roughly 30M+ tokens to encode inside the one-hour public run. With Qwen3-Embedding-0.6B in fp16 on a 3070, a rough guess is a few thousand to ~10k tokens/s. That means full-length encoding of the pool could use most or all of the hour before any reranking. Two things to do early:

- **Benchmark encoder throughput on the real 3070 in week one.** That number decides the passage truncation length and how deep you can rerank.
- **Ask the TAs whether shipping cached passage embeddings in `download.sh` is allowed.** It's a gray area under "no test data use", and the private pool has 25k items you can't precompute anyway. Design so runtime encoding fits the budget.

**2. The metric rewards "a plausible right-type product at rank 1", not "the gold ID at rank 1".** Recall@10 and MRR@10 use the *best gain* in the list, and any judge-accepted alternative gets gain 1. So the ideal top-1 is whatever item Qwen2.5-1.5B is most likely to accept, which may not be the gold. This has three consequences:

- Getting the product type, category, and constraints right matters more than fine-grained identity.
- Diversifying the top-10 is unnecessary.
- Reranking with the judge's own prompt optimizes the metric directly.

Since `evaluate.py` is available locally, report exact-ID recall and judge-graded recall side by side. The gap between them is a finding in itself.

## 1. Architecture for this setting

The closest published benchmark is **Amazon-C4** from the BLaIR paper (Hou et al., arXiv 2403.03952). Its queries are longer and more detailed than keyword search and not always directly relevant to item metadata, including multi-round dialogues and complex user instructions, which is essentially your query style. On conventional product search (ESCI), BLaIR significantly outperformed other dense retrievers and sparse baselines. The broader takeaway from that line of work is that dense bi-encoders beat BM25 on long, intent-heavy queries.

Lexical matching still helps for brands, sizes, and model numbers. A **dense + BM25 hybrid with Reciprocal Rank Fusion** is cheap: BM25 over 75k docs takes seconds on CPU with `bm25s` or `rank_bm25`, if one is in your fixed environment.

**ColBERT-style late interaction** (ColBERTv2; `answerdotai/answerai-colbert-small-v1` via PyLate) is a poor first stage here. 75k docs × ~400 tokens × 128 dims is several GB uncompressed. It's reasonable as a mid-stage reranker, but adds engineering for an uncertain gain over a cross-encoder.

**SPLADE** (`naver/splade-v3`) is a possible third hybrid leg, but it costs about as much as dense encoding. It's the lowest-priority option.

**Recommended pipeline:**

1. Fine-tuned bi-encoder + BM25, fused with RRF, producing a top-50 to top-100 shortlist.
2. A fine-tuned cross-encoder on the top 20–50.
3. Optionally, Qwen2.5-1.5B pointwise scoring of the top 10–20, using the judge prompt.

Each stage should earn its place through an ablation.

## 2. Embedding models

Qwen3-Embedding is still the reference open family. It comes in 0.6B, 4B, and 8B sizes; the 0.6B outputs up to 1024 dims (MRL-truncatable down to 32) with 32k context. It's Apache-2.0, and sources from mid-to-late 2026 still list Qwen3-Embedding as the strongest open option.

| Model | Params | Dim | Max len | fp16 weights | Role |
|---|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 1024 (MRL) | 32k | ~1.2 GB | Main candidate if throughput allows ~256–384 tokens |
| `Qwen/Qwen3-Embedding-4B` / `-8B` | 4B / 8B | 2560 / 4096 | 32k | 8 / 16 GB | Training-time teacher for hard-negative mining and distillation (H200s), not for deployment |
| `BAAI/bge-base-en-v1.5` | 110M | 768 | 512 | ~0.2 GB | Fast baseline, ~5× cheaper per token |
| `Alibaba-NLP/gte-modernbert-base` | 149M | 768 | 8k | ~0.3 GB | Fast, long-context; good mid-size option **[verify current status]** |
| `Snowflake/snowflake-arctic-embed-m-v2.0`, `nomic-ai/nomic-embed-text-v1.5` | ~110–300M | 768 | 8k | <1 GB | Alternatives at the same tier **[unverified rankings]** |
| `BAAI/bge-m3` | 568M | 1024 | 8k | ~1.1 GB | Dense + sparse + multi-vector in one model |
| `hyp1231/blair-roberta-base` / `-large` | 125M / 355M | 768 / 1024 | 512 | small | Product-domain pretrained; **see the contamination warning below** |

**Contamination warning on BLaIR and Amazon-C4.** BLaIR was pretrained on Amazon Reviews 2023, 570M reviews over 48M items in 33 categories. Your benchmark's `parent_asin` IDs and the banned `Abner0803/Amazon-Beauty-S2` almost certainly come from that same corpus. I couldn't find the banned dataset's card to confirm. So BLaIR may have seen your test products' metadata and reviews during pretraining. It isn't literally banned, but it violates the rule's spirit. Get TA sign-off before using it, and if you do, disclose it and treat it as an ablation, not the submission. The same applies to Amazon-C4 as training data.

## 3. Reranking under the budget

Do the arithmetic per stage. A 0.6B cross-encoder at depth 50 with ~450-token pairs is about 1,000 × 50 × 450 ≈ 22M tokens, similar to encoding the whole pool. **Realistic depths are therefore 20–50 for a 0.6B reranker and 10–20 for Qwen2.5-1.5B.** Truncate passages harder at rerank time; the title and category path carry most of the signal.

Reranker evidence is mixed, so test rather than trust benchmarks:

- Qwen's own table, reranking Qwen3-Embedding-0.6B top-100, shows Qwen3-Reranker-0.6B (65.80) ahead of BGE-reranker-v2-m3 (57.03) on their main retrieval column.
- An independent 2026 BEIR evaluation found the opposite: Qwen3-Reranker-0.6B averaged 48.09 versus 56.42 for bge-reranker-v2-m3, which puts Qwen3-Reranker-0.6B *below* its first-stage retriever.
- Another 2026 benchmark again ranks Qwen3-Reranker-0.6B at 68.03 versus 63.07 for bge-reranker-v2-m3.

The lesson is that off-the-shelf rerankers can hurt as easily as help. Both models are worth fine-tuning on your data.

**Using Qwen2.5-1.5B with the judge's prompt** as a pointwise scorer (take P("yes"), or the accept-token logit) is the most metric-aligned option. It's essentially a direct proxy for the gain function. Its weakness is that it can't tell the gold from acceptable alternatives, but the metric doesn't reward that distinction either.

Two ways to use it:

- **Final reorder:** run it only on the top 10–20 to promote the item most likely to be accepted to rank 1. This mainly lifts MRR.
- **Training labels:** use its accept/reject decisions as labels when training the cross-encoder.

Check whether vLLM is in the fixed environment, because plain HF `generate` will be much slower.

**Where reranking gains come from:** mostly from fixing ordering inside a decent shortlist, not from recall beyond it. Jacob et al., "Drowning in Documents" (arXiv 2411.11767) **[from memory, verify]**, report that rerankers can *degrade* when scoring very deep candidate lists. That supports a shallow rerank over a strong first stage. Plot Recall@k of the first stage. If Recall@50 is barely above Recall@10, reranking has little to work with, and the effort belongs in the bi-encoder.

## 4. Training data with one positive and hidden alternatives

The core risk is false negatives. Many "negatives" are acceptable alternatives, which the judge would score as gain 1 at eval.

**Mine positive-aware hard negatives.** NV-Retriever (Moreira et al., arXiv 2407.15831) uses the positive's relevance score as an anchor to filter out false negatives. NV-Embed's version caps negative scores at 95% of the positive score (TopK-PercPos). Sentence-Transformers' `mine_hard_negatives` (with `relative_margin` and `range_min`, skipping the top few ranks) implements this. Use Qwen3-Embedding-8B as the mining teacher on the H200s.

**Know why strong-retriever negatives can backfire.** Prior work found that taking top-ranked non-positives from a stronger retriever made the trained model worse, because those results contain more unlabeled positives.

**Use the judge as a negative filter.** This is specific to your setup. Run Qwen2.5-1.5B with the eval prompt over each mined negative:

- Drop negatives it accepts.
- Optionally recycle them as soft or extra positives.

This aligns training directly with the eval definition of relevance. It costs some H200 hours and is very worth it.

**Two smaller fixes:**

- **GISTEmbedLoss** (arXiv 2402.16829) uses a guide model to drop in-batch negatives that look like positives.
- Avoid putting duplicate-title variants (the 65 groups found in the EDA) in the same batch, since they'd act as false negatives against each other.

**Synthetic queries are your biggest data lever.** You have 4,830 labeled pairs but 45k unlabeled *train-pool* products. Generate train-style 3- or 5-turn conversations for them with a large LLM on the H200s, in the style of InPars (arXiv 2202.05144), Promptagator (arXiv 2209.11755), or GPL (arXiv 2112.07577). Do **not** generate queries for the 25k public-only products; that's training on test passages. Describe the synthetic data in the report.

**External data:** the Amazon **Shopping Queries / ESCI** dataset (Reddy et al., arXiv 2206.06588; `amazon-science/esci-data`). Its queries are short keyword queries, but its Exact/Substitute/Complement/Irrelevant labels map neatly onto "acceptable alternative". "Substitute" pairs are good graded positives for reranker pretraining.

## 5. Generalizing to unseen products

The good news is that a text bi-encoder with no ID embeddings is already inductive. The real risks are subtler:

- **Overfitting to train-pool items** after repeated exposure as positives. Diagnose this by checking what fraction of retrieved top-10 items are train-pool items, against the base rate of 50k/75k.
- **Light fine-tuning can't fix catastrophic drift.** Keep learning rates low (1e-5 to 2e-5), train for 1–3 epochs, and consider LoRA, or weight-averaging the fine-tuned model with the base model (model soup).
- **Your validation split won't measure cold-start unless it's product-disjoint.** Build an internal dev split whose gold products are excluded from training as both positives and negatives. This mimics the public "new item" half without touching public labels for tuning.

Treat public test as a final check, not a tuning target. The private test is 2× larger with even more unseen items.

## 6. Loss functions

- **Bi-encoder:** MultipleNegativesRankingLoss (InfoNCE with in-batch plus mined negatives; scale ~20–50, i.e. temperature 0.02–0.05). Use **CachedMNRL** (GradCache, arXiv 2101.06983) for effective batch sizes of 256–1024 on the H200s. Add GISTEmbed as the false-negative-aware variant. MatryoshkaLoss is optional if you want to trade dimension for speed.
- **Reranker:** listwise softmax cross-entropy over (1 positive + 7–15 negatives), versus pointwise BCE. Better still, **distill** with MarginMSE (Hofstätter et al., arXiv 2010.02666) or KL from a teacher such as Qwen3-Reranker-8B, or from judge probabilities.
- **Sensitivity:** larger batches and harder negatives raise false-negative damage, especially under InfoNCE. The negative-filtering threshold is a hyperparameter to sweep, and reporting that interaction is a good ablation.

## 7. Long descriptions

Mean ~430 tokens with a max of ~1,600 means most passages fit in 512 tokens. Late chunking (Günther et al., arXiv 2409.04701) and chunk-and-pool are therefore unlikely to pay off.

Higher-value moves:

- **Re-template passages** so title, category path, and brand come first.
- **Strip low-signal fields** such as best-sellers rank, shipping weight, and raw dimensions, then truncate (ablate 128/256/384/512 tokens).
- **Optionally embed the title separately** and fuse its score with the description score.

Summarization before embedding with Qwen2.5-1.5B over 75k items is far too slow for the runtime budget.

On the query side, try HyDE (arXiv 2212.10496): have Qwen2.5-1.5B write a hypothetical product description from the conversation. It costs about 1,000 short generations and tends to help zero-shot more than after fine-tuning.

## 8. Ablation plan

| Factor | Levels |
|---|---|
| Query construction | all turns / user turns only / last user turn / weighted recency; with and without instruction prefix |
| Passage formatting | raw / re-templated; max length 128 / 256 / 384 / 512 |
| Base encoder | bge-base / gte-modernbert-base / Qwen3-Emb-0.6B (zero-shot and fine-tuned) |
| First stage | dense / BM25 / RRF hybrid |
| Negatives | in-batch only / BM25 hard / dense hard / dense hard + judge filter |
| Loss | MNRL vs. CachedMNRL (batch size sweep) vs. GISTEmbed |
| Training data | real only / + synthetic train-pool queries / + ESCI |
| Reranker | none / bge-reranker-v2-m3 / Qwen3-Reranker-0.6B / Qwen2.5-1.5B judge-prompt; each zero-shot vs. fine-tuned |
| Rerank depth | 10 / 20 / 50 / 100 against wall-clock time |
| HyDE | off / on |

Report every row on both the cold-start dev split and public, split into seen vs. unseen gold. Include hyperparameters and wall-clock time.

## 9. Visualizations

- **Recall@k curves** (k = 1 to 100) per first-stage variant. This justifies rerank depth.
- **Seen vs. unseen gold** as grouped bars for each system. This is your headline generalization result.
- **Per-category** Recall@10 and MRR@10 across the 6 categories.
- **Gold hit vs. alternative hit,** as stacked bars of top-10 hits that were the exact gold versus judge-accepted alternatives.
- **Latency and VRAM per stage** (encode pool, BM25, ANN search, rerank, judge) against the 60-minute line, as a cost/quality Pareto plot.
- **Cosine-similarity histograms** for gold, judge-accepted negatives, judge-rejected hard negatives, and random negatives, before and after fine-tuning.
- **Pool-membership bias:** the fraction of retrieved items from the train pool versus the 2/3 base rate.
- **Error gallery:** 5–10 failure cases annotated as wrong category, right type but wrong constraint, or alternatives rejected by the judge.

This is detailed enough to keep, so I can turn it into a Doc you can annotate as the experiments come in. A broader background sweep could add verified 2026 throughput numbers on RTX 3070-class GPUs, newer small embedders and rerankers released since mid-2026, and recent product-search papers on conversational queries and false-negative handling.

Here's what the follow-up sweep turned up. The main change is that **my earlier encoding-time estimate was too pessimistic**. That gives you more room for a bigger encoder or deeper reranking.

## Revised compute budget

**Encoding the passage pool.** Runpod's September 2026 benchmark measured bge-m3, a 568M model close in size to Qwen3-Embedding-0.6B. On 512-token passages it embedded about 190 passages/s on an RTX A5000 and about 265/s on an A6000 using TEI. I estimate a 3070 at roughly 0.7–0.8× an A5000 **[my estimate, not measured]**. That gives:

- About 130–150 passages/s at full 512 tokens, so roughly 9–10 minutes for the 75k pool.
- Probably 15–25 minutes with plain transformers, which is likely slower than TEI or vLLM.
- Roughly a third more time for the private 100k pool.

Earlier I said this stage could eat most of the hour. It's more like a quarter to a third of it.

**Two caveats on those numbers:**

- **The serving engine matters a lot.** Qwen3-Embedding-0.6B ran 2.6× faster on vLLM than on TEI in the same test. You can't `pip install`, so you'll be on whatever is in the fixed environment, most likely plain `transformers`/`sentence-transformers`. The benchmark explicitly didn't measure that setup. Use fp16, SDPA attention, and length-sorted batching to close some of the gap.
- **Measure on a real 3070 before committing to anything.** That's still the first task.

**Reranking.** On an RTX A6000 with bf16 and flash-attention, Qwen3-Reranker-0.6B scored about 100 query–document pairs per second. The 4B version scored about 24 pairs/s. Scaled to a 3070 at roughly 60 pairs/s:

| Rerank depth | Pairs (1,000 queries) | Approx. time |
|---|---|---|
| 30 | 30k | ~8 min |
| 50 | 50k | ~14 min |

A 4B reranker is out of budget.

**Revised end-to-end plan:** about 20 minutes to encode the pool, plus about 10 minutes of reranking at depth 30–40, plus an optional Qwen2.5-1.5B pass on the top 10. That fits in an hour with headroom for the private pool.

## New embedders since mid-2025

| Model | Size | Notes | Caveats |
|---|---|---|---|
| `jinaai/jina-embeddings-v5-text-small` | 677M | Built on Qwen3-0.6B-Base, distilled from Qwen3-Embedding-4B. Reports 71.7 on MTEB English vs. 70.5 for Qwen3-Embedding-0.6B. 1024 dims (Matryoshka), 32k context. | Licensed **CC BY-NC 4.0**, fine for coursework but note it in the report. Uses task-specific LoRA adapters (load the `-retrieval` variant), so fine-tuning is a bit more involved. Needs `Query:`/`Document:` prefixes. |
| `jinaai/jina-embeddings-v5-text-nano` | 239M | Built on EuroBERT-210M, also distilled from Qwen3-Embedding-4B. 71.0 MTEB English. | Same license. About 2.5× cheaper than the 0.6B models, so it's a strong fast-baseline candidate. |
| `nvidia/Nemotron-3-Embed-1B-BF16` | ~1–2B | Released July 2026, 2048-dim output, OpenMDW-1.1 license. Pruned and distilled from a Ministral-3 parent. | The parameter count is ambiguous in the docs. The Ministral-3 architecture probably needs a recent `transformers` version, so check the fixed environment. Slower than the 0.6B models. |
| `google/embeddinggemma-300m` | 300M | Compact multilingual model. | Not benchmarked against the others here **[unverified for your task]**. |

Qwen3-Embedding-0.6B is still the safe default: Apache-2.0, well supported, and easy to fine-tune with sentence-transformers. The jina v5 models are the most interesting challengers at the same speed. Run a zero-shot comparison of the two on your cold-start dev split before choosing which one to fine-tune.

## New rerankers

- **`Querit/Querit-Reranker-A0.4B`** (arXiv 2606.19037, June 2026) is an in-house MoE model with 0.4B active parameters. Reranking Qwen3-Embedding-0.6B's top-100, it raised BEIR nDCG@10 from 54.11 to 59.28. Its training recipe also fits your situation well: synthetic-query mining with teacher scores as soft labels, then checkpoint merging. The custom MoE backbone probably needs `trust_remote_code` **[check environment compatibility]**.
- **KaLM-Reranker-V1** comes in Nano, Small, and Large sizes (0.27B, 1B, and 4B active). It encodes passages separately from queries, which makes it faster, and reports BEIR results on par with Qwen3-Reranker. The Nano size might allow deeper reranking **[not evaluated by me]**.
- **jina-reranker-v3.5** is a listwise reranker that I saw listed but didn't investigate.

The reranker picture hasn't changed: independent benchmarks still disagree about Qwen3-Reranker-0.6B versus bge-reranker-v2-m3. Fine-tune two of them and compare.

## Product-search and conversational-retrieval papers

- **Wizard of Shopping** (Li et al., ACL 2025, arXiv 2502.00969). This is the closest external data to your task. It contains 3.6k LLM-simulated, target-oriented shopping conversations across three domains, grounded in a catalog of over 236k products. The arXiv version says the data would be released upon acceptance, so I haven't confirmed it's actually downloadable **[verify]**. Its catalog is Amazon data, so check whether its products overlap with yours before training on it.
- **"LLM-based Semantic Search for Conversational Queries in E-commerce"** (arXiv 2601.16492, January 2026). It fine-tunes a Sentence Transformer for queries and products, generates about 10 diverse queries per product, and prepends structured attributes to product text. That's a direct precedent for your plans for synthetic train-pool queries and passage re-templating.
- **The broader conversational-search literature** splits into two approaches: rewriting the conversation into a standalone query, or training a dense retriever to encode the whole conversation (Mo et al., arXiv 2506.10635). Your concatenate-all-turns setup is the second approach. A rewrite baseline using Qwen2.5-1.5B (in the spirit of ChatRetriever and CHIQ) is a natural ablation partner for HyDE.

## Net changes to the plan

1. **Allow passage length up to 384–512 tokens** instead of 256, subject to your 3070 measurement.
2. **Add jina-embeddings-v5-text-small and -nano** to the encoder shortlist alongside Qwen3-Embedding-0.6B.
3. **Plan rerank depth around 30–40 pairs per query**, and add Querit-A0.4B as a third reranker candidate if it loads in your environment.
4. **Cite arXiv 2601.16492 and Wizard of Shopping** in the report's related work as the nearest precedents.

The "Start research" button on my earlier reply is still there if you want a fuller background sweep; it runs from that button rather than from a chat message.