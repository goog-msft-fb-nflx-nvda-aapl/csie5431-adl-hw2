For this homework, the strongest *practical* direction is a **hybrid two-stage retriever**: a fine-tuned general-purpose dense bi-encoder for broad candidate recall, fused with BM25, followed by a compact cross-encoder reranking roughly 50–100 candidates. Do **not** optimize around catalog-ID memorization or a beauty-only model: half of public targets are products absent from training, and the pool spans six broad product domains. [ppl-ai-file-upload.s3.amazonaws](user-uploaded attachment)

## Recommended submission architecture

| Stage | Recommended implementation | Why it fits this benchmark | Deployment implications |
|---|---|---|---|
| Corpus normalization | Build a consistent product text template: `Title`, `Category`, `Brand`, then description/specification text; preserve numbers, units, colors, sizes, and compatibility terms | Shopping relevance often turns on exact constraints—size, material, fit, brand, category—not just topical similarity | Store normalized text plus raw text. No extra model cost |
| Sparse first stage | BM25 over the normalized product text | Lexical retrieval is especially valuable for exact model names, dimensions, quantities, ingredients, color names, and product-type wording | CPU-friendly; index 75k–100k items easily |
| Dense first stage | Fine-tuned `Qwen/Qwen3-Embedding-0.6B` *or*, if package/runtime compatibility is safer, a Sentence-Transformers-sized baseline such as `mixedbread-ai/mxbai-embed-large-v1` | Dense retrieval captures conversational intent and paraphrase mismatch between multi-turn dialogue and catalog text | Precompute corpus vectors offline; only encode 1,000–2,000 queries at inference |
| Fusion | Reciprocal Rank Fusion (RRF) of BM25 and dense rankings, usually retrieve 100–200 candidates from each | Robust when dense and lexical rankings make different errors; avoids brittle score calibration | Negligible cost |
| Reranking | Fine-tune `cross-encoder/ms-marco-MiniLM-L6-v2` or run a targeted comparison against `BAAI/bge-reranker-v2-m3` / `Qwen/Qwen3-Reranker-0.6B` | Cross-encoders jointly read the dialogue and product candidate, handling fine constraints that bi-encoders blur | Rerank 50 first; test 20/50/100. MiniLM is the safest 8GB baseline |
| Final output | Return top 10 after reranking; preserve the RRF order as a tie-breaker | Aligns with graded Recall@10 and MRR@10 | Fits the required function |

The core reason to prefer this over bi-encoder-only is that your queries contain **dialogue state** and product constraints, while each product has a relatively long, noisy description. A cross-encoder can decide whether a candidate satisfies the whole request rather than merely sharing semantic vocabulary. Amazon’s Shopping Queries / ESCI benchmark makes the same conceptual distinction: it labels product results as Exact, Substitute, Complement, or Irrelevant, explicitly recognizing that product search has graded, substitute-like relevance rather than a single exact-ID notion. [github](https://github.com/amazon-science/esci-data)

Your evaluator is even more favorable to this approach: it gives gain 1 to valid non-gold alternatives. Consequently, aim for *constraint-satisfying product neighborhoods*, not an overfitted single-label classifier.

## Models to evaluate

### Dense retrieval shortlist

| Model | Parameters / dimensions | Context | Recommendation for this homework |
|---|---:|---:|---|
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B; up to 1,024 dimensions, configurable output dimension | 32k tokens | **Best modern candidate to test.** It accommodates the complete product text without aggressive truncation and can reduce output dimension for a smaller FAISS index. Its 0.6B parameters make it materially more plausible than 4B/8B alternatives on 8GB VRAM.  [huggingface](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| `BAAI/bge-m3` | 1,024 dimensions | 8,192 tokens | **Best multi-representation experiment.** It supports dense, sparse, and ColBERT-like representations under one model family, with up to 8,192 input tokens. It is valuable for an ablation, but its deployment and indexing complexity make it less attractive than a simple dense+BM25 pipeline unless it clearly wins.  [huggingface](https://huggingface.co/BAAI/bge-m3) |
| `mixedbread-ai/mxbai-embed-large-v1` | 1,024 native dimensions; supports reduced output dimensions | 512 tokens | **Strong lightweight baseline.** It is English-focused and retrieval-oriented, but requires truncation/chunking for long product text. Test it with title/spec-first formatting rather than blindly embedding the whole description.  [mixedbread](https://www.mixedbread.com/docs/models/embedding/mxbai-embed-large-v1) |
| `intfloat/e5-mistral-7b-instruct` | 7B; 4,096-dimensional embeddings | up to 4,096 tokens recommended | Useful only as an offline teacher or high-GPU training/dev baseline—not as your final 8GB deployment model. Its size is fundamentally hostile to your inference requirement.  [huggingface](https://huggingface.co/intfloat/e5-mistral-7b-instruct) |
| `Qwen/Qwen3-Embedding-4B` / `8B` | 4B / 8B; 2,560 / 4,096 dimensions | 32k | Do not make these the submission default. They can be valuable as teacher models for distillation or offline pseudo-labeling, but 8GB deployment, query latency, package size, and concurrent reranking leave little safety margin.  [huggingface](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |

### Reranker shortlist

| Model | Size | Practical role |
|---|---:|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22.7M parameters | **Primary production reranker.** Very cheap, easy to fine-tune, and appropriate for 50–100 candidates per query. It was trained for MS MARCO passage ranking, so it gives a defensible IR-specific baseline.  [huggingface](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) |
| `BAAI/bge-reranker-v2-m3` | 0.6B parameters | **Quality-oriented experiment.** It supports long inputs, but its authors recommend `max_length=1024` because that was the fine-tuning length. Given 1,000 public queries and a one-hour allowance, reranking 50 candidates is plausible but must be benchmarked on the actual RTX 3070.  [huggingface](https://huggingface.co/BAAI/bge-reranker-v2-m3/discussions/9) |
| `Qwen/Qwen3-Reranker-0.6B` | 0.6B parameters | **High-upside current candidate.** It shares the Qwen3 embedding family’s 32k context design. Compare it at 50 candidates only after MiniLM establishes a latency baseline.  [huggingface](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| `Qwen2.5-1.5B-Instruct` | fixed available evaluator model | Use as an optional **offline teacher**, not default online reranker | Prompt/listwise scoring is expensive, nondeterministic unless carefully constrained, and produces a train–test mismatch if your prompt format differs from the evaluator. The course-approved availability makes it legal, but not automatically the best system design. |

### Exact model choice

Start with this decision sequence:

1. **Baseline:** BM25 + `mxbai-embed-large-v1`, RRF, MiniLM-L6 cross-encoder reranking top 50.
2. **Likely best final:** Replace the dense encoder with fine-tuned `Qwen3-Embedding-0.6B`, keep BM25/RRF, rerank top 50 using the best latency-feasible reranker.
3. **Research/high-upside branch:** `bge-m3` dense+sparse or dense+BM25, with either MiniLM or BGE reranking.
4. Choose by **public graded MRR@10 first**, then graded Recall@10, unseen-target MRR@10, and measured RTX-3070 end-to-end latency—not by train retrieval loss or exact-ID recall alone.

Avoid using Qwen3 4B/8B as the final default because their model size is disproportionate to a 75k–100k-candidate product search task and leaves an unnecessarily fragile 8GB deployment path. Qwen3’s smaller 0.6B embedding model already provides a 32k context window and up to 1,024-dimensional vectors. [huggingface](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

## Training design

### Split correctly

Do **not** report only your 4,830-example train performance. Construct a development protocol that directly tests catalog generalization:

- Use a **product-disjoint validation split**: split on `target parent_asin`, not merely dialogue ID.
- Prefer a 80/20 target split, with all dialogues for a repeated target assigned to the same partition.
- Create a stronger cold-start validation condition by withholding a subset of products from the retrieval corpus during model fitting, then adding them back only for evaluation.
- Report public results separately for:
  - products that belong to the train pool;
  - the 25k public-only products;
  - each top-level category;
  - all public queries combined.

This matters because exactly 500 of 1,000 public queries target the added 25,000 catalog items, and the private pool introduces another 25,000 new items. A system that learns product identity rather than semantic matching can look strong in random splits while failing where the benchmark is designed to be difficult. [ppl-ai-file-upload.s3.amazonaws](user-uploaded attachment)

### Dense retriever objective

Use a contrastive objective with in-batch negatives:

\[
\mathcal{L}_{\mathrm{InfoNCE}}
=
-\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(s(q_i,p_i^+)/\tau)}
{\sum_{j=1}^{B}\exp(s(q_i,p_j)/\tau)}
\]

where \(q_i\) is one dialogue, \(p_i^+\) is its labeled product, and all other products in the batch act as negatives.

Use `CachedMultipleNegativesRankingLoss` when possible. It preserves the in-batch-negative/InfoNCE objective while using gradient caching to support substantially larger effective batches without proportional VRAM growth; Sentence Transformers documents it as a preferred option over conventional MNRL when batch size is constrained. [sbert](https://sbert.net/docs/sentence_transformer/loss_overview.html)

Suggested starting parameters:

| Parameter | Starting value | Ablation values |
|---|---:|---|
| Query max length | 512 tokens | 256, 384, 512 |
| Product max length | 768–1,024 tokens for 512-context encoders; full text for Qwen3/BGE-M3 | 256, 512, 768, 1,024 |
| Effective batch size | 128–512 | 64, 128, 256, 512 |
| Temperature / scale | model default first | Tune only after negative-mining baseline |
| Epochs | 1–3 | Early stop by product-disjoint validation MRR@10 |
| Learning rate | \(1\times10^{-5}\) to \(3\times10^{-5}\) for full fine-tuning | Lower for 0.6B models; use LoRA if needed |
| Hard negatives/query | 1–4 | 0, 1, 2, 4 |
| Dense retrieval depth | 100–200 | 50, 100, 200, 500 |

### Hard-negative policy

Your one-positive-label setting makes hard negatives necessary—but dangerous. Some topically similar products could be valid alternatives under the Qwen evaluator, so treating every nearest neighbor as an explicit negative may suppress the exact alternatives that earn credit.

Use a **conservative, multi-source** negative pool:

- Mine BM25 top-ranked candidates that are not the labeled positive.
- Mine dense top-ranked candidates using the current or base encoder.
- Prefer candidates in the same category/product type, since cross-category negatives are usually too easy.
- Exclude exact duplicate titles and obvious variants where possible.
- Filter out candidates that score too similarly to the positive under a strong cross-encoder or Qwen teacher; retain “hard but plausibly non-relevant,” not “possibly another excellent recommendation.”
- Mix hard negatives with in-batch negatives; do not make all negatives hard from the outset.

The Sentence Transformers MS MARCO hard-negative procedure follows this same broad logic: retrieve candidate negatives using lexical/dense retrieval and use a cross-encoder margin so highly scored candidates are not blindly passed as negatives.  Their training materials also emphasize that cross-encoder strength depends heavily on hard negatives. [github](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/training/ms_marco/README.md)

A sound schedule:

1. Epoch 0–1: positive pairs + large in-batch negatives only.
2. Mine BM25 and base-dense top-50 candidates.
3. Filter potential false negatives using a reranker margin and lexical duplicate checks.
4. Fine-tune another 1–2 epochs with 1–2 explicit hard negatives/query plus in-batch negatives.
5. Re-mine once only if public/dev metrics improve. With only 4,830 dialogues, repeatedly mining can overfit to artifacts of the train catalog.

### Reranker training

Build each reranker example as:

```text
Query:
[all dialogue turns, role-tagged]

Candidate product:
Title: ...
Category: ...
Brand: ...
Description/specifications: ...
```

Train with the labeled product as positive and 3–8 mined candidates as negatives. Use pairwise/listwise contrastive ranking rather than independent binary labels where available. The operational target is not calibrated relevance probability; it is that the positive or another valid recommendation moves into the highest ranks.

For MiniLM:

- Use a total input cap of 512 tokens initially.
- Allocate roughly 160–220 tokens to dialogue and the remainder to product text.
- Preserve title, category, brand, and structured specs before generic narrative description.
- Rerank 50 candidates first. Only evaluate 100 after measuring actual GPU throughput.

For 0.6B rerankers:

- Quantize only if it is allowed by the supplied runtime stack and does not add package dependency risk.
- Test `max_length=512` and 1,024.
- A 1,024-token reranker input is reasonable for product descriptions, but full 6,500-character text across 50–100 candidates is a throughput trap. `bge-reranker-v2-m3` technically supports 8,192 tokens, but its authors recommend 1,024 because that is its fine-tuning length. [huggingface](https://huggingface.co/BAAI/bge-reranker-v2-m3/discussions/9)

## Long descriptions and dialogue construction

### Query representation

Do **not** default to final-user-turn-only. The final turn averages only 84 characters and often depends on the preceding assistant/user constraints; all turns concatenated average 411 characters. Start with the full role-labeled conversation. [ppl-ai-file-upload.s3.amazonaws](user-uploaded attachment)

Test these variants:

| Query variant | Expected value |
|---|---|
| Full role-tagged dialogue | Primary choice; retains constraints accumulated over 3 or 5 turns |
| User turns only | Strong ablation; drops assistant wording that may be generic or distractive |
| Final user turn only | Low-cost baseline; likely loses earlier preferences and constraints |
| Query rewrite/summary generated offline | Research variant; may clarify constraints but risks hallucinating or dropping exact product terms |

A simple deterministic representation is preferable initially:

```text
User: ...
Assistant: ...
User: ...
Assistant: ...
User: ...
```

Role tags are useful because the assistant turns may contain suggestions or clarifying paraphrases, while user turns state requirements.

### Product representation

Because the metadata is flattened into prose, use structured serialization rather than treating the description as an undifferentiated blob:

```text
Product title: {title}
Product category: {parsed category path if recoverable}
Brand: {brand if recoverable}
Key specifications: {dimensions, weight, material, size, color, compatibility}
Description: {remaining text}
```

Do not use an LLM extraction pipeline as the first submission. Regex/light parsing for stable markers such as `Brand`, `Dimensions`, `Color`, `Material`, units, and category path is easier to reproduce and much lower risk. Then compare raw vs. templated product text experimentally.

### Chunking decision

For your data, many product texts will fit within a 512–1,024-token window; only the long tail needs special treatment. Therefore, **do not chunk all 50k–100k products by default.** It multiplies index size, increases retrieval ambiguity, and makes parent-product aggregation necessary.

Recommended hierarchy:

1. **Full-document encoding** with Qwen3-Embedding-0.6B or BGE-M3, if measured deployment memory/time supports it.
2. **Head-preserving truncation**: title + category + brand + specs + first body segment, using 512–1,024 tokens.
3. **Selective chunking only for overlength documents**, perhaps 2–4 chunks with parent-level `max` aggregation:
   \[
   s(q,p)=\max_{c \in \text{chunks}(p)} \cos(e_q,e_c)
   \]
4. Compare against mean/max pooling and title-only/head-only variants.

Late chunking is a credible research ablation, not a necessary default. It encodes the long document before pooling segments, so each chunk has full-document contextualization; the method reports improved retrieval across several tasks and can be implemented without extra training using long-context embedding models. But it increases implementation complexity and storage, and gains depend on the model/dataset. [arxiv](https://arxiv.org/html/2409.04701v3)

Given your product texts top out at roughly 6,500 characters—not tokens—and average only ~1,720 characters, the likely ranking is:

- full text in a long-context dense encoder;
- well-formatted 512–1,024-token head/spec truncation;
- selective chunking;
- sophisticated late chunking.

Treat that ranking as a hypothesis to test, not a claim to report without measurements.

## Architecture tradeoffs

### Bi-encoder only

**Advantages**
- Lowest query-time cost.
- Precompute all catalog embeddings.
- Easiest way to scale to the 100k private catalog.
- Strongest fit for unseen items because product text directly determines the vector.

**Weaknesses**
- Dot-product similarity can miss conjunctive constraints: “fragrance-free, travel-size, safe for a specific material.”
- Long descriptions dilute critical fields.
- Hard-negative supervision is essential.

Use this as your essential baseline and possibly final fallback if reranking provides only marginal MRR improvement.

### Sparse + dense hybrid

**Advantages**
- BM25 handles exact entities, units, titles, brands, and alphanumeric strings.
- Dense models handle paraphrases and conversational intent.
- RRF is robust and avoids fragile cross-model score normalization.
- Particularly appropriate for semi-structured product metadata.

**Weaknesses**
- Two indexes and tuning knobs.
- BM25 can over-weight noisy metadata unless the text template is controlled.

This is the default recommendation.

### Late interaction / ColBERT

ColBERT retains token-level representations and scores via MaxSim, approximately:

\[
s(q,d)=\sum_{i=1}^{|q|}\max_{j=1}^{|d|} q_i^\top d_j.
\]

That retains fine-grained matching while allowing precomputed document representations. The original ColBERT work framed it as a way to get much of cross-encoder effectiveness with substantially cheaper retrieval; ColBERTv2 added cross-encoder distillation, hard-negative mining, and residual compression to reduce index footprint by roughly 5–8×. [arxiv](https://arxiv.org/abs/2004.12832)

**For this assignment:** it is a legitimate high-upside experiment, especially because exact specifications and multi-constraint dialogue matter. However, it is **not** the lowest-risk final system:

- A 100k-product token index can be much larger than dense vectors.
- Long descriptions require token-length control and aggressive indexing decisions.
- Package/runtime integration is more failure-prone than FAISS + BM25 + a standard cross-encoder.
- The public/private time rule makes a benchmarked end-to-end implementation mandatory.

Use `lightonai/colbertv2.0` or `jinaai/jina-colbert-v2` only as an explicit architecture ablation, not your first working pipeline. Jina-ColBERT-v2 is designed for long-context and multilingual late interaction, and reports that reducing token-vector dimensions from 128 to 64 can cut storage by up to 50% with little performance loss in its evaluation. [arxiv](https://arxiv.org/html/2408.16672v2)

### LLM reranking

A Qwen2.5-1.5B listwise or pointwise reranker is tempting because the evaluator itself uses that model for acceptability. Still, I would not make it the main production reranker before empirical validation:

- It is substantially slower than MiniLM over 50–100 candidates.
- Prompt length scales with candidate description length.
- Listwise prompts limit candidate count or require tournament schemes.
- Deterministic prompting reduces variance but does not guarantee calibrated rankings.
- It may overfit to the evaluator’s behavior on public data if used carelessly.

Better use: generate **offline soft supervision**. For each training dialogue, score the gold and mined candidates with the available Qwen model, preserve high-scoring alternatives as “do not treat as negative,” and use medium-scoring candidates for reranker distillation. Report this transparently as externally generated teacher supervision, not as ground truth.

## Evaluation plan

### Core metric table

Your report’s primary table should look like this:

| System | Public Recall@10 | Public MRR@10 | Seen-target MRR@10 | New-target MRR@10 | Time/query | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| BM25 |  |  |  |  |  |  |
| Dense base model |  |  |  |  |  |  |
| Dense fine-tuned |  |  |  |  |  |  |
| BM25 + dense RRF |  |  |  |  |  |  |
| RRF + MiniLM rerank@50 |  |  |  |  |  |  |
| Best reranker/system |  |  |  |  |  |  |

Use the **grader’s relevance logic**, not only exact target-ID metrics. Because graded alternatives count, a system may improve official MRR even when exact-ID MRR remains flat.

### Required ablations

Prioritize the following; they map directly to the assignment’s real risks.

| Ablation | Compare | Main question answered |
|---|---|---|
| Query context | Full dialogue vs user-only vs final-turn-only | Does multi-turn history convey retrieval-critical constraints? |
| Product serialization | Raw text vs title/spec-first template | Does light structure improve retrieval? |
| Dense model | Base vs fine-tuned; Qwen3-0.6B vs mxbai vs BGE-M3 | Which model transfers to mixed-category product text? |
| Negatives | In-batch only vs BM25 hard negatives vs dense hard negatives vs filtered mixture | Do hard negatives help without punishing acceptable alternatives? |
| Candidate generation | BM25 vs dense vs RRF | Is lexical/dense complementarity real on this catalog? |
| Reranking | None vs rerank@20/50/100 | Where does MRR saturation justify latency? |
| Long text | 512 head vs 1,024 head vs selective chunking/full context | Which description treatment captures constraints? |
| Generalization | Seen vs new public products | Does the approach survive catalog expansion? |

### Visualizations worth including

1. **Recall@k and MRR@k curves**, \(k \in \{1,3,5,10,20,50,100\}\), for BM25, dense, RRF, and reranked systems. This separates first-stage candidate recall from reranker ordering quality.

2. **Seen vs. new target paired bars** for Recall@10 and MRR@10. This should be prominent, because 50% of public targets are new products. [ppl-ai-file-upload.s3.amazonaws](user-uploaded attachment)

3. **Per-category MRR@10/Recall@10 heatmap or grouped bars.** It tests whether the broad general-domain model behaves consistently across Beauty, Pet Supplies, Toys & Games, Sports & Outdoors, and Home & Kitchen.

4. **Latency–quality Pareto plot**: x-axis end-to-end milliseconds/query, y-axis public MRR@10; label rerank depths. Include peak VRAM. This justifies the final system under the actual 8GB/one-hour constraint.

5. **Similarity distributions**: positive vs BM25 hard negative vs dense hard negative vs random negative. Plot cosine scores before and after fine-tuning. If hard negatives overlap heavily with positives, discuss potential alternative-product false negatives.

6. **Rank-transition/Sankey-style table**: among queries where RRF retrieved a relevant item in top 50, show where reranking moved the first relevant candidate—from rank 11–50 into top 10, rank 4–10 into top 3, and so on. This makes reranker value concrete.

7. **Qualitative error taxonomy** with 10–20 manually sampled public errors: wrong category, missed exact specification, brand/name mismatch, long-description truncation, conversational-coreference failure, and plausible alternative rejected by the fixed judge.

## Execution order

1. Implement canonical text construction, BM25, a dense base encoder, FAISS/indexed cosine retrieval, RRF, and evaluation instrumentation.
2. Establish BM25, dense-only, and RRF baselines on a target-disjoint train validation set.
3. Fine-tune one compact dense model using cached MNRL/InfoNCE with only in-batch negatives.
4. Add conservative BM25+dense hard negatives and re-evaluate, especially on unseen product targets.
5. Fine-tune MiniLM cross-encoder and measure rerank depths 20, 50, and 100 on the actual RTX 3070.
6. Test Qwen3-Embedding-0.6B and/or BGE-M3 only after the compact baseline is reproducible.
7. Run final public evaluations and a complete hardware benchmark using 75k products and 1,000 queries.
8. Package only artifacts demonstrably necessary for the winning system; include total artifact size and cold-start load time.

## Concrete recommendation

Submit the best measured version of:

```text
Full role-tagged dialogue
   ├── BM25 top 200
   └── Fine-tuned Qwen3-Embedding-0.6B top 200
             ↓
        RRF union
             ↓
  Fine-tuned MiniLM-L6 cross-encoder rerank top 50
             ↓
         final top 10
```

If Qwen3-Embedding introduces dependency, package-size, or 8GB throughput risk, substitute `mxbai-embed-large-v1` and use a 768–1,024-token specification-first product representation. If reranking top 50 does not materially improve **graded public MRR@10** over RRF, submit RRF-only rather than adding runtime risk.

The most important empirical claim your report should establish is not “our model trained well.” It is: **the chosen pipeline maintains performance when the target product was never available during training**, while still meeting the 8GB, one-hour, and download constraints.