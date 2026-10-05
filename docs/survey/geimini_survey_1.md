## **1\. State-of-the-Art Retrieval Architectures (50k–100k Scale)**

For asymmetric, short-conversational-query to long-product-passage retrieval at a 50k–100k candidate scale within an 8GB VRAM footprint and 1-hour time budget, multi-stage hybrid pipelines significantly outperform single-model solutions.

Conversational Query (3–5 turns)  
       │  
       ├──► BM25 (pyserini / bm25s) ─────────────► Top-100 candidates  
       │                                                │  
       └──► Dense Bi-Encoder (FAISS / USearch) ──► Top-100 candidates  
                                                        │  
                                            Reciprocal Rank Fusion (RRF)  
                                                        │  
                                                  Top-50 candidates  
                                                        │  
                                            Cross-Encoder Reranker  
                                                        │  
                                                  Final Top-10

### **Bi-Encoder (Dense Dual-Encoder)**

Dense dual-encoders embed queries and product passages into a shared vector space independently. Pre-indexing 75,000 product embeddings into an in-memory index (e.g., FAISS or USearch using FP16 or INT8 scalar quantization) requires under 200 MB VRAM/RAM for a 1024-dimensional model. During evaluation, query encoding takes \~5–10 ms per query on an RTX 3070, making this stage fast and lightweight.

### **Late-Interaction (ColBERTv2 / PyLate)**

ColBERT preserves token-level embeddings and uses a MaxSim operator. While ColBERTv2 achieves strong recall on asymmetric tasks, indexing 75,000 products with \~250 tokens per product produces \~18.75 million token vectors. The resulting PLAID index requires 2–4 GB of system RAM/VRAM. Although technically runnable on 32GB RAM, token-level retrieval latency across 1,000 queries approaches 15–20 minutes, offering a tighter speed-to-recall trade-off than a dense bi-encoder coupled with a cross-encoder.

### **Sparse Retrieval (BM25 / SPLADE)**

Lexical search via BM25 (bm25s or pyserini) acts as a complement to dense retrieval. BM25 catches exact keyword matches (e.g., model numbers, specific brand names, exact dimensions) that dense embeddings can smooth over. Running BM25 over 75,000 items takes less than 3 seconds on CPU. Neural sparse methods (SPLADE) expand keywords effectively, but BM25 is faster and incurs zero VRAM footprint.

### **Hybrid Fusion (BM25 \+ Dense Bi-Encoder via RRF)**

Merging the top-100 candidates from BM25 and the top-100 candidates from a dense bi-encoder using Reciprocal Rank Fusion (RRF) consistently yields higher candidate recall than either retriever alone. RRF is parameter-free and computes instantly:

\$\$RRF\\\_Score(p) \= \\sum\_{m \\in M} \\frac{1}{k \+ r\_m(p)}\$\$  
where \$M \= \\{\\text{BM25}, \\text{Dense}\\}\$, \$r\_m(p)\$ is the rank of product \$p\$ in retriever \$m\$, and \$k \= 60\$.

## **2\. Pre-trained Embedding Models on HuggingFace**

Given that products span 6 broad Amazon categories (Sports & Outdoors, Pet Supplies, Toys & Games, Home & Kitchen, Beauty, Beauty & Personal Care), open-domain instruction-tuned models generalize best across the catalog without overfitting to beauty-specific terms.

| Model Name on HuggingFace | Parameters | Embedding Dim | Max Seq Length | FP16 VRAM Footprint | Key Recommendation & Trade-offs |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct | 1.5B | 1536 | 32,768 | \~3.1 GB | **Top Pick for Quality:** Based on Qwen2 architecture; matches the target environment. Excellent zero-shot generalization to unseen products. |
| BAAI/bge-m3 | 567M | 1024 | 8,192 | \~1.2 GB | **Top Pick for Hybrid Search:** Natively supports dense, multi-vector, and sparse embeddings simultaneously with strong long-text capabilities. |
| BAAI/bge-large-en-v1.5 | 335M | 1024 | 512 | \~0.7 GB | **Top Pick for Speed:** Fast fine-tuning on H200, minimal memory footprint during inference, high baseline retrieval recall on BEIR/MTEB. |
| intfloat/e5-large-v2 | 335M | 1024 | 512 | \~0.7 GB | Strong asymmetric instruction support (query: ... vs passage: ...). Simple to fine-tune with standard contrastive loss. |

## **3\. Reranking Strategies Under Hardware & Time Constraints**

The hardware constraints are an RTX 3070 (8GB VRAM, 32GB RAM) and an evaluation time limit of 1 hour for 1,000 queries against a 75,000-item pool.

┌────────────────────────────────────────────────────────────────────────┐  
│ Candidate Shortlist Depth (M \= 50 vs M \= 100\)                         │  
├────────────────────────────────────────────────────────────────────────┤  
│ • M \= 100: Top-100 candidates \-\> 100,000 cross-encoder evaluations      │  
│   Latency: \~35-45 minutes on RTX 3070 (Batch Size 32, FP16)           │  
│ • M \= 50:  Top-50 candidates  \-\> 50,000 cross-encoder evaluations       │  
│   Latency: \~15-20 minutes on RTX 3070 (Batch Size 32, FP16)           │  
└────────────────────────────────────────────────────────────────────────┘

### **Cross-Encoder Reranking (BAAI/bge-reranker-v2-m3 or BAAI/bge-reranker-large)**

Cross-encoders jointly attend to query and product text, capturing fine-grained interactions that dual-encoders miss. Evaluating 50 candidates across 1,000 queries requires 50,000 forward passes. At a batch size of 32 in FP16 on an RTX 3070, bge-reranker-v2-m3 takes \~18 minutes, well under the 60-minute limit.

### **LLM Reranker (Qwen2.5-1.5B-Instruct)**

Using the evaluation judge model (Qwen2.5-1.5B-Instruct) directly as a pointwise or listwise reranker is possible:

* **Pointwise Scoring:** Prompt the LLM to output Yes/No for relevance and use the log-probability of Yes as the reranking score.  
* **Latency Trade-off:** Pointwise reranking over 20 candidates per query requires 20,000 LLM calls. Even with vLLM or optimized FlashAttention-2, inference takes \~12–15 minutes on an RTX 3070\. However, empirical benchmarks show dedicated cross-encoders (like bge-reranker-v2-m3) outperform 1.5B general LLMs on pure ranking accuracy while consuming less VRAM.

### **Strategy Recommendation**

Use a 2-stage setup: **Hybrid Retrieval (BM25 \+ Bi-Encoder) \$\\rightarrow\$ Top-50 Shortlist \$\\rightarrow\$ Cross-Encoder (bge-reranker-v2-m3) Rerank \$\\rightarrow\$ Top-10 Output.**

## **4\. Training-Data Construction & Negative Mining**

The dataset contains only one gold label target per conversation query. Effective dense retrieval training requires constructing high-quality negative pairs without penalizing unlabeled valid alternatives.

    Gold Target (Passage \+) ──────────────────────────────► Positive Pair  
                                                                │  
    Mined Candidates (BM25 / Bi-Encoder Top-100)                │  
       ├── Rank \#1 \- \#4  ─────────────────────────────────► Omit (Potential Valid Alternatives)  
       └── Rank \#5 \- \#50 ─────────────────────────────────► Hard Negatives (-)

### **Mining Hard Negatives**

> 1. **Source Strategy:** Run BM25 and an un-tuned bi-encoder baseline over the training queries to retrieve the top-100 passages for each query.  
> 2. **Filtering Potential Acceptable Alternatives:** Because non-gold products in top results can be judged as valid recommendations, candidates ranked \#2 through \#4 by BM25 or semantic similarity are often functional duplicates or acceptable alternatives. Omit ranks \#1–\#4 (excluding gold) and sample hard negatives from ranks **\#5 to \#50**.  
> 3. **Negative Ratio:** Use 1 gold target \+ 7 mined hard negatives per query, plus in-batch negatives during multi-GPU training on 4x H200.

### **Handling Graded Relevance / Label Noise**

To prevent the model from penalizing non-labeled acceptable products, train using **Margin MSE (Teacher Distillation)**:

* Pass the query, gold target, and mined negative through a strong teacher model (e.g., BAAI/bge-reranker-v2-m3).  
* Train the student bi-encoder to match the score *difference* (margin) assigned by the teacher cross-encoder rather than forcing a strict binary 0/1 target.

## **5\. Generalizing to Unseen Products (Cold-Start Catalog Shift)**

50% of the public evaluation queries (500 out of 1,000) target products that never appeared in the training pool. To prevent memorization and ensure zero-shot generalization:

### **No Explicit Product ID Embeddings**

Ensure the architecture strictly processes text inputs (title \+ description) and never learns fixed lookup embeddings tied to parent\_asin.

### **Structured Passage Formatting**

Unstructured Amazon descriptions embed structured fields (brand, category path, dimensions) as prose. Formatting product text into a clean schema before embedding improves feature alignment:

Plaintext  
Category: {category\_path} | Brand: {brand} | Title: {title} | Details: {clean\_description}

### **Instruction-Aware Queries**

Prefix conversation queries with explicit instructions during embedding to ground the model on intent:

Plaintext  
Instruct: Given a multi-turn customer shopping conversation, retrieve the product passage that best satisfies the user's explicit and implicit constraints.  
Query: {turn\_1} \\n {turn\_2} ... \\n {turn\_n}

## **6\. Loss Functions for Asymmetric Fine-Tuning**

### **MultipleNegativesRankingLoss (MNRL / InfoNCE)**

Standard cross-entropy loss over a similarity matrix. For a query \$q\_i\$, positive product \$p\_i^+\$, and negatives \$p\_j^-\$:

\$\$\\mathcal{L}\_{MNRL} \= \-\\log \\frac{\\exp(\\text{sim}(q\_i, p\_i^+) / \\tau)}{\\exp(\\text{sim}(q\_i, p\_i^+) / \\tau) \+ \\sum\_{j} \\exp(\\text{sim}(q\_i, p\_j^-) / \\tau)}\$\$  
where \$\\text{sim}(u, v) \= \\frac{u \\cdot v}{\\Vert{}u\\Vert{} \\Vert{}v\\Vert{}}\$ and temperature \$\\tau \\in \[0.02, 0.05\]\$.

### **Margin MSE Loss (Knowledge Distillation)**

When mined hard negatives contain potential valid alternatives, Margin MSE matches the relative score distance predicted by a cross-encoder teacher:

\$\$\\mathcal{L}\_{MarginMSE} \= \\left( \\Big(\\text{sim}(q, p^+) \- \\text{sim}(q, p^-)\\Big) \- \\Big(S\_{\\text{teacher}}(q, p^+) \- S\_{\\text{teacher}}(q, p^-)\\Big) \\right)^2\$\$

### **Cross-Entropy Reranker Loss**

For fine-tuning the cross-encoder stage over query-passage pairs:

\$\$\\mathcal{L}\_{CE} \= \-y \\log(\\sigma(S(q, p))) \- (1 \- y) \\log(1 \- \\sigma(S(q, p)))\$\$

## **7\. Sequence Length & Product Description Handling**

Product descriptions average 1,720 characters (\~350–400 tokens) but reach up to 6,516 characters (\~1,300–1,500 tokens).

Passage Text (up to 6,500 chars)  
  │  
  ├──► Head Truncation (First 512 Tokens) ──► Retains Title, Brand, Category Path & Main Features  
  │  
  └──► Late Chunking / Mean Pooling ────────► Chunk 1 (0–400) ──┐  
                                             Chunk 2 (300–700) ─┼──► Mean Pooling  
                                             Chunk 3 (600–1000) ┘

* **Head Truncation at 512 Tokens:** Amazon descriptions place high-value tokens (Title, Brand, Category, Bullet Points) at the beginning. Truncating at 512 tokens preserves \~90% of descriptive features while keeping memory consumption low.  
* **Late Chunking (Alternative for Long Passages):** For passages exceeding 512 tokens, pass the text through models supporting longer context windows (e.g., bge-m3 with 8k limit or gte-Qwen2-1.5B-instruct with 32k limit). Alternatively, split passages into overlapping 400-token chunks with 100-token overlaps, compute chunk embeddings, and apply mean pooling.

## **8\. Ablation Experiments Plan**

Structure the empirical evaluation in the report using these controlled comparisons:

| Ablation ID | Target Hypothesis | Experimental Variants | Evaluated Metrics |
| :---- | :---- | :---- | :---- |
| **Abl-1** | Query Context Length | Last Turn Only vs. Last 2 Turns vs. All Turns Concatenated | Recall@10, MRR@10 |
| **Abl-2** | Retriever Architecture | BM25 vs. Dense Bi-Encoder vs. Hybrid (BM25 \+ Dense RRF) | Recall@10, MRR@10 |
| **Abl-3** | Embedding Model Backbone | bge-large-en-v1.5 vs. bge-m3 vs. gte-Qwen2-1.5B-instruct | Recall@10, Latency (ms) |
| **Abl-4** | Hard Negative Mining | In-Batch Random Negatives vs. BM25 Hard Negs vs. Bi-Encoder False Positives | Recall@10, MRR@10 |
| **Abl-5** | Reranker Stage & Depth | No Reranker vs. LLM Reranker (Qwen2.5-1.5B) vs. Cross-Encoder (bge-reranker-v2-m3) at Top-20, Top-50, Top-100 | Recall@10, Total Time (min) |
| **Abl-6** | Catalog Generalization Gap | Split Public Eval Set into **Seen** (Train Pool) vs. **Unseen** (25k New Items Pool) targets | Recall@10 (Seen) vs. Recall@10 (Unseen) |

## **9\. Visualization & Quantitative Analysis Plan**

Include these specific figures and visualizations in the report to present the quantitative findings clearly:

> 1. **Recall@k and MRR@k Curves (\$k \\in \\{1, 2, 5, 10, 20, 50, 100\\}\$):** Line chart illustrating candidate recall progression across pipeline stages (BM25 vs Dense vs Hybrid vs Reranked).  
> 2. **Seen vs. Unseen Catalog Generalization Gap:** Grouped bar chart comparing Recall@10 performance on the 500 Seen train-pool products against the 500 Unseen public-pool products to demonstrate zero-shot generalization.  
> 3. **Category Breakdown Analysis:** Breakdown of Recall@10 across the 5 primary categories (Sports & Outdoors, Pet Supplies, Toys & Games, Home & Kitchen, Beauty) to verify non-beauty domain generalization.  
> 4. **Latency vs. VRAM Trade-off (Pareto Frontier):** Scatter plot charting Recall@10 against inference runtime (in minutes) and peak VRAM allocation (GB) across candidate depth configurations (\$M \\in \\{20, 50, 100, 200\\}\$).  
> 5. **Cosine Similarity Score Distribution Histogram:** Density plot showing embedding similarity distributions between Query-Gold pairs, Query-Hard Negative pairs, and Query-Random Negative pairs to visualize representation separation.

What specific aspect of this proposed retrieval and reranking pipeline would you like to explore or benchmark first?