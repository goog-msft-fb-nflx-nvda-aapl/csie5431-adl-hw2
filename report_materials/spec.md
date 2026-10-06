# ADL 2026 Homework 2: Retrieval-Augmented Generation

Consolidated from `ADL2026-HW2.pptx.pdf` (ground truth — this file supersedes
`ADL2026-HW2.pptx.md`, a Qwen OCR extraction kept only for reference; spot-checked
against the PDF page by page, no discrepancies found, but the PDF wins on conflict).

**Deadline:** 2026/10/25 23:59:59

## Task

A customer asks a salesbot to recommend a product related to their need, over a short
multi-turn conversation. Build the **retrieval** half of a RAG pipeline: given the
conversation so far (always ending on a user turn, before the bot's recommendation),
retrieve + rerank product specs from a pool, such that the true target product (and/or
other LLM-judged-acceptable alternatives) lands in the top 10.

```
Load data -> Retrieve & Rerank -> Save results
```

- Only `retrieve_and_rerank()` in `retrieval_utils.py` may be modified. Everything else
  in the provided harness (`retrieval.py`, `evaluate.py`, `eval_utils.py`) is fixed and
  must not be changed during testing.
- `retrieval.py` is what the TA runs:
  `python retrieval.py --dialogues "dialogues_path" --products "products_path" --output "output_path"`

## Dataset

Two parts: short user/bot conversations, and a product pool for retrieval. Each
conversation has one gold target product, but other products in the pool may also be
acceptable — judged by a fixed LLM (`Qwen2.5-1.5B-Instruct`). Train + public-test splits
are provided; private test is held out.

| Split | Dialogues | Product pool |
|---|---|---|
| Train | 4,830 | 50,000 (base pool) |
| Public test | 1,000 | 75,000 (base 50,000 + new 25,000) |
| Private test | 2,000 | 100,000 (public 75,000 + new 25,000) |

See `docs/EDA.md` for what's actually in this data (category mix, text lengths,
train/public pool overlap, etc.) — confirmed from the real files, not just the slides.

## Evaluation (`evaluate.py`, fixed, not uploaded in the submission)

- In addition to the gold product, other relevant products count as correct — judged by
  `Qwen2.5-1.5B-Instruct` per retrieved candidate (binary accept/reject).
- Metrics: **Recall@10** and **MRR@10**, both over the graded-relevance gains the judge
  assigns (gold = gain 1, LLM-accepted alternative = gain 1, rejected = gain 0).
- TA uses their own copy of `evaluate.py` (identical to the one provided) — don't upload it.
  ```
  python evaluate.py --retrieval-results "path" --dialogues "path" --products "path" --output "path" --batch-size 8
  ```

## Environment

- Python 3.12.9
- Recommended packages (can swap others, but must list in `requirements.txt`):
  `transformers==4.56.1`, `torch==2.8.0`, `datasets==4.0.0`, `tqdm==4.67.1`,
  `faiss-gpu-cu12==1.12.0`, `sentence-transformers==5.1.0`, `python-dotenv==1.1.1`,
  `accelerate==1.10.1`, `huggingface_hub==0.24.0`, `gdown`
  (note: the provided `requirements.txt` in this repo pins slightly different versions
  for `accelerate`/`huggingface_hub`/`datasets` — `>=` not `==` for some; `numpy==1.26.4`
  added. Both are fine per "can use other packages as long as listed.")

## Rules — cannot do

- Use external training data from `Abner0803/Amazon-Beauty-S2` (HF) — almost certainly
  the dataset this benchmark's train/public/private splits were built from; using the
  raw HF version directly risks leaking test overlap and is explicitly banned.
- Fine-tune on test data. Try to find private test data.
- Cheating/plagiarism: using others' code (published or classmates'), sharing trained
  models/predictions/report answers, publishing code before the deadline.
- Violations -> zero/negative score, school punishment.

## Submission

Zip named `<student-id>.zip` (lowercase), containing a folder of the same name:

```
r13000000.zip
└── r13000000/
    ├── download.sh
    ├── run.sh
    ├── retrieval_utils.py
    ├── requirements.txt
    ├── report.pdf
    ├── README.md
    └── code/script   (all code/script used to train or write the report)
```

- `download.sh`: downloads model + external training data only. No `pip install`, no
  modifying the test environment, nothing beyond downloading. Must finish within 30 min
  at 10MB/s. Don't modify after the deadline (considered cheating). Keep URLs valid 3+
  weeks post-deadline. Executed before evaluation.
- `run.sh`: reproduces the training process. Must match what `README.md` describes
  step by step. No/empty `README.md` costs 2%. Unreproducible result risks more points.
- **Execution environment**: Ubuntu 22.04, 32GB RAM, **RTX 3070 8GB VRAM**, 20GB disk.
  Same conda env/packages as above. Public-set retrieval must finish within **1 hour**.

## Grading

**Model performance (14%)** — all four numbers gated simultaneously per tier:

| | Public Recall@10 | Public MRR@10 | Private Recall@10 | Private MRR@10 |
|---|---|---|---|---|
| Boss baseline | 0.950 (1%) | 0.800 (1%) | 0.930 (1%) | 0.780 (1%) |
| Strong baseline | 0.885 (1%) | 0.680 (1%) | 0.880 (1%) | 0.675 (1%) |
| Simple baseline | 0.860 (2%) | 0.650 (2%) | 0.845 (1%) | 0.635 (1%) |

**Report (6%)** — lose 2% if file structure is wrong.

- **Q1: Retrieval & Reranking Strategy (4%).** Describe the strategy in detail: how
  training data is constructed (anchor/positive/negative sampling, external data types
  used), loss function spec + rationale, hyperparameters, how many strategies were tried
  and why the submitted one wins.
- **Q2: Generation Analysis (2%).** Use `Qwen2.5-1.5B-Instruct` to generate responses
  from retrieved specs. >=5 generation results from different samples with qualitative
  analysis. >=3 different generation prompts compared. Include all related files in
  `code/script`. Scored on richness/depth of analysis.

## Late submission

- 0 < late <= 2 days: score x 0.70. > 2 days: score x 0.0.
- Only the last NTU COOL submission counts — a post-deadline update costs the penalty
  even if an earlier on-time submission existed.

## Contact

NTU COOL discussion forum first, then `adl-ta@csie.ntu.edu.tw` (prefix `[ADL2026 HW2]`).
TA hours: Thu 16:00-17:00 @ R524 (陳剛頡), Wed 14:30-15:30 @ Google Meet (彭紀綸).

## Provided files (downloaded into this repo)

Source: Google Drive folder linked from the slides' "Homework 2 files" (page 3) —
`https://drive.google.com/drive/folders/1tBS9Qk8s86WzVabpVeHtbIZr9bfKDJ8I`.

- `data/{train,public}/{dialogues.jsonl,products.jsonl}` — the actual data.
- `provided/retrieval.py`, `provided/evaluate.py`, `provided/eval_utils.py` — fixed
  harness, reference copies only, kept out of the submission (TA uses their own).
- `provided/README_provided.md` — the TA's own quickstart (superseded by this repo's
  top-level `README.md` once that's written).
- `retrieval_utils.py`, `requirements.txt` at repo root — the two provided files that
  **do** need to end up in the submission; `retrieval_utils.py` is the only one meant
  to be edited.
