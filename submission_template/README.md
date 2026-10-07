# ADL 2026 HW2 - product retrieval for a sales chatbot

Submission contents: `retrieval_utils.py` (the only edited provided file), `requirements.txt`,
`download.sh`, `run.sh`, `report.pdf`, `code/script/` (all training / analysis / report scripts).

Repository (private until after the deadline): https://github.com/goog-msft-fb-nflx-nvda-aapl/csie5431-adl-hw2

## Method in one paragraph

`retrieve_and_rerank()` encodes every product with a fine-tuned Qwen3-Embedding-0.6B (dense, 384 tokens), also ranks
products with BM25, fuses both top-200 lists with reciprocal rank fusion (k=60), reranks the top 30
with a fine-tuned Qwen3-Reranker-0.6B (yes/no logit score, 1,400 characters of product text) and returns the top 10.
A wall-clock guard lowers the encoding length / rerank depth if the measured speed would exceed the 1-hour limit.
Details, ablations and numbers are in `report.pdf`.

## Reproducing the submitted result (inference)

1. Use the environment from `requirements.txt` (Python 3.12.9, one GPU with 8GB VRAM).
2. `bash download.sh` - downloads into this folder `models/dense`, `models/reranker` (about 2.4GB, our fine-tuned
   models) and `external/judge_labels_train.json` (1.7MB, see "External data" below).
3. Run the TA's harness from a directory that contains the TA's `retrieval.py`, `evaluate.py`,
   `eval_utils.py`, this `retrieval_utils.py` and `models/`:
   `python retrieval.py --dialogues data/public/dialogues.jsonl --products data/public/products.jsonl --output results/retrieval_results.json`
   then `python evaluate.py --retrieval-results results/retrieval_results.json --dialogues data/public/dialogues.jsonl --products data/public/products.jsonl --output results/evaluation.json --batch-size 8`.

Knobs (environment variables read by `retrieval_utils.py`, defaults are the submitted setting):
`HW2_DENSE_LEN` (384), `HW2_DEPTH` (30), `HW2_DOC_CHARS` (1400), `HW2_BM25` (1), `HW2_BUDGET` (3000 seconds, the time
guard), `HW2_DENSE` / `HW2_RERANK` (model directories).

## External data

No external dataset is used. `external/judge_labels_train.json` is derived only from the provided training data: the
provided `Qwen2.5-1.5B-Instruct` judge (the TA's `evaluate.py` judge prompt, gold product as reference) was run on the
top-20 zero-shot candidates of each of the 4,830 training dialogues (93,198 judgments; accepted = 1, rejected = 0).
It is produced by `code/script/judge_labels.py` (step 2 of `run.sh`); because that step takes about 2 hours 20 minutes
on a shared H200 (not measured on a 3070), `download.sh` fetches the file and `run.sh` only regenerates it if the file
is missing (`rm external/judge_labels_train.json` to force it).

## Reproducing training

`bash run.sh` from this folder, with the TA's `data/train/{dialogues,products}.jsonl` present as `data/train/` (override
the location with `HW2_DATA`). It runs, in order: (1) zero-shot BM25 + Qwen3-Embedding-0.6B candidates for every train
dialogue (`cands.py`), (2) the judge labels as above (`judge_labels.py`, skipped when the file exists), (3) dense
fine-tuning (`ft_biencoder.py`), (4) reranker fine-tuning (`ft_reranker.py`), writing `models/dense` and
`models/reranker`. All four steps fit an 8GB GPU: peak allocated memory measured on an H200 under a 7GiB cap was 3.34
(step 1), 4.07 (step 2), 4.99 (step 3) and 5.85 GiB (step 4). Measured on a shared H200 (not on a 3070): the whole `run.sh` with the
labels step skipped took 40 minutes (step 1 about 6 minutes, step 3 16 minutes, step 4 about 15-45 minutes depending on GPU
load); running it from the unzipped submission under a 7GiB cap and then the TA's harness gave graded R@10 0.979 and
MRR@10 0.895 on public. The base models (`Qwen/Qwen3-Embedding-0.6B`,
`Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen2.5-1.5B-Instruct`) are fetched from the Hugging Face hub by the scripts.

Other scripts in `code/script/`: `zs_eval.py` (zero-shot BM25 / dense baselines, also imported by `cands.py`),
`rerank_eval.py` (rerankers over cached candidates), `q2_generate.py` (report Q2 generation analysis),
`make_figures.py` (report figures), `vram.py` (optional memory cap `HW2_VRAM_GB`, peak-memory report), plus the TA's
`eval_utils.py` / `evaluate.py` copies that the scripts import.
