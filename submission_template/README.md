# ADL 2026 HW2 - product retrieval for a sales chatbot

Submission contents: `retrieval_utils.py` (the only edited provided file), `requirements.txt`,
`download.sh`, `run.sh`, `report.pdf`, `code/script/` (all training / analysis / report scripts).

## Method in one paragraph

`retrieve_and_rerank()` encodes every product with a fine-tuned Qwen3-Embedding-0.6B (dense), also ranks
products with BM25, fuses both top-200 lists with reciprocal rank fusion (k=60), reranks the top 30
with a fine-tuned Qwen3-Reranker-0.6B (yes/no logit score) and returns the top 10. Details, ablations
and numbers are in `report.pdf`.

## Reproducing the submitted result (inference)

1. Use the environment from `requirements.txt` (Python 3.12.9, one GPU with 8GB VRAM).
2. `bash download.sh` - downloads the two fine-tuned models into `./models/dense` and `./models/reranker`
   (about 2.4GB). No external training data is used.
3. Run the TA's harness from a directory that contains the TA's `retrieval.py`, `evaluate.py`,
   `eval_utils.py`, this `retrieval_utils.py` and `models/`:
   `python retrieval.py --dialogues data/public/dialogues.jsonl --products data/public/products.jsonl --output results/retrieval_results.json`
   then `python evaluate.py --retrieval-results results/retrieval_results.json --dialogues data/public/dialogues.jsonl --products data/public/products.jsonl --output results/evaluation.json --batch-size 8`.

Knobs (environment variables read by `retrieval_utils.py`, defaults are the submitted setting):
`HW2_DENSE_LEN` (512), `HW2_DEPTH` (30), `HW2_DOC_CHARS` (1800), `HW2_BM25` (1), `HW2_DENSE` /
`HW2_RERANK` (model directories).

## Reproducing training

`bash run.sh` with the TA's training data in `data/train/{dialogues,products}.jsonl` (override with
`HW2_DATA`). It runs, in order: (1) zero-shot BM25 + Qwen3-Embedding-0.6B candidates for every train
dialogue (`cands.py`), (2) the official judge over the top-20 candidates of each train dialogue
(`judge_labels.py`), (3) dense fine-tuning (`ft_biencoder.py`), (4) reranker fine-tuning
(`ft_reranker.py`). Trained on one NVIDIA H200 (step 3 about 35 minutes and step 4 about 20 minutes
when the GPU is shared; step 2 about 1.5 hours on a shared GPU). The base models
(`Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen2.5-1.5B-Instruct`) are fetched
from the Hugging Face hub by the scripts.

Other scripts in `code/script/`: `zs_eval.py` (zero-shot BM25 / dense baselines), `rerank_eval.py`
(rerankers over cached candidates), `q2_generate.py` (report Q2 generation analysis),
`make_figures.py` (report figures), plus the TA's `eval_utils.py` / `evaluate.py` copies that the
scripts import.
