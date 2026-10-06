#!/usr/bin/env bash
# Reproduces training of the two submitted models (dense retriever, reranker) from the TA's
# training data. Needs a GPU with >= 24GB memory (we trained on an H200), HF hub access for the two
# public base models, and data in $HW2_DATA (default ./data) laid out as data/train/{dialogues,products}.jsonl.
set -euo pipefail
cd "$(dirname "$0")"
export HW2_DATA="${HW2_DATA:-data}"
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
S=code/script
mkdir -p cache models results

# 1. zero-shot BM25 + Qwen3-Embedding-0.6B candidates for every train dialogue (used for negative mining)
python $S/cands.py --split train --model qwen3-0.6b --n 200
# 2. official judge (Qwen2.5-1.5B-Instruct, gold as reference) over the top-20 candidates of each train dialogue
python $S/judge_labels.py --split train --n 20 --bs 32
# 3. dense retriever: CachedMNRL, 7 mined hard negatives per query
python $S/ft_biencoder.py --out models/dense --n-neg 7 --lr 3e-5 --epochs 4 --dev-frac 0
# 4. reranker: listwise softmax CE, 1 positive (gold or judge-accepted alternative) + 15 judge-rejected negatives
python $S/ft_reranker.py --out models/reranker --alt-prob 0.5 --lr 2e-5 --group 16 --qbs 2 --use-all-train
