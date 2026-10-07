#!/usr/bin/env bash
# Reproduces training of the two submitted models (dense retriever -> models/dense, reranker -> models/reranker).
# Every step fits an 8GB GPU (peak allocated memory measured on an H200 with a 7GiB cap: step 1 3.3GiB,
# step 2 4.1GiB, step 3 5.0GiB, step 4 5.9GiB). Expects data/train/{dialogues,products}.jsonl (override with HW2_DATA).
# Needs HF hub access for the public base models (Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, Qwen2.5-1.5B-Instruct).
set -euo pipefail
cd "$(dirname "$0")"
export HW2_DATA="${HW2_DATA:-data}"
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
S=code/script
mkdir -p cache models results external

# 1. zero-shot BM25 + Qwen3-Embedding-0.6B candidates for every train dialogue (hard-negative mining, judge candidates)
python $S/cands.py --split train --model qwen3-0.6b --n 200

# 2. official judge (Qwen2.5-1.5B-Instruct, gold as reference) over the top-20 candidates of each train dialogue.
#    download.sh already fetches these labels (external/judge_labels_train.json, produced by this very command), so the
#    slow step is skipped; delete that file to regenerate it from scratch.
if [ ! -f external/judge_labels_train.json ]; then
  python $S/judge_labels.py --split train --n 20 --bs 4
fi

# 3. dense retriever: CachedMNRL, 3 mined hard negatives per query
python $S/ft_biencoder.py --out models/dense --n-neg 3 --lr 3e-5 --epochs 2 --max-len 384 --mini-bs 8 --dev-frac 0

# 4. reranker: listwise softmax CE, 1 positive (gold or judge-accepted alternative) + 7 judge-rejected negatives
python $S/ft_reranker.py --out models/reranker --group 8 --qbs 2 --lr 2e-5 --doc-chars 1400 --alt-prob 0.5 --use-all-train
