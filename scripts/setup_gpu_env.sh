#!/usr/bin/env bash
# Builds the conda env used for every experiment (tested on gsm-gpu2, CUDA 12.x driver).
# Usage: bash scripts/setup_gpu_env.sh [env_name]    (default adl_hw2_env)
# Note: the env was built with Python 3.10; the grading environment is Python 3.12.9, which we did not test.
set -euo pipefail
ENV_NAME="${1:-adl_hw2_env}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
"$CONDA" create -y -n "$ENV_NAME" python=3.10
PIP="$(dirname "$CONDA")/../envs/$ENV_NAME/bin/pip"
"$PIP" install torch==2.8.0 sentence-transformers==5.1.0 transformers==4.56.1 numpy==1.26.4 \
    accelerate faiss-gpu-cu12==1.12.0 tqdm scikit-learn
# installed afterwards; datasets must stay on 4.0.0 / huggingface-hub <1.0 or transformers 4.56.1 breaks
"$PIP" install "datasets==4.0.0" "huggingface-hub>=0.34,<1.0" matplotlib gdown
echo "env ready: $(dirname "$CONDA")/../envs/$ENV_NAME/bin/python3"
