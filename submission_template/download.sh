#!/usr/bin/env bash
# Downloads the two fine-tuned models (dense retriever + reranker, ~2.4GB total) into ./models.
# No external training data is used.
set -euo pipefail
cd "$(dirname "$0")"
MODELS_URL="${MODELS_URL:-__SET_MODELS_URL__}"
if [[ "$MODELS_URL" == __SET_* ]]; then echo "models URL not set" >&2; exit 1; fi
python -m gdown --fuzzy "$MODELS_URL" -O models.tar
tar -xf models.tar
rm -f models.tar
test -f models/dense/config.json && test -f models/reranker/config.json
