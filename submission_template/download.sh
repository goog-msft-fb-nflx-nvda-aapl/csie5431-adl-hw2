#!/usr/bin/env bash
# Downloads the two fine-tuned models (dense retriever + reranker, ~2.4GB total) into ./models.
# No external training data is used.
set -euo pipefail
cd "$(dirname "$0")"
MODELS_URL="${MODELS_URL:-https://drive.google.com/file/d/11UhLDC84CfulU4-plYaUMDsj4ZACADPN/view?usp=sharing}"
if [[ "$MODELS_URL" == __SET_* ]]; then echo "models URL not set" >&2; exit 1; fi
FILE_ID="$(echo "$MODELS_URL" | sed -E 's#.*/d/([^/?]+).*#\1#')"
python -m gdown "$FILE_ID" -O models.tar
tar -xf models.tar
rm -f models.tar
test -f models/dense/config.json && test -f models/reranker/config.json
