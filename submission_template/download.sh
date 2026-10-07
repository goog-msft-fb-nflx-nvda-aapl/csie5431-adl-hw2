#!/usr/bin/env bash
# Downloads the two fine-tuned models (models/dense, models/reranker, ~2.4GB) and the judge labels used as training
# supervision (external/judge_labels_train.json, ~1.7MB; derived from the provided train data with the provided
# Qwen2.5-1.5B-Instruct judge, see README). No other external data is used.
set -euo pipefail
cd "$(dirname "$0")"
MODELS_URL="${MODELS_URL:-https://drive.google.com/file/d/11UhLDC84CfulU4-plYaUMDsj4ZACADPN/view?usp=sharing}"
FILE_ID="$(echo "$MODELS_URL" | sed -E 's#.*/d/([^/?]+).*#\1#')"
python -m gdown "$FILE_ID" -O models.tar
tar -xf models.tar
rm -f models.tar
test -f models/dense/config.json && test -f models/reranker/config.json && test -f external/judge_labels_train.json
