#!/bin/bash
# Dev convenience only — NOT the submission's download.sh (that one fetches our
# trained model/external data, not the TA's provided dataset).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

gdown --folder "https://drive.google.com/drive/folders/1tBS9Qk8s86WzVabpVeHtbIZr9bfKDJ8I" -O hw2_files_tmp
mkdir -p data
mv hw2_files_tmp/data/train data/train
mv hw2_files_tmp/data/public data/public
rm -rf hw2_files_tmp
echo "data/{train,public} populated."
