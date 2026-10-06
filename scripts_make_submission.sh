#!/usr/bin/env bash
# Assemble ./submission/<id>/ and <id>.zip from the repo (no models, no data). Usage: bash scripts_make_submission.sh r05921008
set -euo pipefail
ID="${1:?student id}"
OUT="submission/$ID"
rm -rf submission && mkdir -p "$OUT/code/script"
cp retrieval_utils.py requirements.txt "$OUT/"
cp submission_template/{run.sh,download.sh,README.md} "$OUT/"
cp docs/report.pdf "$OUT/report.pdf"
cp code/*.py "$OUT/code/script/"
cp provided/eval_utils.py provided/evaluate.py "$OUT/code/script/"
(cd submission && zip -qr "$ID.zip" "$ID")
ls -R submission | head -30
