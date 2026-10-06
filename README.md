# ADL HW2 — RAG product retrieval

Retrieval half of a RAG pipeline: given a short multi-turn sales conversation (always
ending on a user turn, before the bot's recommendation), retrieve + rerank product specs
from a pool of 50k-100k items so the true/acceptable target lands in the top 10.
Graded on Recall@10 / MRR@10 (gold + LLM-judged acceptable alternatives), public +
private test, gated at three baseline tiers. Deadline 2026/10/25 23:59:59.

**Status**: final pipeline built and checked (BM25 + fine-tuned Qwen3-Embedding-0.6B, RRF, fine-tuned
Qwen3-Reranker-0.6B on the top 30). Public graded R@10 0.987 / MRR@10 0.908 (H200, `docs/WORKLOG.md` has every
experiment). Not measured: an RTX 3070 run and the private split.

## Layout

- `docs/spec/spec.md` — consolidated assignment spec (PDF-verified).
- `docs/EDA.md` — findings from the actual downloaded data.
- `docs/survey/` — deep-research prompt(s) and responses for the SOTA/method survey.
- `docs/WORKLOG.md` / `docs/TODO.md` — experiment log and backlog (HW1-style).
- `provided/` — TA's fixed harness (`retrieval.py`, `evaluate.py`, `eval_utils.py`) —
  reference copies only, never modified, not part of the submission.
- `retrieval_utils.py` — the only file we implement into; lives at repo root because
  it ships at the submission root too.
- `requirements.txt` — submission-root file, update as dependencies are added.
- `data/` — gitignored, populated by `download_hw2_data.sh` (the actual train/public
  dialogues + product pools, from the TA's Google Drive folder).
- `code/` — experiment/training/analysis scripts (copied to `code/script/` in the submission by
  `scripts_make_submission.sh`); `code/launchers/` has the shell launchers used on the GPU box.
- `scripts/setup_gpu_env.sh` — builds the conda env used for the experiments.
- `submission_template/` — `run.sh`, `download.sh`, `README.md` shipped in the zip.
- `results_archive/` — raw result files and judge labels from the GPU runs.

## Getting the data

```bash
bash download_hw2_data.sh   # pulls data/ from the TA's Drive folder into data/
```

## Environment (GPU box, used for all experiments)

```bash
bash scripts/setup_gpu_env.sh            # conda env adl_hw2_env, pinned versions
bash download_hw2_data.sh                # needs gdown on PATH (installed by the script above)
```

## Reproducing the experiments

Training steps are in `submission_template/run.sh` (it expects the `code/script/` layout, so build it with
`bash scripts_make_submission.sh <student_id>` and run it from `submission/<student_id>/` with `data/` next to it).
Launchers for the individual experiments are in `code/launchers/`.
