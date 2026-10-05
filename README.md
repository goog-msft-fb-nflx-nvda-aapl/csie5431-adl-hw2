# ADL HW2 — RAG product retrieval

Retrieval half of a RAG pipeline: given a short multi-turn sales conversation (always
ending on a user turn, before the bot's recommendation), retrieve + rerank product specs
from a pool of 50k-100k items so the true/acceptable target lands in the top 10.
Graded on Recall@10 / MRR@10 (gold + LLM-judged acceptable alternatives), public +
private test, gated at three baseline tiers. Deadline 2026/10/25 23:59:59.

**Status**: setup phase — data downloaded, EDA done, no training/experiments run yet.
See `CLAUDE.md` for full project context (GPU server, GitHub rules, working style) and
`docs/TODO.md` for what's next.

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
- `code/` — experiment/training/analysis scripts (anything beyond `retrieval_utils.py`
  itself), per the submission's `code/script` requirement.

## Getting the data

```bash
bash download_hw2_data.sh   # pulls data/ from the TA's Drive folder into data/
```

## Environment

```bash
conda create -n adl_hw2_env python=3.12.9
conda activate adl_hw2_env
pip install -r requirements.txt
```
