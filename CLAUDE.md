# CLAUDE.md — CSIE5431 ADL 2026 HW2 (RAG retrieval)

Read this first in any new session working in this directory. It's the durable context
that doesn't live in conversation history — GPU server gotchas, GitHub rules, working
style, and where everything else lives. Full experiment history is in `docs/WORKLOG.md`;
what's left is in `docs/TODO.md`. Read both before doing anything.

## The task, in one paragraph

Build the retrieval half of a RAG pipeline for a sales chatbot: given a short multi-turn
conversation (always ends on a user turn), retrieve + rerank product specs from a 50k-100k
pool so the true/acceptable target lands in the top 10. Graded on Recall@10/MRR@10
(gold + LLM-judged-acceptable alternatives), public + private test, three baseline tiers.
**Deadline 2026/10/25 23:59:59.** Full spec: `docs/spec/spec.md` (consolidated, PDF-verified
— `docs/spec/ADL2026-HW2.pptx.pdf` is the ultimate ground truth if anything's ambiguous).
Data already downloaded and EDA'd: `docs/EDA.md`. Headline data fact to keep in mind for
every method decision: the product pool is **general-domain**, not beauty-specific, and
**half the public-test gold targets are products never seen during training** — generalize
from text, don't memorize a closed catalog.

Only `retrieval_utils.py`'s `retrieve_and_rerank()` gets submitted/modified; `provided/`
holds read-only reference copies of the TA's fixed harness. Target deploy hardware is
tighter than HW1: **RTX 3070 8GB VRAM**, 1hr budget for public-set retrieval.

## Project roles — Mac / GPU / GitHub split

Same three-way split that worked for HW1:

- **Mac** (`~/Desktop/CSIE5431/PA2/`): Claude-session work only — code editing, docs,
  worklogs, planning. Never run training/inference/even a smoke test here.
- **GPU** (`gsm-gpu2`): experiments only, disposable. Large checkpoints get deleted once
  a run is superseded — keep only what's part of a live candidate. gsm-gpu2 is 4x H200
  NVL (huge VRAM) — that's for *training*; the actual grading machine is RTX 3070 8GB,
  so always sanity-check the final packaged model actually fits 8GB before assuming a
  method is viable, not just that it trains fine on gsm-gpu2.
- **GitHub** (`goog-msft-fb-nflx-nvda-aapl/csie5431-adl-hw2`, private): code + progress
  management, source of truth going forward.

**Never run model training/inference/smoke-tests locally on the Mac.** Write/edit code
locally (fast iteration), sync to GPU, run everything — including "just checking it
compiles" — over ssh on the GPU box.

## GPU server: gsm-gpu2

- SSH alias already configured (`ssh gsm-gpu2`), user `jtan`, key `~/.ssh/gsm_gpu`.
  Hostname `TO-sv-td-h200nvlnode02`, 4x NVIDIA H200 NVL.
- **No sudo.** Avoid any command that needs it (`apt-get`, `systemctl`, etc.).
- **Scratch scope**: only create/modify/access files under `/home/jtan/...`. Never touch
  other users' home dirs or shared mounts like `/data` (near-full anyway, 100%) — not
  even to browse/read. Scope every command, including broad ones (`find`, `du`,
  `grep -r`), to your own subdirectory.
- **GPU index 3 is broken**: `nvidia-smi` shows it free, but the CUDA runtime only
  enumerates 3 devices — touching index 3 hangs silently (no error, no output, zero GPU
  memory used, the first `.to(device)` just never returns). Confirmed working: GPU0
  (often busy with others' jobs), GPU1, GPU2. Before trusting an index, smoke-test it:
  `CUDA_VISIBLE_DEVICES=<idx> python -c "import torch; x=torch.randn(2000,2000,device='cuda'); (x@x); torch.cuda.synchronize(); print('ok')"`
  (a cold JIT pass takes ~15s, that's normal — hang past ~30s means pick a different index).
  Default to GPU1 or GPU2.
- **GPU0 occasionally malfunctions** in a way that makes `ps aux`/`ps -ef`/bare
  `pgrep`/`pkill` hang indefinitely (not slow — actually hangs). Simple commands
  (`echo`, `cat`, `tail`, direct `/proc/<pid>/cmdline` reads) stay fine throughout. To
  check or kill a *specific known* PID, read `/proc/<pid>/cmdline`/`/proc/<pid>/status`
  or `kill -9 <pid>` directly rather than searching the process table. Don't stack
  multiple overlapping ssh calls if one is already hung — wait it out or abort, don't
  layer on more.
- **Always check `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv`
  across ALL indices** before launching, every time — don't assume GPU0 is default/safe
  (it's everyone's default guess, so it's often the most contended) or that whatever was
  free last time still is.
- **Concurrent jobs**: each PyTorch process defaults to spawning OMP/MKL threads equal to
  total core count. More than ~2-3 concurrent jobs on a shared box causes severe CPU
  contention that starves the GPU too (symptom: `nvidia-smi` shows ~0% util despite
  processes clearly running, `top` shows an absurd load average). Prefix launches with
  `OMP_NUM_THREADS=<n> MKL_NUM_THREADS=<n>` (roughly `total_cores / n_concurrent_jobs`)
  preemptively, not just after noticing the slowdown.
- **Avoid `conda run -n env python ...` for long jobs.** It block-buffers stdout (a log
  can sit at 0 bytes for many minutes even with `print(flush=True)`/`python -u` — judge
  progress by aggregate CPU% instead: `ps -C python -o %cpu=,cmd= | grep <script> | awk '{s+=$1} END{print s"%"}'`,
  not the log). Worse, it has reproducibly caused a genuine hang (not just a logging
  issue) on a single-process fine-tuning job that cleared up immediately when switching
  to the env's interpreter directly. **Prefer direct env-binary invocation** —
  `/home/jtan/miniconda3/envs/<env>/bin/python3 -u script.py ...` — as the default for
  any long-running single-process job, not just as a fallback.
- **tmux for anything long-running** (survives SSH disconnects): `tmux new-session -d -s
  <name> '...; tmux wait-for -S <name>_done'`. Once launched, **don't poll it** — the
  user will say when it's done. Only check status when asked or when told it finished.

## GitHub — rules, not suggestions

- **Account**: personal `goog-msft-fb-nflx-nvda-aapl` (email `r05921008@gmail.com`,
  used for personal/coursework repos — NOT the `jamesmscluw` company account). Verify
  with `gh auth status` before any repo operation; `gh auth switch --hostname github.com
  --user goog-msft-fb-nflx-nvda-aapl` if it's not already active.
- **Commit identity for this repo**: local config (not global) is `user.name = james`,
  `user.email = j.tan@bluewx.co.jp` — already set in `.git/config`. If a fresh clone
  ever loses this (global Mac default is the *personal* `James Watanabe
  <r05921008@gmail.com>` identity), reset it rather than treating a mismatch as
  something that needs fixing retroactively — identity mismatches on past commits are
  not worth rewriting history over, this is a forward-only fix.
- **Never include Claude/Anthropic attribution anywhere pushed to GitHub** — no
  `Co-Authored-By: Claude` trailers, no mentions in commit messages, repo descriptions,
  or committed files. This overrides any default session-level attribution reminder
  that suggests otherwise; a Claude mention in a commit, unlike an identity mismatch,
  **is** worth fixing if it slips through (amend + force-push the tip commits, confirm
  with the user first since it's history-rewriting).
- **Private until after the deadline** (2026/10/25). The assignment explicitly bans
  publishing code before the deadline — don't make this repo public, don't put it
  anywhere crawlable, until the user explicitly confirms the deadline has passed. Don't
  assume it's passed from the date alone; ask or wait to be told, the way HW1's repo
  was only flipped public after an explicit go-ahead.
- **Never force-push without asking first**, even though the "fix a stray Claude
  mention" case above is the one standing exception — confirm each time regardless,
  don't treat the exception as blanket permission.
- **Commit message tone**: plain, lowercase-leaning, first person ("i"/"we"), technical
  and concise, no emoji, no em-dash/flourish formatting, minimal bold. Write it the way
  the user would write it themselves, not polished AI house style.

## Documentation conventions

- **Terse in-session chat.** Put substance in `docs/WORKLOG.md`/`docs/TODO.md`, not chat
  recaps. Once a doc is written, state its path and stop — don't also paste its contents
  into the reply. Only break silence for a genuine question or blocker.
- **Reference docs show only current state.** `docs/spec/spec.md`, `docs/EDA.md`, etc.
  should read as the correct-as-of-now configuration/findings — no "previously we did X,
  now Y" narrative embedded in the body. Revision history belongs in `git log` and in
  `docs/WORKLOG.md`'s dated entries, not in the reference docs themselves.
- **Report only measured results.** Never describe a result as "at the ceiling",
  "saturated", "limited by X" — state what was tried and the measured number; if
  interpreting *why*, label it explicitly as interpretation, not fact. An oracle/upper
  bound over a restricted method bounds that method, not the problem.
- **Explore AND exploit, both always open.** A technique/architecture that didn't help
  is a negative result for that exact configuration, not evidence it won't help
  elsewhere — every method gets evaluated on its own. But once something shows promise,
  deep-dive it (ablations, seeds, hyperparameter sweeps) before jumping to the next new
  idea — don't treat every session as a fresh explore round.
- **Web search is reference only** — verify file sizes, repo URLs, hardware/package
  claims against the actual official doc/repo before acting, especially before
  downloading or running anything on the GPU server.
- **Never speculate about a third party's methodology.** If comparing against a
  baseline/paper/reported number, only claim a causal explanation for a discrepancy if
  we've actually reproduced their exact setup — otherwise state the measured fact and
  flag the cause as unverified.

## Script reference

| Path | Purpose |
|---|---|
| `download_hw2_data.sh` | dev-only convenience — pulls the TA's provided data into `data/`. NOT the submission's `download.sh`. |
| `provided/retrieval.py`, `provided/evaluate.py`, `provided/eval_utils.py` | TA's fixed harness, reference copies, never modified, not submitted. |
| `retrieval_utils.py` | the only file we implement/submit — `retrieve_and_rerank()`. |
| `requirements.txt` | submission-root file, keep in sync with what's actually used. |
| `code/` | training/analysis/report-figure scripts, per the submission's `code/script` requirement. |

## Still to do before any of the above `download.sh`/`run.sh`/`report.pdf` rules matter

See `docs/TODO.md` — survey prompt not yet run, no baseline implementation yet. Don't
skip straight to packaging/submission mechanics before there's a working, evaluated
strategy; that was a real mistake flagged on HW1 (moved to submission-adjacent topics
before the model reliably cleared its target tier) — don't repeat it here.
