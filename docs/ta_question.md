# Question for the TA: how is `run.sh` checked? (ADL2026 HW2)

**Where to send:** NTU COOL discussion forum first; otherwise `adl-ta@csie.ntu.edu.tw` with the subject prefix
`[ADL2026 HW2]` (contact section of the assignment slides).
**Suggested subject:** `[ADL2026 HW2] r13921031 - is run.sh executed during grading, and on what hardware?`

---

Hello,

I am r13921031. I have one question about `run.sh`, plus a smaller follow-up. Everything the question depends on is
quoted below so it can be answered without opening my files.

## 1. What the assignment says (quoted from the Submission and Execution-environment parts of the slides)

> `download.sh`: downloads model + external training data only. No `pip install`, no modifying the test
> environment, nothing beyond downloading. Must finish within 30 min at 10MB/s. ... Keep URLs valid 3+ weeks
> post-deadline. Executed before evaluation.
>
> `run.sh`: reproduces the training process. Must match what `README.md` describes step by step. No/empty
> `README.md` costs 2%. Unreproducible result risks more points.
>
> **Execution environment**: Ubuntu 22.04, 32GB RAM, **RTX 3070 8GB VRAM**, 20GB disk. Same conda env/packages as
> above. Public-set retrieval must finish within **1 hour**.

The slides state that `download.sh` is "executed before evaluation". They do not say whether `run.sh` is executed
by the graders, on which machine, or with what time limit.

## 2. What my submission does

`retrieve_and_rerank()` runs a fine-tuned Qwen3-Embedding-0.6B (dense) plus BM25, fuses them with reciprocal rank
fusion, then reranks the top 30 with a fine-tuned Qwen3-Reranker-0.6B. The two fine-tuned models are fetched by
`download.sh` (about 2.4GB, from Google Drive with `gdown`).

Inference, measured with your `retrieval.py` and `evaluate.py` on the public split on an NVIDIA H200 (I do not have
an RTX 3070): 449 s for retrieval, 3.45GB peak GPU memory, graded Recall@10 0.987 and MRR@10 0.908. The code also
has a wall-clock guard that lowers the passage length and rerank depth if the measured speed would exceed the
1-hour limit.

`run.sh` reproduces training in four steps (all scripts are in `code/script/`, described in `README.md`):

| step | script | what it does | measured time on a shared H200 |
|---|---|---|---|
| 1 | `cands.py` | zero-shot BM25 + Qwen3-Embedding candidates for each of the 4,830 train dialogues | about 5 min |
| 2 | `judge_labels.py` | runs the provided `Qwen2.5-1.5B-Instruct` judge (from your `evaluate.py`, gold as reference) on the top-20 candidates of every train dialogue (93,197 judgments) | about 1.5 h |
| 3 | `ft_biencoder.py` | full fine-tune of Qwen3-Embedding-0.6B, batch 128, 4 epochs | about 30-40 min |
| 4 | `ft_reranker.py` | full fine-tune of Qwen3-Reranker-0.6B, 1 epoch | about 20-30 min |

Training used an H200 (141GB). I did not measure peak training memory; full fine-tuning of 0.6B models with AdamW
is not expected to fit 8GB (not tested). No external dataset is used. The only extra supervision is the output of
the provided judge model on the provided training dialogues (step 2).

## 3. Questions

**Q1 (needed).** Is `run.sh` executed during grading?
- If **no**, i.e. it is only read and compared against `README.md`, then my current `run.sh` is fine and I will leave it.
- If **yes**, on which hardware and with what time limit? My `run.sh` needs a large-memory GPU and a few hours, so it
  would fail on the RTX 3070 8GB machine described above. Is it acceptable that `run.sh` assumes a larger GPU, or
  should it be limited to what fits the 8GB machine?
- If it is executed, in which directory will the training data (`data/train/...`) be when `run.sh` starts? Mine reads
  it from `./data` unless `HW2_DATA` is set.

**Q2 (follow-up, only matters if `run.sh` is executed).** Step 2 costs about 1.5 hours and its output is small (about
100KB: for each train dialogue, which of its top-20 candidates the provided judge accepted). Is it acceptable for
`download.sh` to also fetch these saved labels (they are derived only from the provided training data and the
provided judge) so that `run.sh` can skip step 2? Or must `run.sh` regenerate them from scratch?

Thank you.
r13921031
