# ADL HW2 — TODO / backlog

## Current state (2026-10-05)

- [x] Spec read + consolidated (`docs/spec/spec.md`), PDF-verified against the Qwen OCR.
- [x] Data downloaded (`data/train`, `data/public`), EDA done (`docs/EDA.md`).
- [x] Repo + `CLAUDE.md` set up.
- [ ] Deep-research survey prompt (SOTA retrieval/reranking methods, HF models/datasets,
      what ablations, what visualizations/analysis) — in progress, see `docs/survey/`.
- [ ] Run the survey externally (Claude/Gemini/Perplexity), synthesize responses into a
      numbered backlog (same pattern as HW1's round-2 backlog).
- [ ] Baseline `retrieve_and_rerank()` implementation (something simple that runs
      end-to-end through `retrieval.py` + `evaluate.py` first, before optimizing).
- [ ] Training-data construction strategy for a bi-encoder/reranker (anchor = query,
      positive = gold + judge-acceptable alternatives if obtainable, negative sampling
      strategy — in-batch, hard negatives from BM25/dense-retrieval misses, etc.) —
      report Q1 requires this to be explained in detail either way.
- [ ] At least 3 distinct retrieval/reranking strategies tried, compared, and written
      up (report Q1 explicitly asks "how many did you try, why is the submitted one best").
- [ ] Q2 (generation analysis): Qwen2.5-1.5B-Instruct generation over retrieved specs,
      >=5 examples, >=3 different prompts compared.
- [ ] Package for submission: `download.sh` (model + external data only, <=30min @
      10MB/s, no pip install), `run.sh` (reproduces training), `README.md`
      (step-by-step, must match `run.sh`), `report.pdf`.
- [ ] Verify `run.sh`/`evaluate.py` finishes public-set retrieval within 1hr on RTX 3070
      8GB VRAM equivalent constraints (tighter than HW1's 3080Ti 10GB — mind model size).

## Open questions / risks

- `Abner0803/Amazon-Beauty-S2` ban — likely the source dataset this benchmark was built
  from. Don't use it directly; if a strategy wants external data, needs to be clearly
  something else and clearly described per the rules.
- 8GB VRAM execution ceiling is tighter than HW1 — any reranker/embedding model choice
  needs to fit comfortably alongside the fixed `Qwen2.5-1.5B-Instruct` judge if both run
  in the same pass (check whether `evaluate.py`'s judge and our own retrieval model need
  to coexist in VRAM, or run as separate processes/stages).
- Public-set retrieval 1hr budget — need to benchmark early whether a reranking stage
  (cross-encoder over candidates) is affordable at 75k pool / 1k queries, or whether it
  has to be restricted to a bi-encoder top-N shortlist before any reranking.
