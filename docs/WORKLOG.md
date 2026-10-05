# ADL HW2 — experiment log

## 2026-10-05 — Setup: spec intake, data download, EDA

- Read `ADL2026-HW2.pptx.pdf` (ground truth) + cross-checked the Qwen-extracted
  `.pptx.md` against it page by page — no discrepancies found. Consolidated into
  `docs/spec/spec.md`.
- Pulled the actual provided files from the TA's Google Drive folder (link embedded in
  the PDF, page 3) via `download_hw2_data.sh` / `gdown --folder`. Matches the slides'
  stated sizes exactly: train 4,830 dialogues / 50,000 products, public 1,000 dialogues
  / 75,000 products.
- Ran EDA directly on the real files (not just the slide numbers) — see `docs/EDA.md`.
  Headline finding: the product pool is general-domain (6 roughly-even top-level
  categories: Sports & Outdoors, Pet Supplies, Toys & Games, Home & Kitchen, Beauty,
  Beauty & Personal Care), not beauty-specific despite the `beauty-*` dialogue-ID
  prefix, and 500/1000 public-test gold targets are products never seen in the train
  pool at all — a retriever/reranker needs to generalize from product *text*, not
  memorize train-pool identities.
- Set up this repo (`goog-msft-fb-nflx-nvda-aapl/csie5431-adl-hw2`, private) and
  `CLAUDE.md` with project conventions (GPU server gotchas, GitHub rules, working
  style) carried over from the HW1 project, so a fresh session has full context without
  re-deriving it.
- Next: write the deep-research survey prompt (incorporating the EDA findings above),
  run it externally (Claude/Gemini/Perplexity), then start the candidate-method backlog
  the same way HW1 did — see `docs/TODO.md`.
