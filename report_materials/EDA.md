# HW2 data — EDA findings

Computed directly from the downloaded `data/{train,public}/{dialogues,products}.jsonl`
(see `docs/spec/spec.md` for sizes/counts already stated in the slides — this doc only
covers what's NOT stated there).

## Dialogues

- `dialogue_id`, `conversation` (list of `{role, content}`), `target` (gold `parent_asin`).
- Always ends on a `user` turn — the bot's actual recommendation turn is never included;
  `eval_utils.build_eval_query` concatenates all turns verbatim as the query text.
- Turn count is bimodal, exactly 3 or 5 (roughly 50/50 split in train), never anything else.
- Query text length (all turns concatenated): train mean 411 chars / median 386, range
  154-842. First user turn alone averages 117 chars; the final (pre-recommendation) turn
  alone averages 84 chars — later turns are shorter/more pointed, not more verbose.
- Train: 4,830 dialogues but only 4,547 unique targets (283 targets reused by exactly 2
  dialogues each, max 2 — minor, not worth special handling).
- Public: 1,000 dialogues, 1,000 unique targets (no reuse).

## Products

- `parent_asin`, `title`, `description` (free text — description embeds structured
  Amazon metadata as prose: category path, dimensions, weight, brand, best-sellers
  rank, etc., not separate fields).
- Product text (title+description) length: train pool mean 1,720 chars / median 1,691,
  range 24-6,516. No empty descriptions in the train pool (50,000/50,000 have one).
- 65 exact-duplicate titles across 133 products in the train pool (color/size variants
  of the same listing, e.g. "Freeman Cucumber Facial Peel-Off Mask - 6 oz") — a small
  near-duplicate-candidate problem for any retriever, not large enough to dominate.

## Pool structure (important for retrieval strategy)

- Train pool (50,000) is an **exact subset** of the public pool (75,000) — the public
  pool adds exactly 25,000 new items on top, nothing removed/changed.
- **500 of the 1,000 public dialogues target one of the 25,000 NEW items** — i.e. half
  the public eval targets a product that never appeared anywhere during training. A
  retriever/reranker that overfits to memorized train-pool product identities (rather
  than generalizing from product *text*) will cap out around 50% recall no matter how
  well it fits train. This is the single most important structural fact for picking a
  strategy — favor embedding/reranking methods that generalize over the text, not
  anything that treats the product catalog as a fixed closed set.
- The product pool is **not** single-domain despite dialogue IDs being prefixed
  `beauty-*`. Top-level categories in the train pool split roughly evenly 6 ways:
  Sports & Outdoors (9,922), Pet Supplies (9,900), Toys & Games (9,869), Home & Kitchen
  (9,790), Beauty (6,148), Beauty & Personal Care (3,852). Gold targets are NOT
  Beauty-only either — they're spread across all 6 categories (train: Beauty 2,113,
  Pet Supplies 571, Toys & Games 565, Sports & Outdoors 570, Home & Kitchen 568,
  Beauty & Personal Care 436). Public test gold targets are almost exactly stratified
  200/200/199/199/199 across the 5 main categories. So this is a **general-domain**
  product-retrieval task, not a beauty-specialist one — a general-purpose sentence/text
  embedding model is a more natural starting point than anything beauty-domain-tuned,
  and the other 5 categories' products mostly function as large-scale "easy negative"
  distractors relative to any single conversation's true category.

## Open questions this EDA doesn't answer (follow up with real experiments, not guesses)

- Whether `Abner0803/Amazon-Beauty-S2` (the banned external dataset) is literally the
  superset this benchmark was carved from, or just adjacent — unverified, don't assume
  either way beyond "don't touch it."
- Private-test pool adds another 25,000 new items on top of public (100,000 total per
  the slides) — same generalization pressure expected to continue, unconfirmed since
  private data isn't available.
- No multi-label ambiguity signal found in dialogues (`target` is always a single ID) —
  whether a conversation could reasonably map to >1 "equally gold" product is an
  `evaluate.py`-judge question (graded relevance), not visible in the raw data itself.
