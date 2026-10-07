I'll work through the six areas with targeted searches, starting with MidiTok's current state.I went through each of your six areas. Overall, the fastest route that stays inside the rules is a REMI-style token stream fed to a small HuggingFace `GPT2LMHeadModel` trained from scratch. For the bonus, an interleaved decoder-only setup reuses that same pipeline. On an H200, compute is not the bottleneck at POP909's size. Overfitting and pipeline bugs are the bigger risks. Details below, with anything I couldn't verify flagged.

## 1. Token representation and MidiTok

**Sequence lengths.** I found no published POP909-specific numbers, but Compose & Embellish gives a close proxy on Pop1K7 (transcribed pop piano). Songs average 104.5 bars, with 5,379 tokens per song for their REMI-style full performance versus 2,081 for CP. That works out to roughly 50 REMI tokens per bar and 20 CP steps per bar.

So a 32-bar piece is about **1.6k REMI tokens** or **~650 CP steps**. MIDI-like will be longer than REMI, because it adds note-off and time-shift events. POP909's three merged tracks may run denser than this, so measure on your own data before fixing the context length. Either way, a 2048-token context covers 32 bars of REMI comfortably, which removes CP's main advantage.

**Which is easiest.** REMI or REMI+ with a plain causal LM is the lowest-risk choice: one token per step, standard `generate()`, standard cross-entropy. CP needs a custom multi-head output and per-type sampling, so it isn't a drop-in `GPT2LMHeadModel`. In the original comparison, REMI+Transformer-XL tended to fall into long loops and never emitted EOS, while CP+Linear had richer structure but chose pitches more aggressively. That loop problem was partly a Transformer-XL effect. With full attention over a 2k context, you avoid most of it at 32 bars.

**Chord conditioning (the "w/ Chord" column).** The standard approach is REMI's: insert `Chord_*` tokens at beat positions. C&E uses Chord tokens covering 12 roots × 11 qualities, which may appear as often as every beat.

For POP909, build these tokens from the dataset's own annotations rather than detecting them. Each song folder includes chord annotation files (start time, end time, chord name), and beat files are provided too. MidiTok's `use_chords` instead detects chords from note onsets and durations, and enabling it slows tokenization. That is a different signal from the annotated chords your bonus Chord Accuracy is measured against.

**MidiTok status.** The PyPI package is `miditok`, and the latest release is 3.0.6.post1 (July 22, 2025), MIT-licensed, Python ≥3.9. Repo: https://github.com/Natooz/MidiTok. There have been no releases in about 14 months, so it is maintained but quiet. Pin the version.

Its quantization setting is `beat_res`: a dictionary mapping beat ranges to samples per beat; for Position-based tokenizers, the Position vocabulary size equals the maximum resolution given. So `beat_res={(0,4):4, (4,12):2}` is roughly `--pos_per_beat 4` (16th notes). Also, REMI with Program and TimeSignature tokens enabled is equivalent to REMI+.

**The big multi-track gotcha.** MidiTok separates tracks only by MIDI program number. Its preprocessing step merges tracks that share a program, and the docs warn that tracks with the same Program will be merged.

I believe all three POP909 tracks are piano (program 0) but haven't confirmed it, so check with `symusic` or `pretty_midi`. If they are, MELODY/BRIDGE/PIANO collapse into one track and you lose track identity. That is fine for unconditional generation but fatal for the bonus task.

The fix is to reassign distinct program numbers per track before tokenizing and map them back on decode. Then one_token_stream_for_programs puts a Program token before each note and serializes all tracks into one sequence. Given this, the provided `tokenization.py` may honestly be less friction. You already know its track handling, and it's what the eval scripts expect.

## 2. Architectures trainable from scratch

**HF GPT-2 from random init: yes, sufficient.** For scale, the original CP and REMI baselines all used 12 layers, 8 heads, hidden size 512, and a 2,048 feed-forward size (about 39–41M parameters), trained on about 1.6k songs. Notably, C&E later released a stage-2 checkpoint on a 12-layer GPT-2 backbone, so the authors themselves moved to vanilla GPT-2.

For POP909's ~900 songs (about half of Pop1K7), I'd go **6–8 layers × 512 hidden, context 2048**, which is roughly 20–30M parameters. Use 12-semitone transposition augmentation, dropout around 0.1, and early stopping on validation NLL. This sizing is my judgment, not a published POP909 result.

**Original repos in 2026: don't adapt them.**

- **Pop Music Transformer** (https://github.com/YatingMusic/remi) requires Python 3.6 and tensorflow-gpu 1.14.0. I'm confident TF 1.14 won't run on Hopper/CUDA 12, so treat it as reference code only.
- **CP Transformer** (https://github.com/YatingMusic/compound-word-transformer) lists Python 3.6 with madmom, miditoolkit and pytorch-fast-transformers. The CUDA-kernel `fast-transformers` dependency is the likely failure point.
- **Compose & Embellish** (https://github.com/slSeanWU/Compose_and_Embellish) recommends Python 3.8, CUDA 10.2, and a pinned fork of fast-transformers, and the model card itself calls fast-transformers outdated. Rule-wise, its headline result uses LMD pretraining, which is extra data and not allowed for you. Its "w/o pretrain" ablation is the relevant comparison point.

A from-scratch HF pipeline will take you less time than fixing any of these environments. Borrow their token designs and sampling recipes, not their code.

**C&E has a nice POP909 shortcut, if you have time.** C&E's stage 1 normally requires skyline melody extraction plus structure analysis. POP909 hands you the MELODY track and annotated chords for free. So a stage-1 "melody + chord" model plus a stage-2 "melody → accompaniment" model *is* your bonus model, reused. Treat it as a stretch ablation, not the main plan.

## 3. Training-time budget

The reported datapoints are all from much older hardware:

- The CP paper assumes a single 11 GB GPU and a willingness to train for 3 days, and reports that CP converged within a day on that single 11 GB GPU, 5–10× faster than prior models.
- C&E trained on an RTX 3090 Ti; its lead-sheet fine-tune took under half a day and its performance model about 3 days, with random 3,072-token crops at batch size 4.

**Extrapolating to an H200 (my estimate, not reported anywhere):** POP909 is about 900 songs × ~5k tokens, or about 4–5M tokens per epoch. With 12 transpositions, that's about 55M tokens per epoch. A ~30M-parameter model in bf16 with SDPA/FlashAttention should get through an epoch in minutes, so **a few hours to good convergence is realistic**. The 3-day figures came from 11–24 GB cards, tiny batches, and full-song contexts.

Expect validation loss to bottom out well before compute runs out. The real levers are augmentation (transposition), model size relative to the data, and which checkpoint you pick. C&E found that checkpoints around 0.4 training NLL sounded best, so the lowest-loss checkpoint is not automatically the best one. Budget 1–2 days total for training, including a couple of reruns, then move on to the bonus.

## 4. MusDr and SI

**MusDr** (https://github.com/slSeanWU/MusDr, MIT) is a plain repo, not a PyPI package. It was tested on Python 3.6, and H, GS and CPI natively support only the Jazz Transformer's event representation. So rely on the course's `eval_basic.py` wrapper rather than raw MusDr. Expect minor `requirements.txt` breakage on modern Python, and pin old numpy/scipy if needed.

**Why SI is slow.** SI is computed from audio. Each MIDI is rendered to audio, then a self-similarity matrix and a fitness scape plot are built from it. MusDr notes the MATLAB path needs more setup but runs faster, while the Python path needs no setup but runs slowly on longer songs.

The cost is structural. A piece with N frames has (N+1)N/2 segments, and the reference code runs a separate accumulated-score-matrix DP over each segment, giving roughly O(N⁴) total work. Even the authors only sampled 200 real songs because fitness scape plots are so costly to compute.

**The good news for you.** The standard setup uses a 1 Hz frame rate and SI timescales of 3–8, 8–15, and 15+ seconds. A 32-bar piece at ~80 BPM is only ~95 s, so N ≈ 95, versus ~240 for a full song. That's roughly 40× cheaper than the full-song case. Run it on your ~20 generated pieces per configuration plus a matched set of real 32-bar excerpts, with `-j` parallelism, and it should be tractable.

If MusDr's bundled Python is still slow, `libfmp` (https://meinardmueller.github.io/libfmp/) has the same routines. It's what another group used to compute SI. I haven't benchmarked it.

**Reference values for real pop piano** (Pop1K7, not POP909; use them as sanity bounds): H1 1.96, H4 2.53, GS 75.5%, and SI (short/mid/long) 43.8/43.1/34.8%. Those SI numbers use C&E's 4–12/12–32/32+ second windows, not the 3/8/15 defaults, so they don't transfer directly. Compute your own real-data baseline with the course scripts. It's the most defensible reference for the report.

## 5. Sampling and inference configurations

Reported settings to start from:

- Pop Music Transformer's demo used temperature 1.2 with top-k 5.
- C&E grid-searched τ ∈ {1.2, 1.3, 1.4} and p ∈ {.95, .97, .98, .99}, picking the combination whose generations best matched validation-set perplexity. It chose τ=1.2, p=.97 for lead sheets and τ=1.1, p=.99 for full performances, by ear.
- From memory and unverified: the CP repo uses per-token-type temperature/top-p (e.g., a very high temperature for velocity). Check `main-cp.py` before citing it.

Matching perplexity to validation data is a principled, cheap way to choose configs, and it gives your report a justification beyond "sounded good."

**On "loss variants":** these need retraining, so they're more expensive than sampling changes. A cheap, sourced alternative third axis is **checkpoint choice**: an early checkpoint vs. the min-validation-loss one vs. an overtrained one, motivated by C&E's ~0.4 NLL finding above.

My suggested ≥3 configs:

1. top-k (k≈5–10, τ≈1.2)
2. nucleus (p≈0.95–0.97, τ≈1.1–1.2)
3. the better of those two at a different checkpoint, or with label smoothing ≈0.1 if you can afford one extra training run

Report H4 and GS for each. The pattern to look for is that more conservative sampling lowers H and raises GS.

## 6. Bonus: interleaved decoder (A) vs. seq2seq (B)

**Recommendation: A, with bar-level interleaving and a one-bar melody lookahead.** It reuses your training pipeline, tokenizer, and `generate()` loop, and there's a proven recipe. C&E's stage 2 interleaves one-bar segments of the condition and the target, marks them with Track_M / Track_X tokens, and at inference moves to the next bar whenever the model emits Track_M.

The inference loop is:

1. Force-feed melody bar *i* (and *i+1* if using lookahead).
2. Sample piano tokens until the track-switch token appears.
3. Inject the next melody bar.
4. Repeat for all 16 bars.

The Anticipatory Music Transformer is a related design: a decoder-only model that interleaves condition tokens with generated tokens to train a conditional generator.

**Why lookahead matters.** The streaming-accompaniment paper finds that more future visibility improves coherence, and naive maximum-likelihood training is insufficient when future context is unavailable. In your offline setting, the cheap fix is to interleave melody bar *i+1* before piano bar *i*. That paper is audio-to-audio and real-time (code: https://github.com/lukewys/stream-music-gen), so take the design principle rather than the code.

**Option B** (`T5ForConditionalGeneration` from scratch) gets full-melody visibility for free, and it's a clean seq2seq setup. The cost is a second data pipeline and a second training run with its own hyperparameters. It's the more principled design, but riskier under a deadline.

PopMAG is architecturally B: an encoder-decoder built on Transformer-XL with a lead-sheet encoder, using the MuMIDI representation, which models a note's multiple attributes in a single step and adds extra long-context memory. However, it isn't open source; one reproduction on LMD got Chord Accuracy around 0.47, so ~0.45–0.5 is a realistic ballpark for your Chord Accuracy.

**Suggested order:** get A working end-to-end first. Try B only if time remains, which also gives the report a natural A-vs-B comparison.

## Suggested timeline

| Days | Work |
|---|---|
| 1–3 | Tokenization (verify track programs) plus real-data metric baselines, including SI on 32-bar excerpts |
| 4–6 | GPT-2 unconditional training with transposition augmentation; sampling configs |
| 7–8 | Prompt continuation |
| 9–14 | Bonus option A |
| 15–20 | Optional B or chord-token ablation, then the report |

Start SI runs early, but on 32-bar pieces they shouldn't be the crisis the assignment warns about.