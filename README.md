# Width vs depth for fact storage in small language models

Three small preregistered experiments (E02, E03, E04) that ask one question:

**At the same number of non-embedding parameters, does a wide-and-shallow or a narrow-and-deep transformer store
more facts? And if wide wins, is that the wide stream itself, or the bigger word table that comes with width?**

The models are about 21M parameters (13.2M of them non-embedding), each trained from scratch on 40M tokens of
general text with 1,920 synthetic facts mixed in. Every experiment was preregistered, with its predictions written
down, before any of its runs.

## The answer at this scale

Wide stores more. On average, about **half** of its lead over deep comes from its bigger tied word table and
**half** from the wide stream. The three seeds disagree a lot on that split. The middle shape's lead over deep was
all table.

Facts stored (mean of 3 seeds, of 1,920) / general held-out loss (nats):

| Experiment | wide 512×4 | mid 384×7 | deep 256×16 | wide − deep | Verdicts |
|---|---|---|---|---|---|
| **E02**, one learning rate for all | 251 / 4.228 | 216 / 4.237 | 131 / 4.527 | +121 | wide > deep; wide ≈ mid; mid ≈ deep |
| **E03**, each shape at its own tuned LR | 323 / 4.218 | 214 / 4.237 | 120 / 4.396 | +203 | wide > mid > deep |
| **E04**, all shapes read the same 256-wide table | 228 / 4.273 | 150 / 4.294 | 120 / 4.396 | +108 | wide > mid; wide > deep; mid ≈ deep |

- **E02 → E03:** fair learning-rate tuning made deep train more steadily, but it didn't make it store more. Wide
  gained the most.
- **E03 → E04:** with every shape reading one 32,768 × 256 tied table, wide still beats deep on every seed
  (+162, +128, +35).
  - Its lead dropped to 53% of E03's on average; by seed it kept 101%, 73% and 13%.
  - Mid fell to a tie with deep.
- **Loss:** with equal tables, wide still beats deep by 0.12 nats and trains in about two thirds of the time. The
  smaller table cost wide and mid about 0.055 nats each.
- **Only well-repeated facts are stored.** Facts seen 1 or 4 times stay at the guessing floor (about 4%). The
  differences are all in facts seen 16 or 64 times; at 64 exposures recall is 34% (wide), 18% (mid) and 12% (deep)
  in E04.

The full write-ups, with setup, predictions, decision rule, tables and limits, are in `docs/`:
[E02](docs/E02_width_vs_depth_facts.md), [E03](docs/E03_fair_lr_rematch.md) and
[E04](docs/E04_equal_embedding.md).

## What this is, and what it isn't

- **A short-budget result.** No model is near full: facts are seen at most 64 times. So this measures how fast each
  shape picks up facts from limited exposure, not how many facts it could hold if trained to saturation. The
  saturated regime may behave differently.
- **Pilot scale.**
  - Three seeds, one 40M-token budget, one text source and one tokenizer.
  - A muP-style learning-rate rule: hidden matrices are scaled by 256/width, embeddings use the base LR.
  - Tied embeddings throughout.
- **E04 changes three things at once for wide and mid:** the table size, the output rank (256) and about 27% of the
  forward compute. It can't separate them. All three now match deep, which was the point.
- **Our own context.** We ran this to check whether growing our 298M model ("H1") by depth only was a bad idea for
  knowledge. H1's own fact recall turned out to be limited by exposure, not room, so this stays a mechanism result.

## How it was run

This method is the reusable part:

1. **Preregister.** Write the question, setup, measures, predictions and decision rule before any run. Then freeze
   the config by sha256 (`python -m recipe freeze E04`).
2. **Pair by seed.** A seed fixes the init, the text order and the fact order, so arms with the same seed see
   identical data.
3. **Tune fairly.** Each shape tries 0.5×, 1× and 2× the base LR on seed 0. It picks by held-out loss, never by
   facts; an edge pick adds 0.25× or 4×. The comparison runs on seeds 1–3.
4. **Decide by a fixed rule.**
   - A shape "stores more" only if the mean gap is at least 58 facts (3% of 1,920) and every seed pair has the
     same sign.
   - A loss guard flags a winner whose loss is worse by more than 0.02 nats.
   - Otherwise it's a TIE.
5. **Reuse only identical runs.** A run is copied from an earlier experiment only when its recipe is
   byte-identical, and it's marked `reused_from`.

Predictions by both people running this (the author and Claude, the AI assistant that co-ran it) are in each
write-up's §4 and are scored in §8.

## What went wrong along the way

- **E02 used one learning rate for every shape**, which may have handicapped deep. E03 fixed that, and the answer
  held.
- **E04's first launch had a precision bug.**
  - Under bf16 autocast, the new table projection made wide and mid's whole residual stream bf16, while deep and
    every earlier run were float32.
  - We caught it from a dtype warning that never appears in the E02/E03 logs. The run was stopped about 4 minutes
    in, before any result, and fixed.
  - `test_residual_stream_stays_float32_under_autocast` pins the fix.
- **The seeds are noisy.** Wide's sd is about 48 facts, and E03's wide seed 3 (412) was an outlier that mostly
  disappeared in E04. Three seeds can't give a small p-value; these are pilot verdicts.

## Reproduce

Requirements: Python 3.12+, `torch` (MPS, CUDA or CPU), `numpy` and `tokenizers`. The runs took 19 to 29 minutes
each on an Apple-silicon Mac (MPS).

**Read the shipped results (no GPU, no data):**

```
python -m recipe report E04
```

It rebuilds `runs/recipe/E04/report.json` from the per-run results.

**Run it yourself:**

1. Pack your own text with `tools/pack_text.py`. You need a general-text training split, a general validation
   split, and a second held-out validation source named `stack`; we used held-out code.
2. Point at the data:
   - `RECIPE_TEXT_ROOT`: the packed text folder;
   - `RECIPE_TOKENIZER`: a Hugging Face `tokenizers` JSON;
   - `RECIPE_EOS`: the end-of-document id.
3. Set `RECIPE_RUNS` to an empty folder, so the shipped results aren't reused.
4. Run `python -m recipe run E03`, then `python -m recipe run E04` (E04 reuses E03's deep runs).

**What isn't included:** H1's tokenizer (32,768 tokens) and its training text are not redistributed. With another
tokenizer the table sizes change, because they scale with the vocabulary, and the absolute numbers will differ.
What should carry over is the comparison.

**Tests:** run `python -m pytest`. The model tests need torch.

## Layout

| Path | What |
|---|---|
| `recipe/proxy.py` | The proxy LM (RMSNorm, RoPE, SwiGLU, GQA, tied table, optional `emb_rank` projection), the data stream, training and probes |
| `recipe/facts.py` | Synthetic biographies: 480 people × 4 attributes, exposure tiers 1/4/16/64 |
| `recipe/loop.py` | Freeze, tuned runs, reuse, paired verdicts, report |
| `configs/E0*.json` | The frozen experiment configs |
| `runs/recipe/E0*/` | Every run's `result.json` (loss history, recall by tier), plus the LR selection, ledger, freeze record and report |
| `docs/` | The three write-ups (lightly edited for publication; see the note at the top of each) |
| `PUBLICATION_SHA256.json` | sha256 of the original result files and write-ups before publication edits |
| `tools/pack_text.py` | Pack plain text or JSONL into the token shards the proxy reads |

## Credits and license

By Johnathon Killaly, with Claude (Anthropic) as research assistant, as part of a personal learning project on
small language models. Code, configs, results and write-ups are under the MIT license.
