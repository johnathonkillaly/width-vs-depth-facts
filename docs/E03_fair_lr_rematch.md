# E03: Fair-tuning rematch of E02 (each shape at its own best learning rate)

> **Published copy.** Lightly edited for publication: internal machine and job-queue references removed, paths made
> repo-relative, and "the owner" replaced by the author's name. The science text (question, setup, predictions, rule,
> results) is unchanged. The private original's sha256 is in `PUBLICATION_SHA256.json`. "Claude" in the prediction
> tables is the AI assistant (Anthropic) that co-ran these experiments; "Johnathon" is the author.

**Status:** PREREGISTERED 11-08 morning, before any E03 run. Johnathon approved it on 11-08 ("please
update the learning-rate rematch, only fair, although I'm not sure that would change the results much").
Frozen by `python -m recipe freeze E03` (sha256 in `runs/recipe/E03/FROZEN.json`).
**Config:** `configs/E03.json`. **Code:** `recipe/loop.py` `run_tuned` (tests in `tests/test_recipe_loop.py`).

### 1. Question

E02 found WIDE_STORES_MORE than deep (251 vs 131 facts of 1,920), but deep also had the worst loss (4.53 vs 4.23),
and its loss varied by 0.41 nats across seeds. Under E02's muP-style rule deep's hidden matrices got the highest
learning rate. Is "wide stores more" real, or did deep just train worse at a rate that suited wide?

### 2. Setup

Everything is as in E02 (same three shapes, text, facts, 40M tokens, optimizer, schedule) except the learning rate.

**Stage 1, tuning (seed 0):** each shape at 0.5×, 1× and 2× the E02 base LR (3e-3, still muP-scaled by 256/width).
The pick is the multiplier with the **lowest general held-out loss**. Facts are never used to choose, so tuning can't
fit the answer. If the pick is at an edge of the grid, one more point beyond it is tried (0.25× or 4×), and the pick
is made again. Picks are written to `runs/recipe/E03/selection.json`.

**Stage 2, comparison (seeds 1, 2, 3):** each shape at its picked multiplier. The tuning seed is excluded, so the
pick can't flatter its own seed.

**Reuse:** a run whose recipe is byte-identical to a finished E02 run with the same seed is copied, not retrained,
and is marked `reused_from`. That covers every 1× run at seeds 0, 1 and 2 (same code, config and data order).

**Cost:** 6 to 9 new tuning runs plus 3 to 9 new comparison runs, at about 22 to 28 minutes each on the Mac GPU,
so roughly 4 to 7 hours. It yields the GPU to a higher-priority job, like E02.

### 3. Measures

Same as E02: primary `facts_seen` (of 1,920), guard `general` held-out loss, plus recall by tier, `facts_unseen`,
`stack` loss and wall-clock time. New: the picked multiplier per shape, and the loss of every tuning run.

### 4. Predictions (written before any run)

| # | Who | Prediction | Would refute it |
|---|---|---|---|
| O1 | Johnathon | the result doesn't change much: wide vs deep is still WIDE_STORES_MORE | TIE or DEEP_STORES_MORE |
| P1 | Claude | deep's pick is below 1× (0.5× or 0.25×) | deep picks 1× or higher |
| P2 | Claude | deep's loss gap to wide shrinks from 0.30 to under 0.10 nats | gap ≥ 0.10 |
| P3 | Claude | deep's seed-to-seed loss spread at its pick is under 0.10 nats (E02: 0.41) | spread ≥ 0.10 |
| P4 | Claude | wide still stores more than deep (agrees with O1), but by less than E02's mean +121 facts | gap ≥ 121, or verdict not WIDE |
| P5 | Claude | wide vs mid stays a TIE | any STORES_MORE verdict between them |

### 5. Decision rule

E02's rule, unchanged, on the comparison seeds 1, 2, 3: A_STORES_MORE if mean d ≥ 58 facts (3% of 1,920) and
every seed pair shares the sign; TIE otherwise; `_BUT_WORSE_LOSS` if the winner's general loss is worse by more
than 0.02 nats. A pilot verdict (three seeds can't give a small p-value).

### 6. Known limits

- Tuning uses one seed, and deep's E02 seed noise was large, so a pick can be off by one grid step.
- The embedding confound stays: wide's tied embedding is twice deep's (16.8M vs 8.4M). E03 fixes only the learning
  rate. If wide still wins, the embedding is the next suspect.
- One learning rate is tuned. Warmup, depth-scaled init and batch size stay at E02's values.

### 7. What happens after

| Outcome | Next |
|---|---|
| WIDE_STORES_MORE again | E02 was not a tuning artifact. Next: equalize the embedding (E04), then a widen-vs-deepen growth test |
| TIE | E02's gap was mostly training quality. Depth-only growth (Stage A) is fine for facts at this scale |
| DEEP_STORES_MORE | depth wins when tuned. Report it as a reversal of E02 |

### 8. Report

Filled 11-08 from `runs/recipe/E03/report.json` (`python -m recipe report E03` reproduces it byte for byte).
Tuning and finals finished 14:10. Mid's seeds 1 and 2 are E02 runs reused unchanged, because mid picked 1×.

**Stage 1, tuning (seed 0, general held-out loss; lower is better):**

| Arm | 0.25× | 0.5× | 1× | 2× | Pick |
|---|---|---|---|---|---|
| wide 512×4 | 4.330 | **4.199** | 4.227 | 4.385 | 0.5× (base 1.5e-3) |
| mid 384×7 | — | 4.287 | **4.231** | 4.506 | 1× (base 3e-3) |
| deep 256×16 | 4.541 | **4.377** | 4.710 | 4.662 | 0.5× (base 1.5e-3) |

Wide and deep first picked 0.5×, the low edge, so 0.25× was added. It was worse for both, so no pick is on an edge.

**Stage 2, comparison (seeds 1, 2, 3 at each arm's pick):**

| Arm | facts_seen by seed (1, 2, 3) | Mean / 1,920 | Tier 1 / 4 / 16 / 64 recall | General loss (mean; seeds) | Train time |
|---|---|---|---|---|---|
| wide 512×4 | 258, 298, 412 | 322.7 | 3.9 / 4.1 / 10.1 / 49.1% | 4.218 (4.204, 4.209, 4.242) | 1,375 s |
| mid 384×7 | 171, 194, 277 | 214.0 | 3.4 / 4.2 / 7.7 / 29.3% | 4.237 (4.226, 4.254, 4.230) | 1,484 s |
| deep 256×16 | 98, 123, 138 | 119.7 | 4.0 / 3.5 / 5.1 / 12.3% | 4.396 (4.375, 4.414, 4.398) | 1,717 s |

| Pair | d by seed | Mean d (margin 58) | Verdict |
|---|---|---|---|
| wide vs mid | +87, +104, +135 | +109 | **WIDE_STORES_MORE** (wide's loss also better, by 0.018 on average) |
| wide vs deep | +160, +175, +274 | +203 | **WIDE_STORES_MORE** (wide's loss better by 0.18) |
| mid vs deep | +73, +71, +139 | +94 | **MID_STORES_MORE** (mid's loss better by 0.16) |

`facts_unseen` is 0 for every run, as in E02.

**Predictions:**
- O1 (Johnathon) **confirmed**: wide vs deep is still WIDE_STORES_MORE, and by more than before.
- P1 **confirmed**: deep picked 0.5×.
- P2 **refuted**: deep's loss gap to wide shrank from 0.30 to 0.18 nats, not to under 0.10.
- P3 **confirmed**: deep's seed spread is 0.04 nats (E02: 0.41).
- P4 **refuted**: wide's lead over deep grew from +121 to +203 facts instead of shrinking.
- P5 **refuted**: wide vs mid is no longer a tie.

**Reading.**
- **Fair tuning made deep train more steadily but did not make it store more.** Deep's loss improved by 0.13 nats
  and its seed noise mostly went away. Its facts went from 131 to 120, which is within noise. So E02's deep result
  was not just a bad learning rate.
- **Wide gained the most from tuning.** At 0.5× it went from 251 to 323 facts, with the best loss of the three. At
  this budget the ranking is now strict: wide > mid > deep on facts, and also on loss.
- **Only well-repeated facts are stored.** Tiers 1 and 4 still sit at the guessing floor (about 4%). The whole
  difference comes from facts seen 16 or 64 times; at tier 64 recall is 49% (wide), 29% (mid) and 12% (deep).
- **The embedding confound is untouched.** The arms have nearly equal non-embedding parameters (13.2M each), but
  wide's tied embedding is 16.8M, mid's 12.6M and deep's 8.4M. The fact ranking follows embedding size exactly, so
  E03 cannot say whether the win comes from a wider stream or a bigger embedding table.
- **Limits.**
  - This is a pilot verdict: three seeds, one 40M-token budget, small proxies.
  - Wide's seed 3 (412) is high, but every seed pair clears the 58-fact margin on its own.
  - Deep's 1× tuning point is the E02 seed-0 outlier (4.71). Even without it, 0.5× beats 2× and 0.25×.

**Next (§7, WIDE_STORES_MORE row).** E02 was not a tuning artifact. The next test is E04: equalize the embedding
(same embedding size, or count it in a fixed total) and rerun wide vs deep at these tuned rates. A
widen-vs-deepen growth test comes only after that. E04 is not preregistered yet; it needs Johnathon's approval.
