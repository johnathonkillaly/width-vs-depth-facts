# E04: Equal word table (E03's three shapes, each reading the same 256-wide tied table)

> **Published copy.** Lightly edited for publication: internal machine and job-queue references removed, paths made
> repo-relative, and "the owner" replaced by the author's name. The science text (question, setup, predictions, rule,
> results) is unchanged. The private original's sha256 is in `PUBLICATION_SHA256.json`. "Claude" in the prediction
> tables is the AI assistant (Anthropic) that co-ran these experiments; "Johnathon" is the author.

**Status:** DONE 11-09 00:52, report in §8 (wide vs deep WIDE_STORES_MORE, +108; R = 0.53; mid vs deep TIE).
PREREGISTERED 11-08 evening (frozen 19:52, config sha256 `92b9531e`), before any E04 run. Johnathon approved it on 11-08 ("e04 is approved,
let's see where results land").
Frozen by `python -m recipe freeze E04` (sha256 in `runs/recipe/E04/FROZEN.json`).
**Config:** `configs/E04.json`. **Code:** `recipe/proxy.py` (`emb_rank`) and `recipe/loop.py`
(reuse from a tuned experiment). Tests: `tests/test_recipe_proxy.py` and `tests/test_recipe_loop.py`.

### 1. Question

In E03 every shape had about 13.2M non-embedding parameters and its own tuned learning rate. Wide (512×4) stored the
most facts (323 of 1,920), then mid (384×7, 214), then deep (256×16, 120). But the tied word table grows with width:
16.8M, 12.6M and 8.4M numbers. The fact ranking follows the table size exactly. **Is wide's lead in the wide stream,
or in the bigger word table?**

### 2. Setup

Everything is as in E03 (shapes, text, facts, 40M tokens, optimizer, schedule, tuning rule, seeds) except the word
table of wide and mid.

**The equal table.** Every arm reads the same 32,768 × 256 table. Wide and mid widen it to their stream width with
one 256 × d matrix P that is tied both ways: in, a token's row e becomes e·P; out, the final hidden state h gives
logits (h·Pᵀ)·Eᵀ. So every arm has the same table size and the same rank-256 output.
- P starts with orthonormal rows scaled by √(d/256). That makes the input scale per dimension and the starting logit
  scale match the full-width table, so only the width changes.
- P is counted as embedding (131,072 for wide, 98,304 for mid). It gets the embedding learning rate, not the
  width-scaled one.
- **Deep is unchanged.** At width 256 it already has this table, so its E04 recipe is byte-identical to E03's.

| Arm | Table + P (E03 → E04) | Non-embedding |
|---|---|---|
| wide 512×4 | 16,777,216 → 8,519,680 | 13,242,880 |
| mid 384×7 | 12,582,912 → 8,486,912 | 13,209,216 |
| deep 256×16 | 8,388,608 → 8,388,608 | 13,246,720 |

**Tuning (seed 0) and comparison (seeds 1, 2, 3):** E03's rule, unchanged.
- Each arm is tried at 0.5×, 1× and 2× the base LR. The pick is the lowest general held-out loss, never facts.
- An edge pick adds 0.25× or 4×, and the pick is made again.
- The finals run at each arm's pick, on seeds that exclude the tuning seed.

**Reuse:** a run whose recipe is byte-identical to a finished E03 run is copied, not retrained, and is marked
`reused_from`. That now includes E03's tuning-sweep runs and its finals at their picked multiplier.
- In practice every deep run is reused: the sweep at 0.25×/0.5×/1×/2×, and seeds 1–3 at 0.5×. This was checked before
  the freeze.
- The default model path (no `emb_rank`) was tested parameter-for-parameter and logit-for-logit against the pre-E04
  `proxy.py`.

**Cost:** 6 to 8 tuning runs plus 6 comparison runs, all wide or mid. Each is about 17 to 25 minutes on the Mac GPU;
the smaller output layer makes them a little faster than E03's. That's roughly 4 to 6 hours. It yields the GPU to a
higher-priority job like E02 and E03, and the runner waits and retries after a yield.

**Code fix before any result (11-08, 19:57).** The first launch (19:53) was stopped about 4 minutes into its first
tuning run, before any result was written.
- **The bug:** the bf16 autocast turned e·P into bf16, so wide and mid's whole residual stream ran in bf16. A plain
  lookup keeps it float32, as in deep and in all of E02/E03.
- **How it was caught:** a dtype warning that never appears in the E02/E03 logs.
- **The fix:** e·P is now computed outside autocast, in float32. `test_residual_stream_stays_float32_under_autocast`
  pins it, and it fails on the code at `bd844ad`.
- The config and its freeze hash are unchanged. Nothing from the stopped run is used.

### 3. Measures

- **Primary:** `facts_seen` (of 1,920).
- **Guard:** `general` held-out loss.
- **Also reported:** recall by tier, `facts_unseen`, `stack` loss, wall-clock time, the picked multiplier, and every
  tuning run's loss.

**Secondary readings**, computed from the same files:
- **Share kept, R:** E04's mean wide − deep fact gap divided by E03's (+203.0). R ≈ 1 means the stream did it; R ≈ 0
  means the table did.
- **Table cost:** E04 wide minus E03 wide, per seed, and the same for mid. The seeds give the same data and fact order
  in both experiments. These use the same margin and sign rule as the primary.

### 4. Predictions (written before any run)

| # | Who | Prediction | Would refute it |
|---|---|---|---|
| O1 | Johnathon | no prediction given ("let's see where results land") | — |
| P1 | Claude | wide vs deep is still WIDE_STORES_MORE | TIE or DEEP_STORES_MORE |
| P2 | Claude | but the gap shrinks: mean wide − deep below E03's +203 (0 < R < 1) | mean d ≥ 203 |
| P3 | Claude | the smaller table costs wide facts: E04 wide below E03 wide on each of seeds 1–3 | any seed where E04 wide ≥ E03 wide |
| P4 | Claude | the rank-256 output costs wide loss: E04 wide's general loss is ≥ 0.02 nats worse than E03 wide's 4.218 | worse by < 0.02 |
| P5 | Claude | mid vs deep becomes a TIE | any STORES_MORE verdict between them |

### 5. Decision rule

E03's rule, unchanged, on seeds 1, 2, 3:
- **A_STORES_MORE** if mean d ≥ 58 facts (3% of 1,920) and every seed pair shares the sign. **TIE** otherwise.
- **`_BUT_WORSE_LOSS`** is added if the winner's general loss is worse by more than 0.02 nats.

This is a pilot verdict: three seeds can't give a small p-value.

### 6. Known limits

- **Three things change together for wide and mid:** the table size, the output rank and part of the compute. At
  this scale the output layer is about half of wide's forward multiply-adds, so they drop about 27%. E04 can't
  separate the three. All three now match deep, which is the point.
- **P adds geometry:** it's under 1% of the non-embedding count, but it changes the input geometry. A table that is
  simply narrower is impossible without some projection when the stream is wider.
- **Pilot scale:** tuning uses one seed, and the comparison three seeds, at one 40M-token budget with small proxies.
  H1's own fact recall is limited by exposure, not room (the capacity map). So this is a mechanism result, not a
  change to H1's training plan.

### 7. What happens after

| Outcome (wide vs deep) | Reading | Next |
|---|---|---|
| WIDE_STORES_MORE, R near 1 | the wide stream itself stores facts at this scale | a widen-vs-deepen growth proxy, both grown from one small parent, as input to the big choice |
| WIDE_STORES_MORE, R well below 1 | both the stream and the table contribute | report the split; same growth proxy next |
| TIE | the bigger table explained the lead | depth-only growth (stage A) is fine for facts; the "new layer for new facts" idea should consider the table and output head |
| DEEP_STORES_MORE | reversal | report it as a reversal of E02/E03 |

### 8. Report

Filled 11-09 from `runs/recipe/E04/report.json` (`python -m recipe report E04` reproduces it byte for byte, sha256
`7e367470`). The run started 19:57 on 11-08 and finished 00:52 on 11-09 with no yields. All deep runs, tuning and
finals, are E03 runs reused unchanged.

**Stage 1, tuning (seed 0, general held-out loss; lower is better):**

| Arm | 0.25× | 0.5× | 1× | 2× | Pick |
|---|---|---|---|---|---|
| wide 512×4 | 4.354 | **4.269** | 4.335 | 4.583 | 0.5× (base 1.5e-3) |
| mid 384×7 | 4.412 | **4.295** | 4.331 | 4.537 | 0.5× (base 1.5e-3) |
| deep 256×16 | 4.541 | **4.377** | 4.710 | 4.662 | 0.5× (reused from E03) |

All three first picked 0.5×, the low edge, so 0.25× was added. It was worse for all three, so no pick is on an edge.
Mid moved from 1× in E03 to 0.5× here.

**Stage 2, comparison (seeds 1, 2, 3 at each arm's pick):**

| Arm | facts_seen by seed (1, 2, 3) | Mean / 1,920 | Tier 1 / 4 / 16 / 64 recall | General loss (mean; seeds) | Train time |
|---|---|---|---|---|---|
| wide 512×4 | 260, 251, 173 | 228.0 | 3.8 / 3.4 / 6.9 / 33.5% | 4.273 (4.267, 4.276, 4.277) | 1,159 s |
| mid 384×7 | 138, 185, 127 | 150.0 | 3.5 / 4.4 / 5.4 / 17.9% | 4.294 (4.296, 4.286, 4.301) | 1,355 s |
| deep 256×16 | 98, 123, 138 | 119.7 | 4.0 / 3.5 / 5.1 / 12.3% | 4.396 (4.375, 4.414, 4.398) | 1,717 s |

| Pair | d by seed | Mean d (margin 58) | Verdict |
|---|---|---|---|
| wide vs mid | +122, +66, +46 | +78 | **WIDE_STORES_MORE** (wide's loss also better, by 0.021) |
| wide vs deep | +162, +128, +35 | +108 | **WIDE_STORES_MORE** (wide's loss better by 0.12) |
| mid vs deep | +40, +62, −11 | +30 | **TIE** (mid's loss better by 0.10) |

`facts_unseen` is 0 for every run, as in E02 and E03.

**Secondary readings:**
- **Share kept, R = 108.3 / 203.0 = 0.53.** By seed it is 162/160 = 1.01, 128/175 = 0.73 and 35/274 = 0.13. So the
  seeds disagree a lot. Most of E03's seed-3 gap was the table, while seeds 1 and 2 kept most of theirs.
- **Table cost (E04 minus E03, same seeds):**

  | Arm | Facts by seed (1, 2, 3) | Mean | By the rule | Loss |
  |---|---|---|---|---|
  | wide | +2, −47, −239 | −95 | TIE (seed 1 has the other sign) | 0.055 worse (4.218 → 4.273) |
  | mid | −33, −9, −150 | −64 | the bigger table stores more | 0.057 worse (4.237 → 4.294) |

**Predictions:**
- O1 (Johnathon): no prediction was given.
- P1 **confirmed**: wide vs deep is still WIDE_STORES_MORE.
- P2 **confirmed**: the gap shrank from +203 to +108 (R = 0.53).
- P3 **refuted, narrowly**: E04 wide was below E03 wide on seeds 2 and 3, but not on seed 1 (260 vs 258).
- P4 **confirmed**: wide's loss is 0.055 nats worse than E03's 4.218.
- P5 **confirmed**: mid vs deep is now a TIE.

**Reading.**
- **The stream and the table both mattered, about half each on average.** With every arm reading the same 256-wide
  table, wide still stores more than deep on every seed and clears the margin. That is about half of E03's lead.
  The split is uncertain, since the per-seed R runs from 0.13 to 1.01.
- **The table explained mid's lead.** Mid fell from MID_STORES_MORE in E03 to a TIE with deep, and it lost facts on
  every seed. The ranking is now wide > mid ≈ deep.
- **On loss, the wide stream is the main effect.** The narrower table and its rank-256 output cost wide and mid about
  0.055 nats each, but wide still beats deep by 0.12 nats with equal tables, in two thirds of deep's time.
- **Only well-repeated facts are stored.** Tiers 1 and 4 are still at the guessing floor (about 4%). The whole
  difference is at tiers 16 and 64; at tier 64 recall is 33.5% (wide), 17.9% (mid) and 12.3% (deep).
- **Limits.**
  - For wide and mid, table size, output rank and about 27% of the compute changed together (§6), and E04 can't
    separate them.
  - This is a pilot verdict: three seeds, one 40M-token budget, small proxies.
  - E03's wide seed 3 (412) was an outlier, and most of it went away here. Seed noise in facts is large (wide's sd
    is 48).
  - The stopped 19:53 launch was never used (see §2).

**Next (§7, the "WIDE_STORES_MORE, R well below 1" row).**
- Report the split, done above.
- The next test is a widen-vs-deepen growth proxy, both grown from one small parent, as input to the big choice. It
  is not preregistered yet, and it needs Johnathon's approval.
- This doesn't change H1's training plan. H1's own fact recall is limited by exposure, not room.
