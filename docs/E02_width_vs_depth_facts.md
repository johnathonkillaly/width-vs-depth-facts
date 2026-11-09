# E02: Width or depth for storing facts? (small proxy)

> **Published copy.** Lightly edited for publication: internal machine and job-queue references removed, paths made
> repo-relative, and "the owner" replaced by the author's name. The science text (question, setup, predictions, rule,
> results) is unchanged. The private original's sha256 is in `PUBLICATION_SHA256.json`. "Claude" in the prediction
> tables is the AI assistant (Anthropic) that co-ran these experiments; "Johnathon" is the author.

**Status:** PREREGISTERED 11-07 evening, before any E02 run. Johnathon approved the plan on 11-07
("Okay I approve"): a capacity map of H1 first, then the recipe loop with width vs depth as its first test.
Frozen by `python -m recipe freeze E02` (sha256 in `runs/recipe/E02/FROZEN.json`).
**Config:** `configs/E02.json`. **Code:** `recipe/` (facts.py, proxy.py, loop.py).

### 1. Question

At the same number of non-embedding parameters, does a wide-and-shallow, H1-shaped or narrow-and-deep
model store more facts? H1's planned next growth stage (stage A) is depth only (20 → 40 blocks at width 1,024). A synthetic-triples
study found facts store better in width, and an earlier fact-layer toy of ours favored a wider MLP over sparse
memory. If depth stores fewer facts per parameter, depth-only growth is the wrong lever for knowledge. If it
stores the same, Stage A's choice is fine for facts.

### 2. Setup

| Item | Value |
|---|---|
| Text | H1's own v2 training text (`general/train`), H1's 32,768-token tokenizer, seq 512, batch 32 |
| Budget | 40M tokens per run (about 2,440 updates) |
| Facts | 480 synthetic people × 4 attributes (birth city, birth year, school, employer) = 1,920 facts |
| Exposure tiers | 120 people each seen 1, 4, 16 or 64 times; one exposure = one short biography with the four canonical sentences shuffled, plus filler (10,200 biographies, 744K tokens, 1.9% of training tokens) |
| Optimizer | AdamW (0.9, 0.95), wd 0.1, LR 3e-3 at width 256, hidden-matrix LR scaled by 256/width (muP-style), 2% warmup, flat, then linear decay over the last 20% to 0.1× |
| Seeds | 0, 1, 2. A seed fixes the init, the text order and the biography order; arms with the same seed see identical data |
| Device | Mac MPS, bf16 autocast. Runs yield the GPU (exit, then retry) when a higher-priority job on the same Mac needs it |

| Arm | Width | Blocks | Heads / KV | FFN | Non-embedding params | Embedding (tied) |
|---|---|---|---|---|---|---|
| wide | 512 | 4 | 8 / 2 | 1,728 | 13,242,880 | 16.8M |
| mid (H1-like aspect) | 384 | 7 | 6 / 2 | 1,296 | 13,209,216 | 12.6M |
| deep | 256 | 16 | 4 / 1 | 864 | 13,246,720 | 8.4M |

The tied embedding differs by arm (it scales with width). That is a known confound: the wide arm also has
more embedding parameters. It is reported, not corrected.

FFN = 3.375 × width, H1's ratio. H1's width/depth is 51; the arms are 128, 55 and 16.

### 3. Measures

- **Primary:** `facts_seen`, the number of the 1,920 facts recalled exactly (teacher-forced greedy: every
  answer token is the argmax) after the canonical sentence prefix, e.g. "Mara Quint was born in the city of".
- **Guard:** held-out loss on `general/validation` (64 windows of 512 tokens).
- **Secondary:** recall by tier (the capacity curve), `facts_unseen` (a Q/A paraphrase never used in
  training: storage that can be read out another way), `stack/validation` loss, wall-clock training time.

### 4. Predictions (written before any run)

| # | Prediction | Would refute it |
|---|---|---|
| P1 | wide stores more facts than deep (WIDE_STORES_MORE) | deep ≥ wide, or a TIE |
| P2 | mid sits between them on facts | mid outside [deep, wide] |
| P3 | deep has the best held-out loss (depth helps composition more than storage) | deep worst on general loss |
| P4 | unseen-wording recall is near zero for all arms at this scale | any arm > 10% unseen recall at tier 64 |
| P5 | deep is the slowest per token on the Mac (more, smaller matmuls) | deep fastest |

### 5. Decision rule

For each pair (A, B), paired by seed, d_s = facts_seen(A) − facts_seen(B):
- **A_STORES_MORE** if mean d ≥ 3% of 1,920 (58 facts) and d_s > 0 for all three seeds; symmetric for B.
- **TIE** otherwise.
- A STORES_MORE verdict gets **_BUT_WORSE_LOSS** if the winner's mean general held-out loss is worse by more
  than 0.02 nats.
- Three seeds can't give a small p-value (an all-positive sign test is p = 0.25 two-sided); the rule asks for
  a consistent sign plus a practical margin, the same standard as our other small pilots. This is a pilot verdict.

### 6. What happens after

| Outcome | Next step for the recipe loop |
|---|---|
| WIDE_STORES_MORE (no worse loss) | Propose a width-growth arm for Stage B (Net2Net-style widening) and test widen vs deepen from the same proxy parent |
| DEEP_STORES_MORE or TIE | Depth-only growth is fine for facts; next round tests FFN ratio (3.375× vs 5×) at fixed shape |
| any _BUT_WORSE_LOSS | Report the trade-off; facts and fluency pull apart, so the round after uses a combined score |

Whatever wins becomes the incumbent recipe of the loop, and a proposal for H1's training plan, not a change
to it.

### 7. Report

Filled 11-08 from `runs/recipe/E02/report.json` (9/9 runs, exit 0).

| Arm | facts_seen by seed (0, 1, 2) | Mean / 1,920 | Tier 1 / 4 / 16 / 64 recall | General loss (mean; seeds) | Train time |
|---|---|---|---|---|---|
| wide 512×4 | 215, 261, 278 | 251.3 | 4.3 / 3.9 / 8.8 / 35.3% | 4.228 (4.227, 4.230, 4.228) | 1,335 s |
| mid 384×7 | 284, 171, 194 | 216.3 | 3.7 / 4.7 / 6.9 / 29.7% | 4.237 (4.231, 4.226, 4.254) | 1,448 s |
| deep 256×16 | 90, 173, 129 | 130.7 | 3.6 / 3.8 / 6.0 / 13.9% | 4.527 (4.710, 4.301, 4.569) | 1,667 s |

| Pair | d by seed | Mean d (margin 58) | Verdict |
|---|---|---|---|
| wide vs mid | −69, +90, +84 | +35 | TIE |
| wide vs deep | +125, +88, +149 | +121 | **WIDE_STORES_MORE** (wide's loss is also better, by 0.30 nats) |
| mid vs deep | +194, −2, +65 | +86 | TIE (seed 1 is −2) |

Predictions: P1 confirmed. P2 holds on means only (mid was the best arm on seed 0, behind deep on seed 1).
P3 refuted: deep had the worst loss, not the best. P4 confirmed: unseen-wording recall is 0 for every run.
P5 confirmed: deep is 25% slower than wide per run.

Reading. Tiers 1 and 4 sit at the guessing floor (about 4%: 1 of 40 cities, 1 of 20 schools, 1 of 20 employers),
so at this budget only facts seen 16+ times are stored at all.
Two confounds keep this from being "depth stores fewer facts":
- **Training quality.** Deep's loss varies by 0.41 nats across seeds (wide: 0.003). Deep seed 0, at 4.71, also has
  only 6% tier-64 recall. Under the muP-style rule deep's hidden matrices got the highest LR (3e-3, versus 1.5e-3
  for wide). A 16-block model at width 256 may need a lower LR or more steps, so deep may simply have trained worse.
- **Embedding.** Wide's tied embedding is twice deep's (16.8M vs 8.4M), so its total parameters are 30.0M vs 21.6M.

Next step (§6 says propose width growth). Because of the confounds, the next round is first a fair-tuning rematch,
E03: an LR sweep per arm (seed 0), then 3 seeds at each arm's best LR. Only if wide still wins there does a
widen-vs-deepen growth test follow.
