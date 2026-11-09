# E05: Exposure saturation (E04's equal-table shapes, with facts seen up to 1,024 times)

> **Published copy.** Lightly edited for publication: internal machine and job-queue references removed, paths made
> repo-relative, and "the owner" replaced by the author's name. The science text (question, setup, predictions, rule,
> results) is unchanged. The private original's sha256 is in `PUBLICATION_SHA256.json`. "Claude" in the prediction
> tables is the AI assistant (Anthropic) that co-ran these experiments; "Johnathon" is the author.

**Status:** REPORTED 11-09 (§8). Preregistered 11-09, before any E05 run. Johnathon approved it on 11-09 ("okay e05a and you can go
past the cap"). Frozen by `python -m recipe freeze E05` (sha256 in `runs/recipe/E05/FROZEN.json`).
**Config:** `configs/E05.json`. **Code:** `recipe/facts.py` (`tiers`), `proxy.py` (`max_fact_share`,
`facts_seen_by_tier`), `loop.py` (`margin_total`, `tier_pairs`, `exposures_to_recall`). Tests are in
`tests/test_recipe_loop.py` and `tests/test_recipe_proxy.py`. The default path was checked parameter for parameter
and logit for logit against the E04 `proxy.py`.

### 1. Question

E02–E04 saw each fact at most 64 times. In that short-budget regime, wide stored the most, and with an equal word
table (E04) it still beat deep by +108 facts. But that measures how fast facts are picked up, not where the shapes end
up. **Given enough repetition, does deep catch up with wide, and if so, how many more repetitions does it need?**

The literature points to the end state: at about 1,000 exposures, GPT-2 models store about 2 bits per parameter
"irrespective of depth or width" (Allen-Zhu & Li, *Physics of Language Models 3.3*). Our 1,920 facts carry only about
10,000 bits, far below any arm's room. So every arm *could* hold them all. E05 asks whether deep actually gets there
within the budget.

### 2. Setup

Everything is as in E04's comparison stage (shapes, equal 256-wide tied table, text, 40M tokens, optimizer, schedule,
seeds 1–3), except:
- **Two more exposure tiers.** 120 new people are seen 256 times and 120 more 1,024 times. The first 480 people,
  tiers 1–64, are exactly E04's people. Total: 720 people, 2,880 facts.
- **Biography share up to 35%.** The extra tiers add about 11.9M tokens, so biographies are 29.7% of training tokens
  (E04: 1.9%). The old 25% cap is raised to 35%, with Johnathon's OK. General text drops from about 39M to 28M
  tokens per run.
- **No tuning stage.** Each arm runs at E04's pick, 0.5× (LR 1.5e-3). All three picked 0.5× in E04, and 0.25× was
  worse for all three.
- **No reuse.** The facts changed, so every run is new.

| Arm | Width × blocks | Table + P | Non-embedding |
|---|---|---|---|
| wide | 512 × 4 | 8,519,680 | 13,242,880 |
| mid | 384 × 7 | 8,486,912 | 13,209,216 |
| deep | 256 × 16 | 8,388,608 | 13,246,720 |

**Cost:** 9 runs of about 19–29 minutes on the Mac GPU, so about 3.5–4 hours. The runs yield the GPU to a
higher-priority job like E02–E04.

### 3. Measures

- **Primary:** facts stored at the top tier, `facts_seen_by_tier.1024` (of 480).
- **Guard:** `general` held-out loss.
- **Per-tier verdicts (`tier_pairs`):** the same rule on each tier's own 480 facts.
- **Exposures to 50% and 90% recall (`exposures_to_recall`):** from each arm's mean recall curve, linearly
  interpolated in log2(exposures). "None" means never reached by 1,024.
- **Also reported:** total facts, recall by tier, `facts_unseen`, `stack` loss and wall-clock time.

### 4. Predictions (written before any run)

| # | Who | Prediction | Would refute it |
|---|---|---|---|
| O1 | Johnathon | **deep still doesn't get there:** at 1,024 exposures, wide still stores more than deep (WIDE_STORES_MORE on the top tier) | TIE or DEEP_STORES_MORE at tier 1,024 |
| P1 | Claude | deep catches up: wide vs deep is a TIE at tier 1,024 | any STORES_MORE verdict at tier 1,024 |
| P2 | Claude | deep needs 2–4× the exposures wide needs to reach 50% recall | a ratio below 2 or above 4, or deep never reaching 50% |
| P3 | Claude | the gap is still open in the middle: wide vs deep is WIDE_STORES_MORE at tier 256 | TIE or DEEP_STORES_MORE at tier 256 |
| P4 | Claude | every arm reaches at least 90% recall at tier 1,024 | any arm's mean below 90% |
| P5 | Claude | the general-loss order stays wide < mid < deep | any other order |

O1 and P1 are direct opposites, so exactly one of them will be confirmed. Both are refuted only if deep stores *more*.

### 5. Decision rule

E04's rule, applied to the top tier on seeds 1, 2, 3:
- **A_STORES_MORE** if every seed pair clears the margin, or if the mean clears it and every seed pair shares the sign.
  The margin is 3% of the tier's 480 facts, 14.4. **TIE** otherwise.
- **`_BUT_WORSE_LOSS`** is added if the winner's general loss is worse by more than 0.02 nats.

The same rule is applied per tier for the secondary readings. This is a pilot verdict: three seeds.

### 6. Known limits

- **Recall near 100% squeezes differences.** If every arm saturates the top tier, a TIE can only say that the gap
  closed, not that it closed at the same speed. The exposures-to-recall reading covers that.
- **The text budget changes.** General text drops to 28M tokens, so the general loss isn't comparable with E04's, and
  the much larger biography share could crowd out the lower tiers. Tiers 1–64 are reported against E04 as a
  descriptive check only.
- **No retuning.** E04's 0.5× pick is reused. A 30% biography mix might move each arm's best LR, perhaps unevenly.
- **Still pilot scale:** one budget, small proxies and a synthetic fact format. It doesn't test capacity, since all
  arms have far more room than the facts need.

### 7. What happens after

| Outcome at tier 1,024 (wide vs deep) | Reading | Next |
|---|---|---|
| WIDE_STORES_MORE (O1) | deep is slower *and* doesn't catch up within this budget; width matters beyond learning speed here | report it; a longer-budget check before any general claim |
| TIE, with a deep/wide exposure ratio > 1 (P1) | width is a learning-speed effect: deep needs more repetitions but gets there | report it; the width-vs-depth line is done |
| DEEP_STORES_MORE | reversal at high exposure | report it as a surprise |

The E05 results and this write-up go into this repo, like E02–E04.

### 8. Report

Filled 11-09 from `runs/recipe/E05/report.json` (`python -m recipe report E05` reproduces it byte for byte, sha256
`e393b06c`). The run started 07:38 and finished 11:16 on 11-09, with no yields.

**Top tier and loss (seeds 1, 2, 3):**

| Arm | Tier-1,024 facts by seed (of 480) | General loss (mean; seeds) | Train time |
|---|---|---|---|
| wide 512×4 | 480, 480, 480 | 4.456 (4.464, 4.453, 4.452) | 1,222 s |
| mid 384×7 | 479, 479, 479 | 4.472 (4.472, 4.475, 4.468) | 1,401 s |
| deep 256×16 | 480, 477, 480 | 4.509 (4.509, 4.493, 4.524) | 1,698 s |

| Pair, tier 1,024 | d by seed | Mean d (margin 14.4) | Verdict |
|---|---|---|---|
| wide vs mid | +1, +1, +1 | +1 | **TIE** |
| wide vs deep | 0, +3, 0 | +1 | **TIE** |
| mid vs deep | −1, +2, −1 | 0 | **TIE** |

`facts_unseen` is 0 for every run.

**Recall by tier (mean of seeds) and per-tier verdicts, wide vs deep:**

| Tier (exposures) | 1 | 4 | 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|---|---|
| wide | 7.0% | 21.9% | 76.9% | 96.5% | 99.8% | 100% |
| mid | 5.3% | 11.9% | 56.3% | 88.0% | 96.8% | 99.8% |
| deep | 5.8% | 11.6% | 51.9% | 88.8% | 97.0% | 99.8% |
| wide − deep, mean d | +6 | +50 | +120 | +37 | +13 | +1 |
| verdict | TIE | WIDE | WIDE | WIDE | TIE (13.3 < 14.4; all seeds +) | TIE |

Wide vs mid has the same pattern (WIDE at tiers 4, 16 and 64; TIE at 1, 256 and 1,024). Mid vs deep is a TIE at every
tier except 16 (MID_STORES_MORE, +21).

**Exposures to reach 50% and 90% recall (log2 interpolation of the mean curve):**

| Arm | 50% | 90% | vs wide (50% / 90%) |
|---|---|---|---|
| wide | 8.1 | 40.3 | — |
| mid | 13.2 | 87.8 | 1.63× / 2.18× |
| deep | 15.0 | 78.1 | 1.85× / 1.94× |

**Predictions:**
- O1 (Johnathon) **refuted**: at 1,024 exposures, wide vs deep is a TIE (480 vs 479). Johnathon was right below that:
  wide stores clearly more at tiers 4, 16 and 64.
- P1 **confirmed**: wide vs deep is a TIE at tier 1,024.
- P2 **refuted, narrowly**: deep needs 1.85× wide's exposures to reach 50%, just under the predicted 2–4× (1.94× at
  90%).
- P3 **refuted, narrowly**: at tier 256, wide leads deep on every seed (+13, +14, +13) but the mean, 13.3, misses the
  14.4 margin, so it's a TIE.
- P4 **confirmed**: every arm is at 99.8% or more at tier 1,024.
- P5 **confirmed**: general loss is wide 4.456 < mid 4.472 < deep 4.509.

**Reading.**
- **Width is a learning-speed effect here, not a storage limit.** Given enough repetition, all three shapes store
  every fact. Deep needs about twice as many repetitions as wide to get there (1.85× to 50%, 1.94× to 90%). The gap
  is widest at 16 exposures (+120 of 480), shrinks by 256 and is gone by 1,024. That is §7's "TIE, ratio > 1" row.
- **Mid ≈ deep again**, as in E04: mid is faster to 50%, deep is faster to 90%, and the top tier ties.
- **Unplanned: many more biographies made the rare facts much easier to learn.** Tiers 1–64 are the same people with
  the same exposure counts as E04, but recall rose a lot. Wide's tier-16 recall went from 6.9% to 76.9%, and its
  tier-64 recall from 33.5% to 96.5%; deep's tier-64 recall went from 12.3% to 88.8%. §6 had worried the bigger
  biography share might crowd the lower tiers out; the opposite happened. A likely reading is that the heavily
  repeated people taught every arm the biography format, so a new person's facts take far fewer repetitions. This
  matches the "celebrity helps minority" effect in *Physics of Language Models 3.1*, but E05 changed the biography
  share and the general text together (§6), so this is descriptive only.
- **On loss, wide is still best** (0.052 nats better than deep, in 72% of deep's time). E05's losses aren't
  comparable with E04's, since the text budget changed.
- **Limits.**
  - Pilot scale: three seeds, one budget, small proxies and a synthetic fact format.
  - Recall near 100% squeezes the top tier, so the exposure ratios carry the speed reading.
  - No retuning at the 30% biography mix (§6).
  - It doesn't test capacity: 2,880 facts are far below every arm's room.

**Next (§7, the "TIE, ratio > 1" row).** The width-vs-depth line is done. The results and this write-up are in this
repo. Nothing changes for H1's training plan.
