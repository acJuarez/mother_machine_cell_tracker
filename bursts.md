# Burst detection: v1, v2 and v3

The reference for `mmtrack/bursts.py` (all the code), `07_build_bursts.py` (the build) and
`test_bursts.py` (the regression contract). Written to be read start to finish by someone who has
not seen the code. Every number here is measured on the shipped outputs.

## The one-paragraph version

A **burst** is a cell whose fluorescence jumps well above its own recent past and then decays
slowly as the cell grows and dilutes the protein. There are three definitions and a ground truth.
**v1** is a hand-built rule: it is right when it fires and finds about one burst in eight. **v2**
is a gradient-boosted classifier: it finds about three in four and is wrong a third of the time.
**v3** is a monotone classifier built so that every statistic stays inside one kymograph. On 57
held-out trenches it scores between the other two on F1. It is the only one whose **per-gene
ordering** agrees with the hand labels (Spearman 0.714, against v2's 0.607 and v1's **−0.393**).
The **manual** layer is 837 bursts marked by eye. It is the reference all three are scored
against, not a fourth definition.

---

## 1. The definitions at a glance

| | **v1** — the rule | **v2** — the classifier | **v3** — the within-trench model | **manual** |
|---|---|---|---|---|
| what it is | 4 OR'd routes, closed form | gradient boosting, 43 features | gradient boosting, 38 features, **monotone** | hand labels |
| scope of its statistics | **(gene, experiment)** | within-trench | within-trench | — |
| signal | `fluor_per_ms` | `fiji_mean` + raw | raw `intensity_mean_fluor` (+ `fiji_mean`) | the viewer |
| events | **158** | **1,041** | **1,065** | **837** |
| holdout P / R / F1 (±1 frame) | 0.795 / 0.146 / 0.246 | 0.661 / 0.789 / **0.719** | 0.560 / 0.700 / 0.622 | — |
| labels missed dataset-wide, ±1 | 723 / 837 (86%) | **184 (22%)** | 267 (32%) | — |
| FPs on 4 held-out empty trenches | **0** | 4 | 2 | — |
| Spearman vs labels, per-gene rate | −0.393 | 0.607 | **0.714** | 1 |
| reproduced how | re-run | **loaded** model | refitted, deterministic | resolved |
| a threshold you can write down? | **yes** | no | no | — |

**Which one to quote:**
- For a **per-gene rate or ranking**, use **v3**.
- For **recall**, or for finding bursts to look at, use **v2**.
- For a **methods sentence with a threshold in it**, use **v1**.
- Always name the definition, and never put `manual` on a rate axis as though it were a rival.

---

## 2. Regenerating everything

```bash
conda run -n mmct --no-capture-output python 07_build_bursts.py     # < 1 min, asserts every count
conda run -n mmct --no-capture-output python test_bursts.py      # a few seconds
```

`poetry` is not installed on the lab machine; the `mmct` conda env is the one that works.
**Inputs** are frozen tables in the git-ignored `data/`, listed in `bursts.INPUTS`. Nothing here
needs the `/Volumes/mcovert` share.

| input | what it is | made by |
|---|---|---|
| `data/all_data_annotated_082026.csv` | one row per cell-frame, 23 experiments, with `fluor_per_ms`, `is_degenerate_mask`, `is_failed_experiment` | §11 of `trackastra_figures_082026.ipynb` |
| `data/fiji_fluor_nospeckle_090926.csv` | per cell-frame values of the viewer's FIJI clip + median layer | `cell_tables.compute_fiji_fluor` |
| `data/annotated_bursts_100226.csv` | the 860 hand-marked bursts (`frame` = **viewer** frame) | the labeller, on the viewer |
| `data/time_dict_082026.json` | per-trench `{start, end}` frame window | `build_time_dict.py` |
| `models/burst_classifier_091026.joblib` | v2's fitted model (a frozen input, see §4) | the v2 fit |

**Outputs** are in `bursts.OUTPUTS`:

| output | rows |
|---|---|
| `data/bursts_v1_100226.csv` | 158 |
| `data/bursts_v2_100226.csv` | 1,041 |
| `data/bursts_v3_100226.csv` | 1,065 |
| `data/bursts_manual_100226.csv` | 837 |
| `data/burst_labels_recheck_100226.csv` | 19 labels that resolve to no cell-frame, each with a diagnosis |
| `data/burst_splits_100226.csv` | v3's trench split: 167 dev / 57 test / 19 unreviewable |
| `data/bursts_viewer_v1v2v3manual_100226.csv` | 1,732, the union the viewer draws |
| `models/burst_v3_100226.joblib` | v3's model |

**The union is not one definition's event set.** It has one row per cell-frame flagged by *any*
layer, plus a boolean `burst_{v}` column per layer. Filter it with
`bursts.load_bursts(version='v3')`.

**The pipeline, in `build_all` order:**
1. **The clean table.** `clean_cells()` is `load_cells()` (which drops the 3 `FAILED_EXPERIMENTS`
   and 2 `FAILED_KYMOGRAPHS`) minus the degenerate masks. That leaves 78,931 cell-frames over 243
   trenches.
2. **Hand labels.** `load_hand_labels` turns 860 labels into 837: 1 is in an excluded trench, 19
   are unresolvable, and 3 are `KNOWN_MISLABELS`. `build_manual` then adds geometry and position.
3. **v1.** `run_v1`.
4. **v2.** `run_v2`: featurize, score with the shipped model, link at 0.75.
5. **v3.** `run_v3`: freeze the splits, build features, fit on dev, score, link at 0.90.
6. **The union.** `build_burst_csv`.

The build reproduces the previously shipped files exactly: identical event sets, columns and row
order, with scores equal to ~1e-16. A rebuilt v3 model predicts identically on all 78,931
cell-frames.

**The viewer.** `generate_kymograph_viewer.py` reads the union by default and draws four nested
outline layers, widest first: **manual** cyan 5.0, **v1** yellow 3.5, **v2** magenta 2.25,
**v3** green 1.0.
- **Why the widths are nested.** Outlines are true mask contours, so two layers that agree draw
  the same polygon. Only nested strokes let you tell agreement apart from "topmost layer only".
- **How to read a burst:**
  - cyan alone: a real burst every definition missed
  - magenta with no cyan: v2 fired where nothing was annotated
- **Rendering needs the share**, because it reads the image stacks.
- **Adding a layer is seven coordinated edits.** `test_bursts.py` §6 checks the invariants.

The figures notebook is `notebooks/analysis_notebooks/trackastra_figures_100326.ipynb`.

---

## 3. v1 — the rule

`bursts.BURST_DEF_V1`, `run_v1`. Four OR'd routes over `fluor_per_ms`. Every event carries a
`route` column saying which route admitted it.

| route | events | the bar | measured band |
|---|---|---|---|
| `absolute` | **97** | clears `m + 4.0·s`, is a local max, prominence ≥ `3.7·s` | `k_prom` plateau **[3.6, 3.8]** |
| `fold_left` | **36** | `peak ÷ max(b_left, 0.5·m)` ≥ 3.6 | **[3.4, 3.79]**; upper edge set by one event |
| `fold` | **18** | `peak ÷ max(b_left, b_right)` ≥ 3.3 | **(3.10, 3.60]** |
| `tukey` | **7** | `peak >` its own trench's `Q3 + 4.0·IQR` | control-clean only at **[4.4, 4.8]** |

**The four routes are strictly nested**: 97 → 115 → 151 → 158, with zero losses. Each route only
ever adds events.

**The signal.** At every frame, take the **maximum** `F` over the cells of one lineage family (a
cell and all its descendants), further split by spatial gaps.
- **Why the maximum.** A burst is inherited by both daughters. The max envelope traces the whole
  episode as one hump and is blind to tracking splits: one bursting cell split across six track
  IDs still gives one event.
- **Why per family.** It separates two bursts running at once in one trench (008/1488, frames
  50–59).

**The ruler.** For each **(gene, experiment)**, `m = median(F)` and `s = 1.4826·MAD`. These are
median and MAD rather than mean and SD, so the bursts being hunted cannot inflate the bar.
Everything is measured in units of `s`.

**Height is not prominence.** Prominence is how far the signal must descend before it can climb
higher.
- A constitutively bright trace has enormous height and near-zero prominence. Hence the
  prominence test.
- Do **not** swap `max(b_left, b_right)` for the lower or left flank *in MAD units*. Measured,
  that scores hupA's best wobble (6.855 s) above a real chpS burst (6.061 s).
- **As a unit-free ratio** the left base does not invert, which is what `fold_left` uses.

**Why the ratio routes exist.** The height bar is absolute *within a gene*, so a trench whose
whole signal sits below its gene's median can never clear it. **Only 106 of 245 trenches (43%) can
ever reach their own height bar.**
- A ratio is unit-free, so one bar serves a 283× brightness range, and a flat-high trace scores
  ≈ 1.0.
- Both ratio routes require `peak ≥ m + 1.0·s` to guard the denominator.
- **`fold_left`'s denominator floor is not optional.** Unfloored, chpS 010/46 scores 30× on a left
  base of 0.006. The floor `f = 0.5` is bracketed on both sides.
- **Why the left base matters.** `max(b_left, b_right)` usually binds on the *right*, because a
  dilution decay rarely returns to its pre-burst floor. So a max base penalises long bursts. The
  worked case is gfcE 013/153 frame 31: fold-left 3.787 passes, every other test fails.

**The tukey route** is a shape statement about one trench rather than the gene baseline.
- It reaches the two documented near-misses no height bar reaches: gfcE 013/153 f31 and alkA
  001/1192 f9.
- **κ = 4.0 is a deliberate call.** 1.5 gives 523 events and breaks every regression trench. No κ
  both adds events and keeps the constitutive controls at zero.
- **Its one hupA event** (012/1165 f81) is bright but not prominent. A marginal distribution
  cannot see a rise.
- **It is evaluated before the fold routes' height floor.** Inheriting that floor would cost
  exactly the near-misses the route exists for.
- **The fence is computed on the same column the detector runs on.** Raw and per-ms fences differ
  by 150–500×.

**Counting each episode once.**
- **Episode merge.** A later peak opens a new burst only if the envelope first fell below the
  route's *reset* level, or it climbed higher than the previous peak by `k_prom·s`.
- **The reset level depends on the route, and getting it wrong is silent:**
  - `absolute`: the height bar
  - `fold` / `fold_left`: `peak / R` (it fell by the factor that qualified it)
  - `tukey`: the fence. Falling through to the height bar would reopen a burst on every wobble.
- **`ROUTE_RANK`** (`absolute < fold < fold_left < tukey`) picks an episode's representative, so
  a ratio candidate on a rising shoulder never absorbs the bright peak behind it.
- **Concurrent merge.** Two events ≤ 5 frames apart and < 1.5 cell lengths apart in y are one
  burst, and the earlier survives. **v1 reports onset; v2 and v3 report the peak.**

**v1 is not local.** Its bar is a (gene, experiment) statistic, so excluding a trench moves the bar
for every surviving trench of that gene in that experiment.
- Excluding gfcE 001/2501 raised the gfcE bar 5.31%. That cost two events inside 2501 and **one
  collateral event elsewhere** (161 → 158).
- **After any exclusion, re-run v1; never filter its output.** Filtering would have given 159.

**What v1 does not see.** 139 of 245 trenches still cannot clear their gene's height bar, so a low
v1 rate partly measures coverage, not biology.

---

## 4. v2 — the classifier (loaded, not refitted)

`run_v2`. A `HistGradientBoostingClassifier` over **43 per-cell-frame features**, all
within-trench, at threshold **0.75**.

**The features** (`featurize`):
- within-trench percentile and normalisations of `fiji_mean` and `intensity_mean_fluor`
- raw values and shape
- temporal features **along the lineage**: lags, trailing max/median, rise, fold above the trailing
  median, asymmetry
- within-frame rank
- per-trench absolute magnitude
- cell-cycle position

**Why "along the lineage" matters.** A track ends at every division, so a plain groupby-shift
loses the signal exactly where a burst is inherited. The top feature is the fold above the
lineage's trailing 5-frame median, the per-cell-frame analogue of v1's `fold_left`.

**What drives a call.** The top driver is one of three quantities:
- percentile within its trench (48.9% of events)
- rank in its frame (41.1%)
- rise above its own past (10.0%)

So the honest gloss is: *brightest in its frame, top few percent of its trench, raised above its
own recent past*. v2 has **no closed form**; a depth-3 surrogate reaches F1 0.589 against 0.667.

**How it was fitted:**
- **Data.** 203 lacZ/gfcE labels from `data/bursts_lacZ_gfcE_091026.csv`, over 60 exhaustively
  reviewed trenches.
- **Cross-validation grouped by trench** (`StratifiedGroupKFold`). A random split trains and tests
  on the same burst.
- **Decay tail masked out of the loss** at ±8 frames: neither positive nor negative.
- **Threshold.** 0.75 is the out-of-fold max-F1 point.

**Why it is loaded, not refitted.** The model predates two changes: three of its labels were
later found to be mislabels (`KNOWN_MISLABELS`), and two trenches were excluded
(`FAILED_KYMOGRAPHS`). A refit would train on different rows and not reproduce the 1,041 events.
Because every feature is within-trench, scoring the clean table equals scoring everything and
filtering.

**Things to know:**
- **The event linker tolerates `1.5·L + 10 px·Δt`** (Δt ≤ 3), a *sum*. The superseded product
  `1.5·L·Δt` reaches 4.5 cell lengths at Δt = 3. It merged two cells 109.6 px apart on 008/1488
  and swallowed two labelled bursts.
- **Fold denominators are floored at 1% of the trench max.** Unfloored folds reach 10⁹.
- **Absolute-magnitude features are kept.** Otherwise the brightest cell of an empty trench always
  scores percentile 1.0.
- **v2 was trained on lacZ and gfcE only.** Its mazF/hupA events (57/25) are unvalidated.

---

## 5. v3 — the within-trench monotone classifier

`run_v3`. A `HistGradientBoostingClassifier` over **38 features in six physically-named blocks**,
every one computed inside a single kymograph.
- **Monotone.** `monotonic_cst` constrains **26** features to +1: brighter than its trench, risen
  above its past, fallen afterwards, brightest in its frame ⇒ more burst-like. The other 12
  (shape, position, cycle, `dilution_rate`, `n_cells_in_frame`, `trench_F_med`) have no
  defensible sign.
- **Threshold 0.90.**

| block | n | asks | attribution |
|---|---|---|---|
| **LEVEL** | 10 | how bright relative to its own trench (incl. per-trench absolute magnitude) | **48.5%** |
| **NEIGHBOUR** | 6 | how it ranks against the other cells in the same frame | 23.4% |
| **RISE** | 5 | how far above its own recent past, along the lineage | 22.7% |
| **SHAPE** | 5 | size and position | 3.4% |
| **CYCLE** | 4 | where in its cell cycle it sits | 1.2% |
| **DECAY** | 8 | does it fall afterwards, faster than growth explains | 0.8% |

**Training discipline:**
- **Splits.** `make_splits` holds out whole trenches, stratified by gene on *labels*, at
  `seed=20261002`. That gives 167 dev / 57 test / 19 unreviewable, and the test split was read
  once.
- **Training rows.** The model trains on **dev only**, with the decay tail around each label
  masked out at `mask_window = 3`.
  - W = 3 is nearly inert: F1 0.609–0.641 over W = 0…12.
  - **Judge on event F1, not AUC.** AUC rises monotonically with W while event F1 peaks.
- **Scoring.** **No trench is scored by a model that saw it.** Dev trenches carry their
  out-of-fold score from 5-fold grouped CV; test and unreviewable trenches get the dev-fitted
  model. (v2's shipped table does not do this.)
- **Linking.** `flags_to_events` reports each component's **peak** (`drift = 10 px/frame`;
  measured median 6.56, p90 18.55). Reported at onset instead, the same events give F1 0.551
  against 0.628.

**Inputs.** v3's base signal is raw `intensity_mean_fluor`. Exposure division is an exact affine
rescale within a kymograph, so it cannot move a within-trench rank. Five features read `t_fiji`,
which is exactly `fiji_mean`.

**The monotone constraint is not a concession:**

| | AUC | PR-AUC | event F1 | F1 at zero FPs on empty trenches |
|---|---|---|---|---|
| unconstrained | 0.9709 | 0.4532 | **0.641** | 0.118 |
| **monotone (shipped)** | **0.9749** | **0.4849** | 0.628 | **0.340** |

**The decay ≈ dilution finding.** Since `C = N/V`, pure dilution gives
`d(ln C)/dt = −d(ln V)/dt`.
- **The measurement.** Walk forward along *both* daughters, taking the max fluorescence and the
  summed area. Over 557 labelled bursts, `decay_rate` is +0.0566 and `dilution_rate` is +0.0490,
  so **`decay_excess` ≈ +0.0025, i.e. zero**.
- **The meaning.** A burst is one synthesis event spread out by growth, not protein being
  destroyed.
- **The consequence.** This is also why the DECAY block carries almost no attribution: it has
  nothing to discriminate with.

**No closed form.** A depth-3 surrogate reaches F1 0.405 against 0.641. **Never describe v3 as
having a threshold.** By attribution, 94.6% of the decision is: *a cell in the top few percent of
its own trench, brightest in its frame, and raised above its lineage's recent past*. That is the
same three axes v2 arrived at independently.

---

## 6. Ground truth and scope

The labeller reviewed every trench of **alkA, araB, baeS, chpS, gfcE, gldA and lacZ end to end**.
**mazF and hupA were never opened.** This is the labeller's statement; no file records it,
and it is load-bearing:
- An unlabelled cell-frame in a reviewed trench is a **true negative**.
- A reviewed trench with no label is **certified empty**. There are 16, and **14 are araB**, so
  every empty-trench argument is largely an araB argument.
- **Nothing about mazF/hupA is measurable**: 19 trenches, zero labels. Every definition's
  counts there are reported, not validated, in either direction.
- The measurable universe is **224 trenches, 74,155 cell-frames, 837 labels, a 1.129% base
  rate**.

**What a label is:**
- **The labelled frame is the burst's peak**: median offset 0, 80% within ±1.
- **`frame` in the spreadsheet is the absolute viewer frame.** `time_frame = frame − start`, with
  `start` from the time dict (`bursts.trench_window`).
- **Ingest repairs:** Excel fill-handle experiment names, and FOV de-padding (`'003'` → `3`).
- **Rejected, not snapped:** a label that resolves to no cell-frame goes to the recheck file with
  a diagnosis. Snapping would invent ground truth.

**"Constitutive genes must score ≈ 0" is wrong.** chpS is the brightest reviewed gene. It carries
100 labels at the joint-highest label rate of any gene (15.04 per 1,000 cell-frames). Its
within-trench CV (0.347) is *above* hupA's (0.246) and mazF's (0.271): bright but not flat.
- **v2 on chpS.** Measured against the labels, 82 of v2's 94 chpS events are correct (P 0.872 /
  R 0.820).
- **Not transitive.** hupA and mazF still have no labels.

**The two excluded kymographs** (`experiment_metadata.FAILED_KYMOGRAPHS`):
- `DUMM_CL008_080825` 002/1355, a 3 px-offset duplicate crop of 002/1352
- `DUMM_giTG064_Glucose_022625` 001/2501, incompletely segmented

**What exclusion costs each definition:**
- **v1** must be re-run (§3).
- **v2 and v3 are only filtered.** Adding the two trenches back changes no v3 feature value of
  any surviving cell-frame (max |diff| 0) and costs no event outside them.
- **The rate denominator** (`bursts.load_divisions()`, reading `data/divisions_082026.csv`) becomes
  5,089 divisions.

---

## 7. Results

**Holdout.** Measured on 57 test trenches with 213 labels, one-to-one optimal matching, never
merged across tolerances:

| | exact P / R / F1 | ±1 P / R / F1 | ±3 P / R / F1 |
|---|---|---|---|
| v1 | 0.641 / 0.117 / 0.198 | **0.795** / 0.146 / 0.246 | **0.949** / 0.174 / 0.294 |
| v2 | **0.567 / 0.676 / 0.617** | 0.661 / **0.789 / 0.719** | 0.717 / **0.854 / 0.779** |
| v3 | 0.417 / 0.521 / 0.463 | 0.560 / 0.700 / 0.622 | 0.669 / 0.836 / 0.743 |

**Nothing overfitted.** Dev → test at ±1 is v1 0.224 → 0.246, v2 0.731 → 0.719, and v3 0.628 →
0.622. v3's dev number is out-of-fold, so for v3 this is a like-for-like generalisation check.
**v2's lead over v3 is real.** Scored only on the 123 dev trenches it was never fitted on, v2
still reaches 0.726.

**Against all 837 labels, dataset-wide, at the exact frame:**

| | lands on a label | missed |
|---|---|---|
| v1 | 92 | 745 (89%) |
| v2 | 555 | 282 (34%) |
| v3 | 407 | 430 (51%) |

Exact-frame matching is a floor on agreement. At ±1, the misses are 723 / 184 / 267.

**The rate.** Bursts per observed division, per gene:

| gene | v1 | v2 | **v3** | manual | v3 n / divisions |
|---|---|---|---|---|---|
| gldA | 0.0727 | 0.2878 | **0.3372** | 0.2151 | 116 / 344 |
| chpS | 0.0140 | 0.2640 | 0.3090 | 0.2809 | 110 / 356 |
| gfcE | 0.0410 | 0.2823 | 0.2823 | 0.2035 | 179 / 634 |
| mazF \* | 0.0000 | 0.2385 | 0.2343 | — | 56 / 239 |
| alkA | 0.0493 | 0.2141 | 0.2334 | 0.1991 | 109 / 467 |
| hupA \* | 0.0114 | 0.2841 | 0.2159 | — | 19 / 88 |
| baeS | 0.0112 | 0.2127 | 0.2151 | 0.2247 | 270 / 1,255 |
| lacZ | 0.0567 | 0.1424 | 0.1381 | 0.1017 | 95 / 688 |
| araB | 0.0246 | 0.1198 | 0.1090 | 0.0874 | 111 / 1,018 |

\* unreviewed: there are no labels, so these rates are unvalidated in either direction.

**Pooling has to be scoped.** Over the 224 reviewed trenches and their 4,762 divisions, the
hand-label rate is **0.1758** (95% CI 0.164–0.188), against v2 0.2014 and v3 0.2079. So v2
over-counts by 1.15× and v3 by 1.18×.
- **Do not pool `manual` over all 5,089 divisions.** That mixes a reviewed-only numerator with an
  all-trench denominator.
- **The over-count is shared and uneven.** Both models over-count gfcE (1.39×), lacZ (~1.38×) and
  araB (~1.3×), and both slightly under-count baeS.
- **v3's one notable excess is gldA (1.57×).** That is also its worst holdout gene.

**The effective-cycle denominator** (`legacy/mmtrack/lineage_stats.py: effective_cycles_by_gene`, 8,889 cycles)
scales every rate by about 0.57 and reorders nothing. Report it as an honesty fix to the absolute
rate, not a new result.

**The ordering is the result:**

```
v1      gldA > lacZ > alkA > gfcE > araB > chpS > hupA > baeS > mazF
v2      gldA > hupA > gfcE > chpS > mazF > alkA > baeS > lacZ > araB
v3      gldA > chpS > gfcE > mazF > alkA > hupA > baeS > lacZ > araB
manual  chpS > baeS > gldA > gfcE > alkA > lacZ > araB         (7 reviewed genes only)
```

**Spearman against the labels, 7 reviewed genes:** v3 **0.714**, v2 0.607, v1 **−0.393**.
- **v1 is negatively correlated with the ground truth.** It ranks lacZ 2nd where the labels rank
  it 6th, and chpS 7th where they rank it 1st.
- **Not an artifact of the denominator.** On raw per-gene counts instead of rates it is v3 +0.679,
  v2 +0.607, v1 −0.559.
- **The open discrepancy is baeS.** The labels rank it 2nd and both models rank it 5th, with 282
  labels behind it.

**A count close to the label count is not the same as finding the labelled bursts.** On chpS, v3
reports 110 events against 100 labels and v2 reports 94. Yet v3 misses 44 of those labels and v2
misses 22. v3 is right about how many and wrong more often about which.

---

## 8. Cautions and data traps

- **Name the definition whenever you quote a count, a rate or a ranking.** Every ranking in this
  project's history has moved under some definition change.
- **The per-gene intervals overlap almost everywhere.** `ci_lo`/`ci_hi` are in every rate table.
  Treat adjacent swaps as unresolved.
- **The numerator counts episodes, the denominator counts divisions.** An inherited burst counts
  once while both daughters' divisions count. So a gene with long bursts relative to its cycle
  scores lower at equal biology.
- **A burst here is a concentration excursion.** `F` is a per-pixel mean, so steady synthesis in
  step with growth registers nothing.
- **`fiji_mean` is not comparable across trenches.** About 51% of cell-frames are exactly 0, and
  the clip floor is 25% of each trench's own brightest pixel. **Never put a MAD-based threshold on
  it.** The (gene, experiment) MAD collapses to zero on 11 of 27 groups, the dim genes. It is
  usable only as a within-trench rank.
- **Read every table with `dtype=bursts.READ_DTYPES`**, or through `load_cells` / `load_bursts`.
  A bare `read_csv` turns FOV `'003'` into `3`, and every join then matches nothing.
- **A viewer frame is not a `time_frame`.** The viewer labels absolute frames (`start + column`);
  every table stores the 0-based column.
- **`centroid_x` is kymograph-space** (offset by `time_frame × frame width`). Use `centroid_y` for
  any distance. Cells advect toward low `centroid_y`, the open end, which fixes `centroid_y_rel`'s
  orientation: 0 = open, 1 = closed.
- **There is no physical scale.** All lengths are camera pixels.
- **The frame interval mixes 5 and 10 min.** The per-cycle rate absorbs most of this.
- **Every definition starts from `ctc_masks`**, so none can see what the segmentation missed.
  Every measurement starts from `cells[~cells.is_degenerate_mask]`. A 1-px mask is one raw pixel
  with no averaging, and every within-trench statistic would spread its error to the whole trench.

---

## 9. Tried and rejected

One line each, so none of it is re-derived.

- **A MAD ruler on `fiji_mean`.** It collapses on 11 of 27 groups (§8). 34 settings were swept and
  none worked.
- **Three successor rules on `fiji_mean`.** All were superseded by v2. **The onset-vs-peak
  reporting trap** came out of them: the same events reported at the peak instead of the onset
  moved an agreement number from 13/156 to 89/156.
- **Tukey's fence on *lit* values.** On a sparsely lit trench the burst inflates its own fence
  (003/1598: fence 2.080, peak 2.062). It fails a regression trench at every multiplier.
- **A per-trench median/MAD baseline for v1.** It loses chpS 010/1188, and it does not reach the
  class it was proposed for. The ratio routes are the mechanism that handles trench-to-trench
  variation.
- **A per-time-frame (leave-one-out) contrast baseline.**
  - It cannot separate a burst from constitutive scatter, and against the labels it is worse than
    no transform (AP 0.185 vs 0.200).
  - A per-frame leave-one-out *mean* is identically the frame mean, so it is not a new statistic.
- **The raw, un-subtracted fluorescence.**
  - Background subtraction clips 17.9% of in-mask pixels (36% for araB), but on the dark side,
    which carries no burst information.
  - Against the labels, the shipped subtracted mean (AP 0.2025) beats every raw variant.
  - Raw is also gone for 8 chpS trenches.
- **Nine image transforms plus a threshold** (tophat, DoG, temporal-median background, a broken
  FIJI floor, ...).
  - Only the shipped FIJI chain beats no transform, and its advantage is concentrated on araB.
  - The best single within-trench threshold reaches F1 0.559 and cannot reach zero false
    positives on an empty trench.
- **Adding v2's missing FIJI features to v3** (44 features instead of 38). F1 went from 0.641 to
  0.635.
- **A closed-form surrogate for v2 or v3.** v2's reaches 0.589 against 0.667; v3's reaches 0.405
  against 0.641.
- **The hand-label template route.** A blank, never-used spreadsheet workflow, deleted in this
  consolidation. All ground truth is `annotated_bursts_100226.csv`.
- **The superseded rules (legacy v1–v11) and their fixture.** Deleted from the code. They are in
  git history before this consolidation.

**Open items, in order:**
1. Label mazF and hupA. Two of nine genes are unmeasurable.
2. Label a *bright* certified-empty trench.
3. Find out why v3 trails v2 on F1. The cause is neither the FIJI features nor in-sample
   contamination.
4. Refit v2 on all 837 labels. The shipped model saw only 203.
