# `trackastra_figures_082026.ipynb` — what it computes, and the math at every step

Companion document to `notebooks/analysis_notebooks/trackastra_figures_082026.ipynb`.
Every symbol used in an equation is defined in the glossary first, with units.

**Two things the notebook now applies once, at load, rather than in sections of their own:**
the three failed runs are dropped, and fluorescence is divided by each experiment's exposure.
There is no unfiltered frame and no raw-a.u. figure anywhere in it, so every number below is on
the 20-experiment frame and in **a.u./ms** unless it is explicitly labelled otherwise.

---

## 0. Symbol glossary

**No physical pixel size (µm/px) is recorded anywhere in this repo**, so all lengths are in raw
camera pixels and cannot be converted to microns from this dataset. That constraint is the reason
several thresholds below are expressed as *ratios* rather than absolute distances.

| Symbol | Meaning | Units | Source column |
|---|---|---|---|
| $F$ | Mean pixel value inside one cell's mask on the **background-subtracted** fluorescence channel, divided by that experiment's exposure | a.u. **per pixel per ms** (a.u./ms) — a *density*, not a total | `fluor_per_ms` |
| $F_{\text{raw}}$ | The same before the exposure division. Used only for the reference table in notebook section 5 | a.u./px | `intensity_mean_fluor` |
| $A$ | Segmented mask area of one cell | pixels² | `area` |
| $L$ | Cell long-axis length | pixels | `axis_major_length` |
| $W_c$ | Cell short-axis length | pixels | `axis_minor_length` |
| $y$ | Centroid position **along the trench axis** | pixels | `centroid_y` |
| $t$ | Frame index | dimensionless integer (0–178 in this set) | `time_frame` |
| $\tau$ | Acquisition interval | minutes per frame (5 or 10 here) | `minutes_per_frame` |
| $E_{\exp}$ | Fluorescence camera exposure | milliseconds (150 / 250 / 450 / 500) | `fluor_exposure_ms` |
| $m$ | Median of $F$ over all cell-frames of one (gene, experiment) | a.u./ms | computed |
| $\mathrm{MAD}$ | Median absolute deviation of $F$ about $m$ | a.u./ms | computed |
| $s$ | Robust scale estimate, $s = 1.4826\,\mathrm{MAD}$ | a.u./ms | computed |
| $k$ | Threshold multiplier on $s$ (shipped value 4.0) | **dimensionless** | `BURST_DEF['k']` |
| $W$ | Half-width of the local-maximum window | frames | `BURST_DEF['window']` |
| $\mathcal{E}_f(t)$ | Max-fluorescence envelope of lineage family $f$ at frame $t$ | a.u./ms | computed |
| $\mathrm{prom}$ | Topographic prominence of a peak on $\mathcal{E}_f$ | a.u./ms | computed |
| $\delta$ | Frame-to-frame displacement in cell-length units | **dimensionless** (px ÷ px) | `delta_y_cell_lengths` |
| $T_d$ | Doubling time of one dividing cell | minutes | computed |
| $n$ | Count of burst events in a group | dimensionless | computed |
| $N$ | Denominator count (pairs, cells) or exposure (cell-hours) | dimensionless, or hours | computed |

Units for `area` and the axis lengths were confirmed empirically rather than assumed:
median $A = 249$ px² against median $L \cdot W_c = 326.9$ px², a ratio of **0.762**, which matches
$\pi/4 = 0.785$ for an ellipse. So $A$ is in px² and $L, W_c$ in px.

### The one unit fact that governs everything

$F$ is a mean **per pixel**, so it behaves like a *concentration*. Two cells of different size
carrying the same protein density have the same $F$. Every threshold in this notebook is therefore
a statement about concentration, never about total protein. Section 7 below shows what that does to
the interpretation of a burst. Dividing by exposure does not touch that: it is a per-experiment
constant, so $F$ stays a density and simply changes what one unit of it means.

### Keys

| Key | Definition | Meaning |
|---|---|---|
| `KYMO_KEY` | (`experiment_name`, `FOV`, `trench_id`) | one kymograph / trench |
| `CELL_KEY` | `KYMO_KEY` + `track_id` | one unique cell |

`track_id` is the `ctc_masks` label. **`centroid_x` is kymograph-space** — offset by
$t \times \text{frame width}$ — so it must never be used for distance. Every spatial calculation
here uses `centroid_y`.

### The dynamic-range problem

Median baseline $F$ per (gene, experiment) spans **0.0141 a.u./ms (alkA) → 3.99 (hupA)**, a
**283×** range. Exposure normalisation does not fix this and in fact widens it slightly — the raw
spread was 218× — because the term it removes is not the dominant one: expression level is. See
section 3 for why removing exposure can widen a spread. No absolute cutoff can serve all ten
genes, which is why every criterion below is relative: a ratio, or a multiple of that gene's own
noise.

---

## 1. Δy jump QC — notebook section 3

Pairs link a cell to its successor. For **within_track** pairs (consecutive frames of one
`track_id`):

$$\Delta y = y_{t+1} - y_t \;[\text{px}], \qquad
\Delta F = F_{t+1} - F_t \;[\text{a.u./ms per frame}], \qquad
L_{\text{pair}} = \tfrac{1}{2}\left(L_t + L_{t+1}\right) [\text{px}]$$

For **division** pairs (parent's last frame → each daughter's first frame),
$L_{\text{pair}} = L_{\text{parent}}$.

The jump criterion:

$$\delta = \frac{\Delta y \;[\text{px}]}{L_{\text{pair}} \;[\text{px}]} \;\Rightarrow\; \text{dimensionless},
\qquad \texttt{is\_jump} = |\delta| > k_\delta$$

with $k_\delta = 1.0$ for within-track pairs and $1.5$ for division pairs, both dimensionless.

**Why a ratio.** A tracking error means the track landed on a *different* cell, which in a trench
of stacked cells takes roughly one cell length of apparent motion. Because $\delta$ is px ÷ px it is
independent of magnification, cell size and frame interval — and since no µm/px scale exists in this
dataset, a ratio is the only threshold that could be physically meaningful. The division bound is
wider because a real division displaces the centroid further than one frame of growth.

**Guard.** Rows with $L_{\text{pair}} \le 1$ px are segmentation failures, not evidence of a hop,
and would divide toward infinity; they take the dataset median length instead.

**Result.** **1,456 of 81,002 pairs (1.80%)** — 1,307 of 70,444 within_track (1.86%) and 149 of
10,558 division (1.41%). Median cell length 36.6 px within_track, 53.8 px division.

---

## 2. Doubling time — notebook section 4

$$T_d\;[\text{min}] = \left(t_{\text{last}} - t_{\text{first}}\right)[\text{frames}] \times \tau\;[\text{min/frame}]$$

where $t_{\text{first}}$ and $t_{\text{last}}$ are the first and last frames the **parent** track was
observed. The denominator of every mean is **divisions, not lineages** — each division event
contributes exactly one observation.

`division_events` collapses the pair table's one-row-per-daughter into one row per event and re-keys
`parent_track_id → track_id`, so the doubling time belongs to the cell that actually divided. Two
asserts protect the definition:

- every parent divides at exactly one frame — otherwise "first to last frame" is not one cell cycle
- `division_frame == last_frame` — the pair table and the cell table must agree on where a track ends

Two qualifying flags:

| Flag | Definition | Why it matters |
|---|---|---|
| `birth_observed` | the parent has a parent of its own | without it $T_d$ is **left-censored**: the cell was already alive when imaging began, so $T_d$ underestimates |
| `has_jump` | the dividing cell has ≥ 1 within-track pair with $\lvert\delta\rvert > 1.0$ anywhere in its lifetime | such a lifetime is not one cell's, so its $T_d$ is meaningless regardless of how clean the division looks |

**Result (failed runs excluded).** 5,279 divisions, pooled mean 65.87 min; the no-jump subset
(92.6% of events) gives 65.01 min.

The tables report both a **grand mean** (weights by division count) and a **mean of per-experiment
means** (weights experiments equally). They diverge when experiments contribute unequally, so both
are shown rather than silently picking one.

### $T_d$ is one frame interval short of a generation time

Measured exactly, with **0 exceptions** over all 5,279 divisions:

$$n_{\text{frames}} = t_{\text{last}} - t_{\text{first}} + 1,
\qquad
T_d = \left(n_{\text{frames}} - 1\right)\tau$$

So $n_{\text{frames}}$ counts **samples** and $T_d$ counts **intervals**. A parent on frames
$[f, l]$ divides *at* $l$ and its daughter's first frame is $l+1$, so birth-to-birth is
$n_{\text{frames}}$ intervals — one more than $T_d$ reports. This is harmless for a mean doubling
time (it is a constant $-\tau$ offset, and $\tau$ is stated) but **not** harmless anywhere $T_d$ is
inverted. See below.

### Effective cell cycles — notebook section 14

The per-cell-cycle burst rate divides by **observed divisions**, which only counts cycles the
tracker closed. `lineage_stats.cell_cycles_per_trench` converts a trench's whole observation record
into cycle units instead:

$$\text{eff cycles} = \frac{n_{\text{cell-frames}}}{\operatorname{mean}\left(n_{\text{frames}}\right)}
\qquad\text{over full cycles of that trench}$$

with a **full cycle** being `birth_observed & ~has_jump & n_frames >= 2` (5,279 → 4,225). Stricter
than the tables above, which report those flags rather than applying them — justified because the
cycle length here is inverted and then multiplied by the entire observation record, so the 186
single-frame parents would give an infinite rate rather than merely a soft one.

**`mean(n_frames)`, not `mean(T_d)/\tau`.** Worked on a synthetic trench — one cell slot, 100
frames, division every 10, so ten complete cycles: $100/10 = 10$ is right and $100/9 = 11.1$ is
not. Using $T_d$ would inflate every effective count by 12%. Asserted in the notebook.

**Cell-frames, not trench time-frames.** Trenches hold 1.4–9.8 cells per frame (CV 0.38), so the
two re-weight rather than rescale. Trench-frames give **0.36×** the division count — they
undercount the observation ~2.8× by counting the trench once instead of each cell in it.
$\tau$ cancels between numerator and divisor, so the 5/10 min mix needs no normalising (measured:
1.663 at 5 min, 1.726 at 10 min).

**Result.** 255 trenches, median cycle length **75 min**, **9,230** effective cycles against 5,279
observed divisions — **1.68–1.86× per gene, 1.75× pooled**, a 1.11× spread. Pass it to
`bursts_per_cell_cycle(denom=...)`. Two limits: `eff_cycles` is an *estimate*, so the Poisson
bounds (which treat the denominator as a known count) are too narrow; and it **rescales without
reordering** — under v6 the gene ranking is unchanged, under v12 only adjacent near-ties swap.
Do not read a changed absolute rate as a new result.

---

## 3. Exposure normalisation — applied at load

$$F \;\left[\frac{\text{a.u./px}}{\text{ms}}\right]
\;=\; \frac{F_{\text{raw}}\;[\text{a.u./px}]}{E_{\exp}\;[\text{ms}]}$$

Every gene except murQ spans two or three exposure settings, so $F_{\text{raw}}$ is not comparable
across genes. `fluor_exposure_ms` is a single value per experiment (asserted in the load cell), so
this is a per-experiment constant rescale.

### It cannot change the burst results, and does not

Baselines are computed per $(\text{gene}, \text{experiment})$, and $E_{\exp}$ is constant within an
experiment. So $m$, $\mathrm{MAD}$, $s$, the height bar $m + ks$ and the required prominence $ks$
all scale by the same $1/E_{\exp}$, and every criterion in section 4 is a comparison between
quantities that scaled identically. Verified rather than argued: re-running the whole pipeline
after the switch gives the same **94** events, the same per-gene counts, the same 7-row merge
audit and a bit-identical $k$-sweep table. (The only thing that could have broken it is
`compute_baselines` clipping `mad_scale` at $10^{-6}$; the smallest scaled value here is
$1.4 \times 10^{-3}$.)

### The linearity assumption only half-holds — measured

Dividing by exposure assumes linear detector response over 150–500 ms. Comparing each gene's
**replicate spread** (max/min of its per-experiment means) before and after:

| | genes, raw fold → per-ms fold | reading |
|---|---|---|
| **shrinks** | araB 2.96 → 1.07, lacZ 3.27 → 1.18, baeS 17.6 → 5.28 | exposure really was the confounder; these are the genes with several replicates genuinely spread across settings |
| **unchanged** | mazF 1.97 → 1.97; murQ has one experiment | only one exposure among the replicates that differ |
| **grows** | alkA 2.23 → 7.42, gldA 1.10 → 3.66, hupA 1.03 → 3.08, gfcE 1.34 → 2.87, chpS 1.20 → 2.77 | their raw means already agreed across different exposures, so dividing pulls them apart |

For the third row exposure is almost perfectly **confounded with experiment**: all but one
replicate sits at 150 ms and the odd one out at 450 or 500 ms, so "the 500 ms run" and "that
particular acquisition" are the same thing. alkA is the sharpest case — its 500 ms run has a
*lower* raw mean (7.5) than three of its four 150 ms runs (7.7–16.7), which a linear detector
cannot produce at equal expression.

**It is not clipping.** The brightest single pixel anywhere in the filtered dataset is **2,432
counts**, nowhere near any plausible ADC ceiling, and hupA — the brightest gene, and the one whose
raw means match across a 3× exposure change — tops out there rather than piling up flat.

So: quote the normalised figure for the **cross-gene ranking** (the raw/normalised rank table
shows only two adjacent swaps), and do not read a 2–3× difference between two experiments of the
*same* gene as biology.

---

## 4. Burst detection — notebook section 7

### 4a. The signal: a max envelope

For lineage family $f$ (a connected component of the parent forest within one trench):

$$\mathcal{E}_f(t)\;[\text{a.u./ms}] \;=\; \max_{c \,\in\, f \,\text{present at frame}\, t} F_c(t)$$

**Why the max, not a mean or a sum.** After a division both daughters inherit the burst. The max
traces the episode as one smooth hump; a mean would dilute it by however many cells happen to be
present, and a sum would rise with cell count alone.

It also makes detection **blind to tracking fragmentation** — `track_id` does not appear in that
expression. In `CL008_giTG068_072925 | FOV 007 | trench 821` the tracker split one bursting cell
across six `track_id`s (18 → 19 → 10 → 28 → 29 → 24 over ten frames, its centroid marching steadily
down the trench at ~25 px/frame). The per-cell view shows four apparent bursts; the envelope shows
one hump, and the detector reports one event.

### 4b. The threshold: robust median and MAD

Computed per **(gene, experiment)**, over all cell-frames in that group:

$$m = \operatorname{median}\{F\}, \qquad
\mathrm{MAD} = \operatorname{median}\left|F - m\right|, \qquad
s = 1.4826 \cdot \mathrm{MAD}$$

all three in a.u./ms.

**Why median/MAD rather than mean/SD.** A standard deviation is squared-error based, so a handful of
outlier bursts inflate it, raising the threshold and hiding the very events being hunted. MAD is
rank-based — a single outlier moves it by at most one rank position.

**Why the 1.4826.** For $X \sim N(\mu, \sigma^2)$,

$$\mathrm{MAD} \;\longrightarrow\; \sigma \, \Phi^{-1}(0.75) \;\approx\; 0.6745\,\sigma
\qquad\Longrightarrow\qquad
\sigma \;\approx\; \frac{\mathrm{MAD}}{0.6745} \;=\; 1.4826\,\mathrm{MAD}$$

The constant rescales MAD onto the σ scale, so the dimensionless $k$ reads as "how many sigmas" in
familiar terms on clean Gaussian noise, without inheriting SD's outlier sensitivity.

**Why per (gene, experiment) and not per experiment.** 10 of the 23 experiments hold two genes in
different FOVs, so experiment-only scaling hands a dim gene and a bright gene one shared noise
ruler. Fixing this alone moved chpS from 1.72 → 0.92 bursts per 1000 pairs.

**Worked value** — araB in `CL008_giTG068_072925` ($E_{\exp} = 150$ ms):

$$m = 0.015708,\quad s = 0.003765 \;[\text{a.u./ms}]$$
$$\text{height} = 0.015708 + 4(0.003765) = 0.030769, \qquad
\text{required prominence} = 4(0.003765) = 0.015062$$

### 4c. The three criteria

A frame $t^\ast$ on $\mathcal{E}_f$ is a burst when all three hold:

**(1) Height**

$$\mathcal{E}_f(t^\ast) > m + k\,s$$

**(2) Local maximum**

$$\mathcal{E}_f(t^\ast) \;\ge\; \max_{t \,\in\, [\,t^\ast - W,\; t^\ast + W\,]} \mathcal{E}_f(t)$$

**(3) Prominence**

$$\mathrm{prom}(t^\ast) \;=\; \mathcal{E}_f(t^\ast) - \max\left(b_{\text{left}},\; b_{\text{right}}\right) \;\ge\; k\,s$$

where $b_{\text{left}}$ is the minimum of $\mathcal{E}_f$ found by walking left from $t^\ast$ until
the signal exceeds $\mathcal{E}_f(t^\ast)$ or the trace ends, and $b_{\text{right}}$ likewise
rightward. Both in a.u./ms.

**Prominence is the load-bearing criterion, and the reason the constitutive control works.** Height
alone cannot separate constitutive from bursty: hupA sits at 1.3–4.0 a.u./ms permanently — two
orders of magnitude above araB — and clears any absolute bar. What defines a burst is that the signal *rose*. A flat-high trace's peaks are barely
above their neighbours, so $\mathrm{prom} \approx 0$.

Measured (on the unfiltered frame, raw a.u.): disabling prominence takes the count from 80 → 167
bursts and moves hupA from 0.00 to 1.32 bursts per 100 cells — from last place to 5th of 10.

**Worked example** — trench `007/821` (araB, 150 ms). Three local maxima clear the height bar:

| frame | $\mathcal{E}_f$ (a.u./ms) | prominence | verdict |
|---|---|---|---|
| 35 | 0.0342 | **0.0077** | rejected — a shoulder on the way up; the trace only dips to 0.0266 before climbing to the much higher 0.0495 |
| 39 | 0.0495 | **0.0354** | **passes** — the prominence is the whole rise |
| 43 | 0.0399 | **0.0005** | rejected — a wobble on the decay (0.0393 → 0.0399 → 0.0169) |

Required prominence 0.0151, height 0.0308. `find_peaks` therefore returns a single peak and the
episode merge never has to intervene.

`window` is inert while prominence is on: 80 bursts at every $W$ from 1 to 8, only biting at $W = 12$
(77 bursts). With prominence disabled $W$ does matter (250 bursts at $W=1$, 167 at 5, 151 at 8),
which is presumably the regime it was originally tuned in. *(That sweep was run on the unfiltered
frame in raw a.u.; the conclusion is about $W$, which neither change touches.)*

### 4d. Episode merge

A burst decaying over 30 frames must count once. Walking candidate peaks forward in time, a later
peak $p$ opens a **new** burst only if

$$\min_{t \,\in\, [\,p_{\text{prev}},\, p\,]} \mathcal{E}_f(t) \;<\; m + k s
\qquad \textbf{(reset to baseline, then re-rose)}$$

$$\textbf{or}\qquad \mathcal{E}_f(p) \;>\; \mathcal{E}_f(p_{\text{prev}}) + k s
\qquad \textbf{(climbed to a genuinely higher maximum)}$$

Otherwise it is the same decaying or plateauing episode and is skipped.

### 4e. Concurrent-burst detection — the v1 → v2 change

**v1** ran one envelope per **trench**. Because that is a scalar max per frame, two bursts running at
once collapse into a single hump and the dimmer one is masked.

The case: `DUMM_giTG059_060_061125 | FOV 008 | trench 1488`, frames 50–59, holds two genuine bursts
and v1 reported one.

($E_{\exp} = 250$ ms here, so the a.u./ms values are the raw ones ÷ 250.)

- **Burst A** — track 41 rises at frames 49–50 and divides at 51 into tracks 49 + 50, both daughters
  carrying the protein. Peak **0.0417 a.u./ms** at frame 52. *v1 counts this once, correctly —
  inheritance is not a new burst.*
- **Burst B** — tracks 44 and 45, two adjacent cells nearer the open end, peaking **0.0384 / 0.0397**
  at frames 54–55. At frame 55 the cells sort by $y$ as 45@88, 44@121, 43@172, 50@217, 49@263 px,
  with a **dark** cell (43, $F = 0.0136$) separating the two bright regions.

Burst B's peak has a measured prominence of **0.0031** against the required 0.0175, because burst A
is still at 0.0366 at frame 53 and leaves no saddle between them.

**v2** detects per **lineage family**, then merges concurrent adjacent events back together. Two
events in one trench are the same burst when

$$\left|t_1 - t_2\right| \le 5\;\text{frames}
\qquad\textbf{and}\qquad
\frac{\left|y_1 - y_2\right|\;[\text{px}]}{\tfrac{1}{2}(L_1 + L_2)\;[\text{px}]} < 1.5 \quad(\text{dimensionless})$$

keeping the **earlier** peak, so timing reports burst *onset* rather than being biased late by
whichever neighbour happened to peak brighter. Cell-length units again, for the same
magnification-independence reason as `flag_jumps`.

| variant | bursts | `007/821` (want 1) | `1488` frames 50–59 (want 2) | constitutive control |
|---|---|---|---|---|
| v1, per-trench | 80 | 1 ✓ | **1 ✗** | holds |
| per-family only | 100 | 1 ✓ | 3 ✗ (splits 44/45) | holds |
| **v2 = per-family + merge** | **95** | **1 ✓** | **2 ✓ (frames 52, 54)** | **holds** |

*That comparison was measured on the unfiltered frame. On the frame the notebook now runs
(20 experiments), v2 gives **94**; with section 6's area floor added as well, v1/v2/v3 give
73/89/89. The verdicts in the two right-hand columns are unchanged in every case.*

Per-family alone over-splits because tracks 44 and 45 sit in *different* lineage families (roots 30
and 31). The merge threshold is insensitive: 1.5, 2.0 and 2.5 all give 95, because the 7 concurrent
pairs in this dataset sit at ≤ 1.06 or ≥ 2.77 cell lengths, with the 1488 pair at 1.41.

---

## 5. Rates and confidence intervals — notebook section 8

$$r_{\text{pair}} = \frac{1000\,n}{N_{\text{pairs}}}
\left[\frac{\text{bursts}}{1000\;\text{pairs}}\right], \qquad
r_{\text{cell}} = \frac{100\,n}{N_{\text{cells}}}
\left[\frac{\text{bursts}}{100\;\text{cells}}\right], \qquad
r_{\text{hour}} = \frac{1000\,n}{N_{\text{cell-hours}}}$$

with

$$N_{\text{cell-hours}} = \sum_{\text{cell-frames}} \frac{\tau}{60}\;[\text{hours}]$$

$N_{\text{pairs}}$ counts **within_track pairs with `is_jump` False** — one consecutive-frame
observation per cell per frame transition, tracking errors removed. It deliberately excludes the
10,409 division pairs, which link a parent's last frame to a daughter's first and are a different
kind of observation; including them inflates the denominator ~13% and lowers every rate
proportionally.

### Exact Poisson (Garwood) 95% interval

On the count $n$ (dimensionless), then divided by $N$ and scaled:

$$n_{\text{lo}} = \frac{\chi^{2}_{0.025,\; 2n}}{2} \quad (= 0 \text{ when } n = 0),
\qquad\qquad
n_{\text{hi}} = \frac{\chi^{2}_{0.975,\; 2(n+1)}}{2}$$

**Why exact rather than $r \pm 1.96\sqrt{n}/N$.** The normal approximation gives $0 \pm 0$ at
$n = 0$ — a claim of certainty from no data — and is badly asymmetric below about $n = 10$. Half the
genes here have $n \le 4$. The Garwood interval inverts the Poisson CDF and is exact by
construction; the $n+1$ in the upper limit is what keeps it finite at zero.

That is why mazF's zero bar still carries an upper bound of **1.39** bursts per 1000 pairs — which is
**above chpS's point estimate of 0.345**. Zero observed bursts is not evidence of a zero rate.

**It is still computed, but as of 091126 it is no longer drawn by default.** `ci_lo`/`ci_hi` (and
`{v}_lo`/`{v}_hi` on `burst_rate_comparison`) are columns of every rate table, and that is where a
quoted interval must come from. The bar plots — `plot_bursts_per_cell_cycle`,
`plot_burst_rate_comparison`, `plot_burst_rank_bars` — now label each bar with its own rate and
`n/N` instead, on the argument that the rate is the quantity the figure is about and it previously
appeared nowhere on the figure. Pass `show_ci=True` to any of the three to get the whiskers back;
that path is unchanged (verified bit-identical on all ten figures). The consequence to keep in mind
is that **a taller bar on a whiskerless figure is not a better estimate** — between adjacent
definitions these intervals do not separate.

### The TOTAL row

Recomputed over the whole frame, never summed. For the count columns the two agree here (each cell
carries exactly one gene, and no row has a null gene), but the **rates** must not be averaged across
genes: TOTAL is total ÷ total, and genes differ ~10× in size. `n_experiments` genuinely cannot be
summed either, since 10 experiments hold two genes.

---

## 6. The failed-run exclusion — applied at load

Three runs are excluded as failed. As of 2026-09-01 this happens in the notebook's load cell, so
there is no unfiltered frame downstream and no duplicated "with / without" pair of figures:

| experiment | holds | note |
|---|---|---|
| `DUMM_alkA_chpS_111325` | alkA (9 trenches), chpS (15) | |
| `DUMM_baeS_gfcE_112025` | gfcE (7 trenches) | **no baeS data despite the name** |
| `DUMM_giTG068_059__lactose_062025` | lacZ (4), hupA (1) | |

Together 9.6% of unique cells (12,696 → 11,476) and 37 of 292 trenches, touching 5 genes:
chpS −37%, gfcE −20%, alkA −18%, hupA −18%, lacZ −7%. araB, baeS, gldA, mazF and murQ are untouched.

**Filtering is exactly equivalent to re-detecting**, because `compute_baselines` groups by
(gene, experiment) — dropping whole experiments cannot change a kept group's median or MAD. That
is what makes it safe to do at load: detecting on the filtered frame and filtering a full-frame
burst table give the same event set.

**Effect on bursts:** 95 → 94. The one lost event is a marginal chpS burst
($F = 0.0171$ against a threshold of 0.0162 a.u./ms — that run's chpS baseline is near zero, itself
a symptom of the failure).

**Effect on doubling time** — measured with the notebook's own `division_events` / `doubling_by_gene`:

| gene | n divisions | mean all (min) | mean kept (min) | Δ (min) | Δ / SEM |
|---|---|---|---|---|---|
| chpS | 543 → 356 | 65.63 | 59.56 | **−6.06** | −3.57 |
| hupA | 104 → 88 | 60.10 | 57.61 | −2.48 | −0.70 |
| gfcE | 873 → 701 | 59.14 | 56.97 | −2.17 | −1.65 |
| lacZ | 729 → 688 | 73.68 | 74.62 | +0.95 | +0.50 |
| alkA | 560 → 467 | 57.22 | 58.70 | **+1.48** | +0.81 |
| araB, baeS, gldA, mazF, murQ | unchanged | — | — | 0.00 | 0.00 |
| **TOTAL** | 5,788 → 5,279 | 66.01 | 65.87 | −0.14 | — |

**Only chpS is a real move** (−3.6 SEM, and it is the gene losing 37% of its divisions). This is
**not** a uniform inflation — alkA and lacZ go *up*, and the pooled total barely budges.

---

## 7. What the units change about reading the results

Because $F$ is a per-pixel mean, it **falls when a cell grows at constant total protein**. Checking
the araB reference burst directly, with total $= F \times A$ in a.u.·px/ms:

| frame | $F$ (a.u./ms) | $A$ (px²) | $F \times A$ |
|---|---|---|---|
| 36 (pre-burst) | 0.0192 | 393 | 7.53 |
| 37 | 0.0539 | 320 | 17.23 |
| 38 | 0.0758 | 385 | 29.19 |
| **39 (peak)** | **0.0763** | **384** | **29.31** |
| 43 | 0.0627 | 462 | 28.97 |
| 47 | 0.0498 | 449 | 22.37 |
| 51 | 0.0492 | 586 | 28.81 |
| **55 (end)** | **0.0410** | **743** | **30.49** |

From peak (frame 39) to frame 55 — the percentages are scale-invariant, so they are the same
numbers the raw-a.u. version of this table gave:

| quantity | change |
|---|---|
| $F$ | 0.0763 → 0.0410 a.u./ms — **−46%** |
| $A$ | 384 → 743 px² — **+93%** |
| $F \times A$ | 29.31 → 30.49 — **+4%** |

**The long "decay" after every burst in this notebook is dilution by growth, not protein loss.** The
cell made a pulse of protein and then simply grew into it. Total fluorescence is flat to within 4%
across 16 frames while the concentration halves. The detector is right to count that as one event,
and the episode-merge rule is doing the correct thing for the correct reason.

It also means these burst rates measure **concentration excursions, not synthesis events**. A gene
that raised total protein steadily in step with growth would register no burst at all.

---

## 8. Results, and their limits

### Burst rate per gene (v2, $k = 4.0$, failed runs excluded)

| gene | $n$ bursts | good pairs | cells | per 1000 pairs | 95% CI | per 100 cells | 95% CI |
|---|---|---|---|---|---|---|---|
| lacZ | 35 | 10,305 | 1,483 | 3.396 | 2.37–4.72 | 2.360 | 1.64–3.28 |
| araB | 22 | 11,598 | 2,240 | 1.897 | 1.19–2.87 | 0.982 | 0.62–1.49 |
| alkA | 9 | 5,839 | 1,016 | 1.541 | 0.70–2.93 | 0.886 | 0.41–1.68 |
| gfcE | 14 | 9,770 | 1,537 | 1.433 | 0.78–2.40 | 0.911 | 0.50–1.53 |
| gldA | 7 | 5,012 | 781 | 1.397 | 0.56–2.88 | 0.896 | 0.36–1.85 |
| murQ | 1 | 1,490 | 253 | 0.671 | 0.02–3.74 | 0.395 | 0.01–2.20 |
| chpS \* | 2 | 5,794 | 785 | 0.345 | 0.04–1.25 | 0.255 | 0.03–0.92 |
| baeS | 4 | 15,336 | 2,673 | 0.261 | 0.07–0.67 | 0.150 | 0.04–0.38 |
| hupA \* | 0 | 1,346 | 187 | 0.000 | 0.00–2.74 | 0.000 | 0.00–1.97 |
| mazF \* | 0 | 2,647 | 521 | 0.000 | 0.00–1.39 | 0.000 | 0.00–0.71 |

\* constitutive — expected to score ~0, and the acceptance test for the whole method.

### What holds

- **The constitutive control passes at every $k$ from 2 to 6**: mazF 10th of 10, hupA 9th, chpS 7th.
  This is the method's main validation and it is non-trivial, because those three genes are also the
  *brightest* in the set — any rule keyed on absolute fluorescence would rank them first.
- lacZ is top and baeS / hupA / mazF are bottom under **both** denominators.
- Conclusions survive the failed-run exclusion (95 → 94 bursts, one marginal event).

### What does not hold — read these as unresolved

- **lacZ does not separate from araB.** CIs 2.37–4.72 vs 1.19–2.87 overlap. Only lacZ against
  baeS / chpS / hupA / mazF is a separation the counts support.
- **The middle of the ranking is noise.** alkA, gfcE and gldA occupy ranks 3–5 in *different orders*
  depending on which denominator is used, with rates within 0.14 per 1000 pairs of each other.
- **Doubling time barely moves under the exclusion** — only chpS shifts meaningfully.

### Uncorrected confounders

- **Frame interval.** $\tau$ mixes 5 and 10 min/frame, and a burst must clear threshold within one
  interval. $r_{\text{hour}}$ absorbs most of this, which is why it is in the summary table;
  $r_{\text{pair}}$ and $r_{\text{cell}}$ do not.
- **Exposure.** Divided out at load, but the linear-response assumption behind that division only
  half-holds — see section 3, where it shrinks the replicate spread for 4 genes and widens it for
  5. Burst detection sidesteps the whole question by being relative to each (gene, experiment)'s
  own noise, so none of the burst results depend on it.
- **v2 at low $k$.** On the *unfiltered* frame chpS climbed to 3rd of 10 at $k = 2$. With the
  failed runs excluded at load that goes away: chpS is 6th at $k = 2$, 5th at $k = 3$ and 7th at
  the shipped $k = 4$, and mazF is 10th throughout. The failed `DUMM_alkA_chpS_111325` run, whose
  chpS baseline is near zero, was producing the low-$k$ excursion.
- **No physical scale.** All lengths are pixels; nothing here can be quoted in µm or µm/min.
- **Exposure is confounded with experiment.** For 5 of the 10 genes every replicate but one sits at
  150 ms, so "the long-exposure run" and "that one acquisition" cannot be told apart by any
  arithmetic on this dataset.
