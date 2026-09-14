# How burst detection works

A companion to `mmtrack/bursts.py` and section 8 of
`notebooks/analysis_notebooks/trackastra_figures_090326.ipynb`. Written to be read start to
finish by someone who has not seen the code. Every number here is measured on the shipped
dataset: 20 experiments, 10 genes, 255 trenches, 11,476 unique cells, 81,799 cell-frames.

---

## The one-paragraph version

We want to find moments where a cell suddenly lights up. For each trench we build one signal
over time — the brightness of the brightest cell in each family at each frame — and look for
peaks on it. A peak counts as a burst if it is **bright** (well above what that gene normally
sits at), if it is a **local maximum**, and if it genuinely **rose** (it stands clear of the
dips on either side). The last test is the important one: it is what stops a permanently bright
gene from scoring, because a flat line has no rises in it. Peaks belonging to the same episode
are then collapsed so one long burst counts once. Everything after that is refinement of those
four ideas.

---

## 1. What is being measured

The signal is one number per cell per frame:

$$F = \frac{\text{mean pixel value inside the cell's mask on the background-subtracted fluorescence channel}}{\text{that experiment's exposure time in ms}}$$

Column name: `fluor_per_ms`, units **a.u. per pixel per ms**.

Two consequences worth holding onto:

- **It is a concentration, not a total.** Two cells of different size with the same protein
  density have the same $F$. So when a cell grows without making more protein, $F$ *falls*.
  This is why the long tail after every burst is dilution by growth rather than protein loss.
- **Dividing by exposure makes experiments comparable** (exposures run 150–500 ms). It changes
  no verdict, because every threshold below is built from the same data and scales identically.

### The problem in one number

The median $F$ per (gene, experiment) runs from **0.0141** (alkA) to **3.99** (hupA) — a
**283×** range. Per gene, medians span 0.016 (araB) to 1.52 (hupA).

So **no absolute cutoff can work.** A threshold that catches an araB burst fires on every hupA
wobble. Every rule below is therefore *relative*: a multiple of that gene's own noise, or a
ratio.

And the hard part: **hupA, mazF and chpS are constitutive** — always on, always bright. They
are the acceptance test. A correct definition must score them near zero *even though they are
the brightest genes in the set*. Any rule keyed on absolute brightness ranks them first.

### A second thing to measure: the FIJI clip + median values

Everything above measures $F$ on the raw background-subtracted image. From v8 onward there is a
**second** measurement, taken on a deliberately transformed image, because that is what the eye
is actually reading when a burst is judged by hand in FIJI. Column names `fiji_mean`,
`fiji_max`, `fiji_frac`; produced by `compute_fiji_fluor` in `mmtrack/track_displacement.py`
into `data/fiji_fluor_nospeckle_090926.csv`.

The transform is two steps, in this order:

1. **Clip.** Set the black point to $v_\text{min} = \lfloor 0.25\,v_\text{max} + 0.5 \rfloor$,
   where $v_\text{max}$ is that trench's own brightest pixel across all frames. This is FIJI's
   "set the minimum to 25% of the max" — a fraction of the **range**, not a percentile of the
   pixel *count*. The two readings are nothing alike here: on background-subtracted data the
   25th percentile is **0** while $0.25\,v_\text{max}$ is 8.5 on the worked trench, and they
   differ on **78%** of rendered pixels, taking the black fraction from 21.9% to **98.9%**. The
   half-up rounding matters and is not decoration: Python's `round(8.5)` is 8, where a person
   types 9.
2. **Median, ImageJ radius 2.0**, on the **assembled kymograph**.

**Why the assembled kymograph and not frame by frame.** FIJI is run on
`subtracted/fluor_kymos/<fov>_<peak>.tif`, which `create_kymograph` writes as a plain
concatenation along $x$ — bit-identical in layout to the viewer's own assembly. So the kernel
**crosses frame seams**, mixing adjacent time points. That is not a rounding difference: frames
are 20 px wide and the kernel spans $\pm 2$, so filtering per frame instead changes **8.3%** of
all pixels, confined to columns 0, 1, 18 and 19 of every frame, by up to 67 grey levels. Every
interior column is bit-identical. **Read the seam columns as a visual aid only** — they are not
a per-timepoint measurement.

**Why "radius 2.0" is 21 points.** ImageJ's `RankFilters` uses $r^2 = \lfloor \text{radius}^2
\rfloor + 1$, so radius 2.0 admits $x^2 + y^2 \le 5$: a 5×5 square minus its corners, **21**
points. It is neither a 5×5 square (25) nor `skimage.morphology.disk(2)`, which is 13 points and
is **bit-identical to ImageJ radius 1.75** — a visibly weaker smooth, differing on 34% of pixels
on real trench data. Edges replicate, as ImageJ does; `scipy`'s default reflect mode disagrees on
~8%.

**No small-object removal.** A third step, dropping visible connected components below 10 px,
shipped briefly and was withdrawn: it had stopped being a denoiser and become a *detection*
parameter — trenches read as fully dark only because sub-10px components had been deleted, not
because nothing cleared the floor. Removing it raises the lit fraction from 40.3% to **48.9%** of
cell-frames, which is why v9's gate had to be re-swept. It remains available but off.

The whole chain is verified **bit-identical** to a manual FIJI session on
`DUMM_giTG059_060_061125` FOV 008 trench 1488 at display range [9, 34], decoded back out of the
viewer's own emitted PNG.

**One measurement, three columns, one candidate set.** `fiji_mean` averages the transformed
values over the cell, `fiji_max` takes its maximum, `fiji_frac` the fraction of its pixels above
the floor. "Any signal" is therefore the *same* set on all three — **32,979** cell-frames — since
all three read the same above-floor pixels. The choice between them only matters once a
*threshold* is applied.

**The consequence that governs every definition built on this column.** The floor is per-trench,
so values are **not comparable across trenches**: a trench's floor is set by its own brightest
pixel, which in a bursting trench is the burst. And ~51% of cell-frames are exactly **0**, which
breaks the ruler of §3 outright — the (gene, experiment) median+MAD **collapses on 11 of 28
groups**, giving $m = 0$ and $s$ floored at $10^{-6}$, so $m + ks \approx 0$ and every local
maximum qualifies. Measured: trenches in a collapsed group fire on **74.1%** of their local
maxima against 42.9% for healthy ones, and the collapsed groups are exactly the dim interesting
genes — all of araB, all of lacZ, all of gfcE. §7's v8 and v10 entries are both about what to use
instead.

---

## 2. Step one: build one signal per trench

A trench holds a stack of cells. Rather than track each cell separately, we take at every frame
the **maximum** $F$ over the cells present:

$$\mathcal{E}(t) = \max_{c \text{ present at frame } t} F_c(t)$$

This is the **max envelope**.

**Why the maximum?** When a bursting cell divides, both daughters inherit the protein. Following
cells individually would count that as several bursts. Following the maximum traces the whole
episode as one smooth hump. It also makes detection blind to tracking mistakes: in one trench
the tracker split a single bursting cell across six different IDs; the per-cell view shows four
bursts, the envelope shows one hump, and the detector reports one event.

A mean would be diluted by however many cells happen to be present; a sum would rise with cell
count alone. The maximum is the only summary that is stable to both.

**Which cells share an envelope** is the one thing that changes between v1, v2 and v3 — see §7.

---

## 3. Step two: build a ruler out of each gene's own noise

For each **(gene, experiment)** group, over all its cell-frames:

$$m = \operatorname{median}(F), \qquad
\mathrm{MAD} = \operatorname{median}\left|F - m\right|, \qquad
s = 1.4826 \cdot \mathrm{MAD}$$

$m$ is the typical brightness, $s$ is the typical scatter. Everything downstream is measured in
units of $s$, which we write as "MAD-scales".

**Why median and MAD instead of mean and standard deviation?** A standard deviation squares its
deviations, so a handful of genuine bursts inflate it — which raises the threshold and hides the
very events being hunted. The MAD is rank-based: one outlier moves it by at most one rank
position. It is robust up to ~50% contamination.

**Where 1.4826 comes from.** For normally distributed data,
$\mathrm{MAD} \to \sigma \cdot \Phi^{-1}(0.75) \approx 0.6745\,\sigma$. So
$\sigma \approx \mathrm{MAD}/0.6745 = 1.4826\,\mathrm{MAD}$. The constant just puts MAD onto the
familiar sigma scale, so "4 MAD-scales" reads like "4 sigma" without inheriting SD's fragility.

**Why per (gene, experiment) and not per experiment?** 10 of the 23 experiments imaged two genes
in different fields of view. Scaling per experiment hands a dim gene and a bright gene one
shared ruler. Fixing this alone moved chpS from 1.72 to 0.92 bursts per 1000 pairs.

---

## 4. Step three: the three tests

A frame $t^\ast$ on the envelope is a burst candidate when all three hold.

### Test 1 — Height: is it bright?

$$\mathcal{E}(t^\ast) > m + k\,s \qquad (k = 4.0)$$

### Test 2 — Local maximum: is it a peak?

$$\mathcal{E}(t^\ast) \ge \max_{t \in [t^\ast - W,\, t^\ast + W]} \mathcal{E}(t) \qquad (W = 5 \text{ frames})$$

### Test 3 — Prominence: did it actually rise?

Walk left from the peak until the signal exceeds it (or the trace ends) and record the
**lowest** value passed; do the same rightward. Call these $b_{\text{left}}$ and
$b_{\text{right}}$. Then

$$\mathrm{prom}(t^\ast) = \mathcal{E}(t^\ast) - \max\!\left(b_{\text{left}},\, b_{\text{right}}\right) \;\ge\; k\,s$$

### Height and prominence are different questions

This is the single most important idea in the method, and the easiest to confuse.

> **Height is altitude above sea level. Prominence is how far you must descend before you can
> climb higher.**
>
> Denver sits 1,600 m up but is not a mountain — walk in any direction and you barely go down.
> A 300 m hill on a plain is unmistakably a hill.

- **hupA is Denver.** Constitutively bright, so enormous height and near-zero prominence. It
  fails test 3 no matter how high the bar in test 1 is. *This is what makes the constitutive
  control work.*
- A dim gene that flares is the hill on the plain: modest height, large prominence.

Measured when the criterion was first tested: switching prominence off took the count from 80 to
167 and moved hupA from 0.00 to 1.32 bursts per 100 cells — from last place to 5th of 10.
(Those absolute counts are from the earlier 23-experiment frame in raw a.u.; the conclusion is
about the ordering, which nothing since has changed.) **Prominence is the load-bearing
criterion.**

A tempting "fix" that must not be made: using the *lower* of the two flanks, or only the
left-hand one, looks more permissive and is worse. Measured, both score hupA's brightest wobble
at **6.855** MAD-scales, *above* a real chpS burst at 6.061 — the rule stops separating them,
and no threshold can restore the ordering. Keep $\max$; move the bar instead.

> **This warning is about the absolute prominence, in MAD units — and only that.** The same
> change to the unit-free **ratio** does *not* invert the control: on left-base fold, hupA tops
> out at 3.372 while a real event reaches 3.787. §7's v6 entry uses exactly that, and the two
> measurements must not be confused. A difference in MAD-scales inherits the gene's brightness;
> a ratio does not.

### A worked example

Both trenches below are araB in `CL008_giTG068_072925`, where $m = 0.01571$ and
$s = 0.00377$ — so the height bar is $0.01571 + 4(0.00377) = 0.03077$ and the required
prominence is $4(0.00377) = 0.01506$ a.u./ms.

**What a burst looks like — FOV 003 trench 1598, track 12.** In a.u./ms:

| frame | 36 | 37 | 38 | **39** | 40 |
|---|---|---|---|---|---|
| $F$ | 0.0192 | 0.0539 | 0.0758 | **0.0763** | 0.0733 |

Frame 39 clears height (0.0763 vs 0.0308) and has prominence **14.70 MAD-scales** against a bar
of 4.0 — this is not a marginal call. Nine frames in that trench clear the height bar; the
detector reports **one** event, at frame 39. The other eight are all on the decay and fail the
±5-frame local-maximum test.

$F$ then falls to 0.0410 by frame 55 while the cell's area grows 384 → 743 px². $F \times$ area
is flat to within 4% — the cell made one pulse of protein and grew into it. The decay is
**dilution, not loss**, which is why the whole hump must count once.

**How near-misses get rejected — FOV 007 trench 821.** Exactly three frames clear the height bar
here, which makes the three tests easy to see side by side:

| frame | $\mathcal{E}$ | prominence | local max? | verdict |
|---|---|---|---|---|
| 35 | 0.0342 | 2.03 s | no | rejected — a shoulder on the way up |
| **39** | **0.0495** | **9.39 s** | yes | **burst** — the prominence is the whole rise |
| 43 | 0.0399 | 0.15 s | no | rejected — a wobble on the decay |

This is also the trench where the tracker split one bursting cell across six IDs (18 → 19 → 10
→ 28 → 29 → 24 over ten frames). A per-cell view shows four apparent bursts. The envelope shows
one hump, and the detector reports one event.

### How often each test bites

Under v3, of the peaks on the envelopes: **327 clear the height bar**, **178** of those are also
local maxima, and of those **94 pass prominence and 84 fail it**. So prominence decides
essentially every borderline case. `window` is nearly inert while prominence is on (80 bursts at
every $W$ from 1 to 8) — tune $k$, not $W$.

---

## 5. Step four: count each episode once

### Episode merge

A burst decaying over 30 frames must not count 30 times. Walking candidate peaks forward in
time, a later peak $p$ opens a **new** burst only if

$$\min_{t \in [p_{\text{prev}},\, p]} \mathcal{E}(t) < m + ks
\quad \textbf{(it returned to baseline, then rose again)}$$

$$\textbf{or} \quad \mathcal{E}(p) > \mathcal{E}(p_{\text{prev}}) + ks
\quad \textbf{(it climbed to a genuinely higher maximum)}$$

Otherwise it is the same decaying or plateauing episode, and is skipped.

### Concurrent merge (v2 onward)

Two events in one trench are the same burst when

$$|t_1 - t_2| \le 5 \text{ frames}
\qquad\text{and}\qquad
\frac{|y_1 - y_2|}{\tfrac{1}{2}(L_1 + L_2)} < 1.5$$

where $y$ is position along the trench and $L$ is cell length. The gap is in **cell lengths**,
not pixels, because no µm-per-pixel scale is recorded anywhere in this dataset — a ratio is the
only threshold that ports across acquisitions.

The **earlier** peak survives, so timing reports burst *onset* rather than being biased late by
whichever neighbour peaked brighter.

---

## 6. Step zero: one input filter

Applied before any definition runs, and not part of any definition's rule:

$$\text{drop a cell-frame if } \text{area} < 0.05 \times \text{that experiment's median cell area}$$

Sorted burst-cell areas were `[1, 1, 1, 2, 2, 117, 131, …]`. Only 0.11% of cell-frames are
≤ 5 px², yet 5 of 95 bursts landed on them — a 48× enrichment. On a one-pixel mask the "mean"
intensity is a single raw pixel with no averaging, which is exactly how a false peak clears a
threshold built from properly averaged cells.

**The floor is relative, not an absolute px² value**, because one acquisition
(`DUMM_giTG66_Glucose_012325`) was recorded at about one third the linear pixel scale of every
other. Its cells are geometrically sound, just recorded on bigger pixels. An absolute 50 px²
floor deleted 749 of its 844 perfectly good cell-frames; the relative floor drops 121 across the
whole dataset, of which 1 is from that run.

---

## 7. The definitions, in order

Each version changes exactly one thing. Nothing is ever edited in place — the older rules stay
so old results remain reproducible.

| | what changed | events |
|---|---|---|
| **v1** | the three tests + episode merge, on **one envelope per trench** | 73 |
| **v2** | one envelope per **lineage family**, then the concurrent merge | 89 |
| **v3** | families also split by **spatial gaps**. Measured: a no-op | 89 |
| **v4** | the height bar and the prominence bar **decoupled** | 99 |
| **v5** | a second **fold** route in, alongside the absolute one | 118 |
| **v6** | the fold ratio anchors on the **left** base, floored | 156 |
| **v7** | a fourth route on the trench's own **quartile fence** | 164 |
| **v8** | the same rules on the **FIJI values** (§1). Rejected — the ruler collapses | — |
| **v9** | **any signal** on the FIJI values, deduplicated in $(t, y)$ | 802 |
| **v10** | **v5's rule** on the FIJI values, scaled to the trench's own peak | 352 |
| **v11** | seeded on each trench's **lit** distribution, clustered, reported at the peak | 469 |

v8–v11 change the **measurement**, not the rule, so they are the first versions that are
*neither* supersets nor subsets of what came before. Do not read 352 against 156 as a definition
change, and do not put them on a `burst_rate_comparison` axis with v4–v7.

### v1 — one envelope per trench

The three tests and the episode merge, exactly as described above, on a single envelope per
trench.

**Its blind spot:** the envelope is one number per frame, so two bursts happening at once in one
trench collapse into a single hump and the dimmer one is invisible.

Real case — `DUMM_giTG059_060_061125` FOV 008 trench 1488, frames 50–59, holds two genuine
bursts in different parts of the trench, separated by a *dark* cell. v1 reported one: with a
single envelope the first burst is still high while the second peaks, so there is no saddle
between them and the second peak's prominence collapses far below the required $4s = 0.0171$
a.u./ms. Split by family, the same peaks measure **5.78** and **6.00** MAD-scales — unambiguous.

### v2 — one envelope per lineage family

A **lineage family** is a cell and all its descendants (a connected component of the
parent-daughter forest within one trench). v2 builds one envelope per family, so two unrelated
bursts no longer share a signal.

That alone over-splits: a single burst spanning two *adjacent* cells the tracker happened to put
in different families gets counted twice. Hence the concurrent merge (§5), which puts them back
together. Measured: **73 → 94** from per-family detection, then **→ 89** after merging 5 of the
7 concurrent pairs it found.

In trench 1488 that is exactly what happens: per-family detection finds peaks at frames 52
(family 40), 54 (family 33) and 55 (family 31); 54 and 55 are adjacent cells 1.41 cell lengths
apart, so they merge and the earlier survives. Final answer, two bursts, at frames 52 and 54 —
which is what the images show.

### v3 — spatial clusters inside a family

v2 still has a blind spot: two co-bright cells that are far apart *within the same family*. v3
subdivides each family by gaps along the trench (a new cluster wherever the gap between adjacent
cells exceeds 1.5 cell lengths), linking clusters across consecutive frames that share a cell.

**Measured, this is a no-op — v3 finds exactly the same 89 events as v2.** Recorded rather than
hidden, because the reason is informative: cells in a trench are physically contiguous, so any
partition by position gets re-joined by cell identity. Splitting on position more aggressively
instead *shatters* the signal into fragments too short for peak detection. The v2 partition is
at a local optimum for this data.

### v4 — stop making one number do two jobs

In v1–v3 the single $k = 4.0$ set **both** the height bar and the required prominence. These are
different questions ("is it bright" vs "did it rise"), and they had never been tested apart.

v4 splits them: $k = 4.0$ for height, $k_{\text{prom}} = 3.7$ for prominence.

**Why 3.7.** Lowering only the prominence bar has a plateau:

| $k_{\text{prom}}$ | events | hupA | mazF |
|---|---|---|---|
| 4.0 | 89 | 0.00 | 0.00 |
| 3.8 | 97 | 0.00 | 0.00 |
| **3.7** | **99** | **0.00** | **0.00** |
| 3.6 | 101 | 0.00 | 0.00 |
| 3.5 | 107 | **0.54** ✗ | 0.00 |

hupA's brightest candidate sits at **3.586** MAD-scales, which is exactly where 3.5 breaks the
control. 3.7 is the middle of the safe plateau, so it is not tuned to any single event.

This caught a real miss: chpS 010/1188 frame 25, which clears height at 5.38 MAD-scales, is a
local maximum, and had prominence **3.851** against a bar of 4.0 — short by 3.7%.

### v5 — a second way in, for dim trenches

Every gate up to here is **absolute within a gene**: a peak must clear $m + ks$ for its
(gene, experiment). So a trench whose *entire* signal sits below its gene's median can never
register a burst, however dramatic the excursion inside it.

This is not rare. **Only 109 of 255 trenches (43%) can ever reach their own height bar.** By
gene: lacZ 87%, hupA 83%, gfcE 54%, alkA 52%, chpS 47%, araB 41%, gldA 32%, murQ 30%, baeS 25%,
mazF 0%.

The case that forced this: alkA `DUMM_giTG069_063_121125` FOV 001 trench 1192. A cell climbs
0.050 → 0.176 a.u./ms — a **3.5× rise**, and the brightest cell-frame in its whole trench — yet
it stands only **1.91 MAD-scales** above its gene's median. It produced *zero* height-clearing
candidates, so prominence was never even evaluated.

v5 adds a **fold route**, OR'd with the absolute one:

$$\frac{\mathcal{E}(t^\ast)}{\max(b_{\text{left}},\, b_{\text{right}})} \ge R
\qquad\text{and}\qquad
\mathcal{E}(t^\ast) \ge m + 1.0\,s
\qquad (R = 3.3)$$

That is prominence again, but as a **ratio** instead of a difference.

**Why this is safe.** A ratio is unit-free, so one bar serves the whole 283× brightness range
with no baseline at all — and a flat-high trace still scores ≈ 1.0, so it keeps exactly the
property that makes prominence work. Measured over all 1,941 local maxima:

| gene | best fold |
|---|---|
| **hupA** (control) | **1.76×** |
| **mazF** (control) | **2.94×** |
| **chpS** (control) | **3.10×** |
| the alkA case above | **3.60×** |

The safe window is therefore $(3.10,\ 3.60]$, and **3.3 is its midpoint** — about 7% clear on
each side. That band is narrow, and it is the honest limit of the criterion: quote it whenever
the number is quoted.

**The second condition is not optional.** A ratio divides by its base, so without a floor a
noise wobble in a near-zero trench manufactures an arbitrarily large fold. Requiring the peak to
sit at least 1 MAD-scale above the gene median guards the denominator.

**Fold cannot replace the absolute route.** With the absolute bar made unreachable, fold alone
finds 48 events and fails a regression trench outright, because it misses the bright canonical
bursts. It only ever *adds*: **v5 = 99 absolute + 19 fold = 118, and is a strict superset of
v4.**

Two merge interactions this needed, both found by testing rather than inspection:

- The episode merge reopens when the envelope drops below $m + ks$ — which a fold event's
  envelope never reaches, so every wobble on its decay would open a new burst. A fold episode
  resets at $\mathcal{E}_{\text{prev}}/R$ instead: it came back down by the factor that
  qualified it.
- Both merges now prefer an **absolute** candidate as the survivor. Otherwise a fold candidate
  on a rising shoulder wins on onset-ordering and absorbs the confirmed bright peak behind it.

### v6 — the ratio anchors on the left base

Both prominence (§4, test 3) and v5's fold divide by, or subtract, $\max(b_{\text{left}},
b_{\text{right}})$. §4 defended that `max` and it was right to: using the *lower* flank breaks
the constitutive control. But `max` has a side effect nobody had looked at.

**Where the two bases come from.** Walk left from the peak until the signal exceeds it, and
record the lowest value passed — that is $b_{\text{left}}$, the trough the burst rose *out of*.
Walk right the same way for $b_{\text{right}}$, the trough it decayed *into*. For a burst these
are not symmetric, and §1 already said why: after a burst, $F$ falls by **dilution as the cell
grows**, not by protein loss. That decay is slow — the araB worked example in §4 is still at
0.0410 sixteen frames past its peak of 0.0763 — and it is often still unfinished when the
recording ends.

So $b_{\text{right}}$ is *usually* the higher of the two, and `max` therefore picks it. The
measure ends up asking "how far above the tail did it get?" instead of "how far did it rise?" —
and the longer the burst, the more it is penalised for being one.

> **Height is altitude, prominence is how far you must descend (§4). The base you divide by is
> the question of *which direction you look*.** Look back and you ask what the cell rose from.
> Look forward and you ask what it has not finished falling to yet.

"Usually" is the honest word, and it is worth the precision: of the 84 peaks v3 rejected on
prominence the right base is the deciding one in **53** — a 63% majority, not a rule. So this
is the single largest rejection mode rather than a universal bias, and a left-base route is
expected to *add* a class of events, not to relabel every rejection.

v6 therefore adds a **third route**, OR'd with the other two:

$$\frac{\mathcal{E}(t^\ast)}{\max\!\left(b_{\text{left}},\; f\,m\right)} \ge R_L
\qquad\text{and}\qquad
\mathcal{E}(t^\ast) \ge m + 1.0\,s
\qquad (R_L = 3.6,\; f = 0.5)$$

`fold_prom` and the absolute rule are untouched, so **v6 is a strict superset of v5** by
construction: 99 absolute + 19 fold + 38 fold-left = **156** events.

**It keeps the property that makes the control work.** A flat-high trace has *no* trough on
either side, so $b_{\text{left}} \approx \mathcal{E}(t^\ast)$ and the ratio is ≈ 1.0. Looking
left does not make hupA a burster; hupA's best left-fold anywhere is 3.372. Note carefully that
this is **not** the left-base rule §4 rejects — that was the *absolute* prominence, in MAD
units, where left-only scored hupA's brightest wobble at 6.855 against a real chpS burst at
6.061. A difference in MAD-scales inherits the gene's brightness; a ratio does not. Keep the
two claims apart.

#### The worked example

gfcE `DUMM_giTG060_064_121425` FOV 013 trench 153, where $m = 0.04566$ and $s = 0.01803$, so
the height bar is $0.04566 + 4(0.01803) = 0.11778$. The envelope, in a.u./ms:

| frame | 0 | 4 | **7** | 12 | 16 | 20 | 24 | 28 | **31** | 36 | 44 | 48 | 52 | **56** | 60 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $\mathcal{E}$ | .036 | .030 | **.028** | .038 | .056 | .063 | .055 | .075 | **.105** | .067 | .072 | .086 | .068 | **.052** | .096 | .067 |
| | | | ↑ left base | | | | | | ↑ peak | | | | | ↑ right base | | |

The trench climbs out of a genuine trough at frame 7 (0.0277), peaks at frame 31 (0.1048), and
then **never returns to that floor** — 25 frames later the lowest it has managed is 0.0520.
That is the asymmetry, in one row of numbers.

| test | value | bar | |
|---|---|---|---|
| height | 3.279 $s$ | 4.0 | fails |
| prominence | 2.927 $s$ | 3.7 | fails |
| fold, $\max$ base ÷ 0.0520 | 2.015 | 3.3 | fails |
| **fold-left, ÷ 0.0277** | **3.787** | 3.6 | **passes** |

The trench's whole-trace maximum is 0.10478 — below its gene bar (0.11778) *and* below its own
v5local bar (0.11090). So it could never produce a height-clearing candidate, and prominence
was never evaluated: this is §8's blind spot one layer deeper, and it is why the trench appears
in **no** table before v6.

#### Why the denominator needs a floor

A ratio divides by its base, so a near-zero base manufactures an arbitrary number. Unfloored,
chpS 010/46 has $b_{\text{left}} = 0.006$ and scores **30.0×**; mazF reaches 14.0×. Flooring
the denominator at $f\,m$ fixes it:

| $f$ | hupA/mazF best | chpS best | the target |
|---|---|---|---|
| none | 14.02 | 30.04 | 3.787 |
| 0.25 | 4.32 | 6.12 | 3.787 |
| **0.5** | **3.372** | **4.22** | **3.787** |
| 0.75 | 2.44 | 3.65 | 3.059 ✗ |

**$f$ is bracketed on both sides**, which is what makes 0.5 a choice rather than a free knob.
Below it the controls stay loose. Above it the floor overtakes the real base: at $f = 0.75$ the
floor is $0.75(0.04566) = 0.0342$, which is *higher* than the target's own left base of 0.0277,
so the floor becomes the denominator and the event drops to 3.059. At $f = 0.5$ the floor is
0.0228, safely under it. Only 0.5 does both jobs.

#### Why 3.6

hupA and mazF stay at **zero** candidates for every bar $\ge 3.4$, and the target is lost above
3.79. The band is **[3.4, 3.79]**, and 3.6 is its mid-plateau.

Be honest about what that is worth. The lower edge is set by a control gene's best wobble, which
is a real distribution; the upper edge is set by **one event**. So the guard is ~5% either side
— *narrower* than v5's (3.10, 3.60], and v5's was already called the honest limit of its
criterion. Quote the band whenever the number is quoted.

Nothing is given up to reach it: alkA 001/1192 frame 9, v5's own motivating case, has
`fold_left` 3.703. chpS 010/1188 frame 25 sits at 3.563, below the new bar, but the absolute
route already had it.

#### What it costs, and what it does not fix

Per gene, against v5: alkA 13 → 23, gldA 15 → 24, gfcE 20 → 28, baeS 7 → 14, lacZ 36 → 38,
chpS 4 → 5, murQ 1 → 2, araB 22 → 22. **hupA and mazF stay at 0.** The gains concentrate in the
genes with the lowest trench eligibility (gldA 32%, alkA 52%), which is the mechanism behaving
as designed rather than a coincidence.

The cost is a **ranking change**: gldA overtakes lacZ, 0.070 against 0.055 per cell cycle. The
claim "lacZ is top under every definition" held through v5 and does not survive v6 — §11 is
written to say so. The intervals overlap, so the new order is not resolved either; what v6
establishes is that the old order was partly a statement about coverage.

v6 does **not** replace the per-trench baseline. It and v5local are not nested — 113 shared, 43
v6-only, 68 v5local-only — because they change different things: v5local moves the *baseline*,
v6 changes the *base of the ratio*. It does resolve 9 of the 77 events that were previously
v5local's alone, which means those 9 were never evidence for re-scoping the baseline at all,
only evidence that the max-base ratio was too strict.

One merge interaction, and it is mechanical: the route-dependent episode reset becomes a
route → divisor lookup, and both merges rank `absolute < fold < fold_left` through `ROUTE_RANK`
instead of testing `route != 'absolute'`. Bit-identical for v1–v5.

### v7 — a quartile fence on the trench's own distribution

Every route in v1–v6 is a location-and-scale statement about the (gene, experiment) median: a
height above `m + ks`, or a ratio against a peak's own flanks. §8 records four attempts at
re-scoping that median to the trench, none shipped. v7 changes the *statistic* instead — each
peak must clear its own kymograph's **Tukey fence**:

$$\mathcal{E}(t^\ast) > Q_3 + \kappa\,(Q_3 - Q_1) \qquad (\kappa = 4.0)$$

computed over that trench's cell-frame distribution of $F$, on the same column the detector runs
on. OR'd in as a fourth route, so **v7 is a strict superset of v6**: 156 → **164**, +8 events,
0 lost, all 8 tagged `route == 'tukey'`.

It exists because the fence reaches what no height bar reaches. On cell-frames it flags gfcE
013/153 frame 31 (2 of 300, one of them the v6 peak) and alkA 001/1192 frame 9 (4 of 275) —
both invisible to the gene bar *and* to a per-trench MAD bar. At the event level v6's ratio
routes already had both, so v7's 8 are a different set again.

#### κ is 4.0, and the textbook 1.5 is not usable

| κ | events | added | hupA | mazF | chpS | regression trenches |
|---|---|---|---|---|---|---|
| 1.5 | 523 | +367 | 5 | 16 | 43 | **broken** |
| 2.5 | 247 | +91 | 2 | 3 | 9 | **broken** |
| 3.0 | 196 | +40 | 2 | 2 | 7 | **broken** |
| 3.4 | 175 | +19 | 2 | 1 | 6 | ok |
| **4.0** | **164** | **+8** | **1** | **0** | **5** | ok |
| 4.4 | 160 | +4 | 0 | 0 | 5 | ok |
| 4.8 | 159 | +3 | 0 | 0 | 5 | ok |
| 5.0 | 156 | +0 | 0 | 0 | 5 | ok |

Every row is a strict superset of v6 with zero losses. Read the table honestly, because it says
something no earlier route had to admit: **there is no κ at which this route both adds events and
leaves the constitutive control untouched by more than a hair.** The control-clean band is
**[4.4, 4.8]** and buys only 4 events; at the textbook 1.5 it is not close — mazF alone takes 16
and all three regression trenches break.

**4.0 is a deliberate call, not a plateau midpoint.** It doubles the yield to +8 for exactly one
hupA event, on the argument that a constitutive gene can still burst and a hard zero is this
project's convention rather than a law. `tukey_iqr_k=4.4` gives the control-clean variant.

Know what that one event is, because it is the textbook failure mode: hupA `DUMM_giTG068_063_061725`
012/1165 frame 81, $F = 7.73$ against a gene bar of 6.17. **Bright but not prominent** — Denver,
in §4's terms. It clears the height bar and fails prominence, and the fence cannot tell the
difference, because a marginal distribution has thrown the time axis away. That is the standing
limitation of this whole route, not a tuning artefact.

#### The rate, and the one ranking claim it breaks

Bursts per cell cycle under v7: **gldA 0.073**, lacZ 0.057, alkA 0.049, gfcE 0.041, murQ 0.028,
araB 0.024, chpS 0.014, **hupA 0.011**, baeS 0.011, mazF 0.000. Total 0.031 over 5,279 divisions.

gldA stays top and mazF stays at a hard zero, but **v7 moves a constitutive control off the
bottom**: hupA goes from 0.0000 under v6 — tied last with mazF — to **0.0114**, which places it
8th, *above* baeS at 0.0112. One event, on the smallest denominator in the set (88 divisions).

Do not read that as hupA bursting more than baeS. The estimates differ by 0.0002, and hupA's
interval is 0.0003–0.0633 against baeS's 0.0061–0.0187 — not distinguishable, and hupA's is wide
precisely because it rests on one event. What it does establish is the cost of κ = 4.0: the claim
**"baeS/hupA/mazF are bottom under every definition" held v1 through v6 and does not survive v7**
(§11). `tukey_iqr_k=4.4` restores it, for 4 events instead of 8.

`figures.plot_burst_rate_comparison(bursts, divisions, versions=('v6','v7'))` draws the grouped
bars; colour encodes the definition there, since gene is on the x-axis.

#### What it does not disturb

gfcE 013/153 keeps `route == 'fold_left'` and alkA 001/1192 keeps `route == 'fold'`, so v7
relabels nothing v6 found — `setdefault` is called for `tukey` after both fold routes, and
`ROUTE_RANK['tukey'] = 3` ranks it last so it is always the candidate dropped in a concurrent
merge and never an episode's representative. chpS and mazF are unchanged from v6 entirely.
v1–v6 are bit-identical at 73/89/89/99/118/156.

#### Two implementation traps, both silent

1. **The fence is evaluated before the fold routes' height floor**, not after. That floor is
   `med + 1.0·s` — a height condition on the gene baseline — and inheriting it would defeat the
   route's entire premise. Measured, it costs exactly the two near-misses v7 exists for.
2. **The episode reset for a tukey peak is the fence itself**, an absolute floor, not a divisor.
   `fold`/`fold_left` fall back by the factor that qualified them; a fence has no factor. Left
   unextended it falls through to `reset = height`, which a tukey peak never reaches, so the
   reset test is always true and every wobble on a decay opens a new burst — inflated counts, no
   error.

And one units trap: the fence must be computed on `bdef['fluor_col']`. Raw and per-ms fences give
identical *verdicts* but differ by **150–500×**, so a fence built on `intensity_mean_fluor` and
compared against `fluor_per_ms` is wrong by up to 500× and raises nothing.

### v8 — the MAD ruler does not survive the change of measurement

The first attempt on the FIJI values (§1) kept the rules and swapped only `fluor_col`. **It does
not work, and the reason is worth keeping.**

Every route in v1–v7 is anchored on $m + ks$ for a (gene, experiment). On `fiji_mean` that
baseline **collapses**: $m = 0$ and $s$ floored at $10^{-6}$ for 11 of 28 groups, so the height
bar is $\approx 4\times10^{-6}$ and every nonzero blip clears it. The ratio routes fail the other
way — $\max(b_\text{left}, b_\text{right})$ is **0** when a trace rises from exactly zero, so
v5's and v6's `base > 0` guard rejects precisely the sharpest rises. A per-kymograph baseline
(§8's idea) does not rescue it either: ~35k rows still carry zero scale and the controls get
*worse*.

v8 tried to repair it with a floored ratio denominator. **34 settings** were swept — $R \in
\{2,3,4,6,10\}$ × floor fraction $\in [0.05, 0.30]$ × gate $\in \{0.02,\dots,0.15\}$, on both
`fiji_mean` and `fiji_max` — and **not one is both constitutive-clean and able to find the three
regression trenches.** The tradeoff is structural, not a tuning failure: every setting that finds
3/3 admits 9–17 mazF events, and every setting that holds mazF at 0 finds at most 1 of 3. v6
recall peaks at **54.5%**.

**The diagnosis is dimensional, and it is the lesson.** On the three regression trenches the
`fiji_mean` envelope maxima are 2.06, 0.73 and 6.48 counts, against display spans of 29, 22 and
25. `fiji_mean` is averaged over a **cell**; the display span is a per-**pixel** range. A
denominator floored at a fraction of the span is therefore the same size as the peak itself, and
no multiplier can fix it. v10 uses a scale drawn from the cell-mean distribution instead.

**Not shipped.** No event CSV exists, and it is absent from the viewer. `BURST_DEF_V8` is kept in
the registry so the negative result is not re-derived.

### v9 — any signal, deduplicated in $(t, y)$

If the transform of §1 has already done the thresholding — clip, median — then **what survives
is the candidate set**, and no ruler is needed at all. v9 drops height, prominence and ratio
together, which is also what makes it immune to the collapse that sank v8.

A candidate is any cell-frame with $\text{fiji\_mean} > 0$. That needs no choice of statistic:
`fiji_mean`, `fiji_max` and `fiji_frac` are positive on the identical 32,979 cell-frames (§1).

Deduplication is **spatial-temporal connected components**, not the concurrent merge of §5: two
lit cell-frames join when

$$|\Delta t| \le 2 \quad\text{and}\quad |\Delta y| \le 1.5\,\bar{L}\,|\Delta t|$$

with $\bar{L}$ the **trench's own** median `axis_major_length`. Each component is one event,
reported at its earliest frame. The $|\Delta t|$ scaling is load-bearing: a track drifts **−4.7
px/frame** (median; −17 at the 10th percentile), so a fixed tolerance fragments any advecting
burst. Using the trench's own cell length keeps it scale-free, which the ~⅓-scale
`DUMM_giTG66_Glucose_012325` needs — the trap §6 records for the area floor.

**The gate is mandatory, and it is what the per-trench floor makes it.** Because the floor is
25% of each trench's *own* max, a uniformly bright trench lights up entirely, so the lit fraction
runs **inversely** to burstiness:

| | hupA\* | mazF\* | chpS\* | baeS | murQ | gfcE | lacZ | araB |
|---|---|---|---|---|---|---|---|---|
| lit fraction | **99.9%** | 85.1% | 75.1% | 65.8% | 33.2% | 26.1% | 13.6% | 4.9% |

\* constitutive control.

So v9 rejects any trench above `lit_gate` lit. The sweep:

| gate | events | REG | hupA\* | mazF\* | chpS\* | v6 coverage |
|---|---|---|---|---|---|---|
| 0.60 | 648 | 3/3 | 0 | 5 | 0 | 74% |
| 0.70 | 691 | 3/3 | 0 | 9 | 6 | 78% |
| 0.80 | 743 | 3/3 | 0 | 9 | 8 | 84% |
| **0.90** | **802** | **3/3** | **0** | **9** | **16** | **96%** |

**0.90 dominates rather than trades.** hupA is 0 at *every* value — it cannot break until 0.994,
its own minimum lit fraction — and mazF is **flat at 9 from 0.70 up**, because mazF's lit
fractions jump from 0.31 straight to 0.98+, so the same single low-lit trench is the only one
admitted anywhere in that range. Raising the gate costs nothing on the controls and recovers
chpS from 0 to 16, which matters because chpS *does* burst (§7, v6's worked example is chpS
010/1188's neighbour).

**What it does not do: agree with v6 event by event.** Coverage of 96% means v6's event
cell-frames fall inside v9's *candidate set*. Exact row agreement is only **16 of 156**, because
v9 reports burst **onset** where v6 reports the **peak** frame — the same physical burst, a
different `time_frame`. Those two numbers are not interchangeable and should never be quoted as
one.

**Superseded by v10**, but **v9 remains the better control performer** (mazF 9 vs 13, and a far
wider clean band), and per cell cycle it is the only FIJI-values definition under which all three
constitutive genes rank last of ten. Events in `data/bursts_v9_nospeckle_090926.csv`. Caveats:
39% of its events are single cell-frames, and duration is unreliable at the top end — the longest
spans 87 frames, which is an over-merged lit trench rather than one burst, since at a two-frame
gap the tolerance reaches 3 cell lengths.

### v10 — v5's rule, on the new measurement

v9 asks *is anything lit*. The filtered traces show something more specific: **sharp rises**. v5
already has the machinery for that — the question is only what to measure the rise *against*,
since the MAD unit does not exist here.

v10 is **v5 verbatim** — same per-lineage-family × spatial-cluster envelope, same $W = 5$, same
episode merge — with exactly one substitution:

$$m = 0, \qquad s = P \equiv \max_t \mathcal{E}(t) \ \text{for that trench}$$

so $k$ and $k_\text{prom}$ become **fractions of the trench's own brightest cell-mean**. This is
why they look nothing like v5's 4.0 and 3.7. The height test is $\mathcal{E} \ge 0.60\,P$ and the
rise test is $\mathrm{prom} \ge 0.40\,P$; the second **is** the sharp rise. Attached by
`attach_fiji_scale`, read through `detect_bursts`' existing `baseline_cols`, so no new detector
exists.

| $k$ | $k_\text{prom}$ | events | REG | hupA\* | mazF\* | chpS\* | v6 exact rows |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.20 | 695 | 3/3 | 7 | 32 | 68 | 118/156 |
| 0.40 | 0.30 | 519 | 3/3 | 2 | 20 | 49 | 105/156 |
| **0.60** | **0.40** | **352** | **3/3** | **0** | **13** | **35** | **91/156** |
| 0.75 | 0.40 | 290 | 3/3 | 0 | 12 | 26 | 80/156 |

\* constitutive control.

**91 of 156 v6 events matched at the exact frame**, against v9's 16 — because keeping v5's
envelope-and-peak machinery means v10 reports the same representative frame v6 does. That is the
strongest frame-level agreement any FIJI-values definition reaches, and it is the argument for
v10 over v9.

**Why the trench's own peak and not something else.** Two other scales were tried and both fail
for v8's reason. The display span $v_\text{max} - v_\text{min}$ is a per-pixel range (~25 counts)
against an area-averaged `fiji_mean` (~2–6): **0** workable settings. The p90 cell-mean reaches
only **1 of 3** regression trenches. The trench max is the only scale tried that is both
non-collapsing and dimensionally matched.

**`fold_prom` is off, and measured inert.** These traces rise from exactly 0, so
$\max(b_\text{left}, b_\text{right}) > 0$ always fails and the fold route admits nothing — the
event set is identical with and without it. All 352 come by the `absolute` route, which here
means "reached 0.60 of the trench peak and rose by 0.40 of it".

**Be honest about the band.** hupA holds at 0 only for $k \ge 0.60$ at this $k_\text{prom}$, and
**two points in that band were tested, not a fine grid** — narrower than v9's 0.60–0.90 plateau
and narrower than v7's κ band. mazF (13) and chpS (35) are **not** zero. v10 buys frame-level
agreement and pays for it in control cleanliness; §9 and §11 both record what that breaks.

8 trenches whose maximum `fiji_mean` is 0 have no usable scale and are skipped. Events in
`data/bursts_v10_090926.csv`.

---

### v11 — seed on the trench's own lit distribution, cluster, report the peak

v9 and v10 are opposite errors on the same measurement, and v11 is what is left when both are
removed.

**The diagnosis.** v10's bar is $0.60\,P$ where $P$ is the trench's single brightest cell-mean.
That is one order statistic, so it moves with whatever the trench's largest excursion happens to
be. Measured: v10 returns a **median of one event per trench** (mean 1.74, max 8, over 202
trenches), and in **157 of the 247 trenches that have any lit cell-frame its bar sits above that
trench's own lit Tukey fence** — across most of the dataset v10 is stricter than "an outlier
among this trench's own lit values", so a trench's second and third excursions cannot register.
The scale is also self-referential in the other direction: **54 of v10's 352 events are in
trenches whose entire maximum is below 1 count**, and 23 have a peak below 0.1 counts, because
$0.60 P$ is small whenever $P$ is.

v9 has the opposite problem — no bar at all — plus a reporting convention that has been
misread. Its 16-of-156 exact-frame agreement was blamed on the candidate set. It is not:

| v9's candidate set and clustering, reported at… | v6 events matched at the exact frame |
|---|---|
| the **onset** (what v9 ships) | 13 / 156 |
| the **peak** | **89 / 156** |
| *(v10, for reference)* | 91 / 156 |

Same events, same clustering, one line different. **v9's candidate set was never the problem**,
and §7's "the strongest frame-level agreement any FIJI-values definition reaches, and it is the
argument for v10 over v9" does not survive this measurement. (The 13 here is measured on the
nospeckle values; the 16 quoted in v9's entry is the earlier speckle-on file.)

**The rule.** v9's structure, with a bar and peak reporting:

1. A cell-frame is a **seed** when it clears its own trench's $m_{\text{lit}} + k\,\sigma_{\text{lit}}$,
   where the median and $1.4826\times$MAD are taken over that trench's **lit cell-frames only**.
2. Seeds are linked into events by v9's rule verbatim — $|\Delta t| \le 2$ and
   $|\Delta y| \le 1.5\,\bar{L}\,|\Delta t|$, shared code (`_link_ty_components`).
3. Each event is reported at its **peak** cell-frame.
4. A seed cluster must span at least `min_frames` distinct frames.
5. `lit_gate` still rejects a trench lit above a given fraction.

**Conditioning on lit is the whole idea, and it is the repair of v8's diagnosis rather than a
repeat of it.** v8 died because $m$ and $s$ were estimated over a **zero-inflated** column: 49%
of cell-frames are exactly 0, so the median is 0, the MAD floors at $10^{-6}$, and every blip
clears $4\times10^{-6}$. Restricting the estimator to the positive part removes the zero
inflation, and unlike the rejected $v_\text{max}-v_\text{min}$ scale it stays dimensionally
matched — a cell-mean bar for a cell-mean measurement.

Growing each event out through the surrounding lit cell-frames gives `n_lit_cell_frames`,
`lit_first`, `lit_last`. **Reported, never gating.** In a mostly-lit trench the lit component is
the whole trench (median 160 cell-frames, max 808), which is precisely v9's over-merge; seeding
first is what stops it becoming the event.

#### The lit Tukey fence fails the regression, and it is worth knowing why

The obvious conditional statistic is Tukey's fence on the lit values. It fails outright — **1 of
3 regression trenches at every multiplier from 1.5 to 6.0**, at every gate.

| regression trench | lit | max | lit Tukey ($\kappa=1.5$) | lit median + 4·MAD |
|---|---|---|---|---|
| `CL008…072925` 003/1598 | 24/451 (5%) | 2.062 | **2.080** ✗ | 0.935 ✓ |
| `CL008…072925` 007/821 | 16/277 (6%) | 0.732 | **0.804** ✗ | 0.233 ✓ |
| `DUMM…061125` 008/1488 | 97/432 (22%) | 6.475 | 1.982 ✓ | 1.550 ✓ |

On a trench that is only 5% lit, the burst *is* a large share of the lit values, so it inflates
$Q_3$ and the IQR together and the fence lands **above the burst itself**. The median and MAD are
not dragged up the same way. This is the mirror image of v8's failure — v8's estimator collapsed
toward zero, this one is pulled up by the signal — and it is why the shipped rule is `mad`.

`otsu` on `log1p` of the lit values passes 3/3 and is **knob-free**, which by this document's own
constant-selection standard is a real virtue. It is not shipped because it costs controls at
matched recall: at gate 0.95 / `min_frames` 2 it gives 542 events with mazF 13 and chpS 44,
against `mad`'s 469 with 9 and 34. It stays selectable as `seed_rule='otsu'`.

#### The constants, each with its band

| knob | shipped | band, measured |
|---|---|---|
| `seed_rule` | `mad` | `tukey` fails REG at every setting; `otsu` passes but costs controls |
| `k` | 4.0 | **no plateau** — see below. Set by precedent, not fitted |
| `lit_gate` | 0.95 | hupA is 0 for **every** gate up to 0.995 |
| `min_frames` | 2 | monotone; 2 is the minimum statement of persistence |
| `frac_min` | 0.0 | **measured inert** |
| `link_dt`, `link_dy_cell_lengths` | 2, 1.5 | inherited from v9, not re-tuned |

**$k$ = 4.0 is inherited, not fitted** — it is the same multiplier v1–v7 use for the height bar.
There is no safe plateau to sit in the middle of; at gate 0.95 / `min_frames` 2 the sweep is
monotone in both directions:

| $k$ | events | hupA | mazF | chpS | v6 exact |
|---|---|---|---|---|---|
| 3.0 | 550 | 0 | 11 | 49 | 93 |
| 3.5 | 496 | 0 | 9 | 39 | 91 |
| **4.0** | **469** | **0** | **9** | **34** | **89** |
| 4.5 | 446 | 0 | 9 | 29 | 87 |
| 5.0 | 410 | 0 | 9 | 25 | 83 |

So $k$ is an honest recall/control trade knob and it is set by precedent. Say so when quoting it.

**The `lit_gate` band is wide, which is the one place v11 is on firmer ground than v10.** At
$k=4.0$, `min_frames` 2:

| gate | events | REG | hupA | mazF | chpS | trenches |
|---|---|---|---|---|---|---|
| 0.90 | 442 | 3/3 | 0 | 9 | 30 | 163 |
| **0.95** | **469** | **3/3** | **0** | **9** | **34** | **174** |
| 0.99 | 481 | 3/3 | 0 | 11 | 40 | 181 |
| 0.995 | 481 | 3/3 | 0 | 11 | 40 | 181 |
| 1.01 (no gate) | 487 | 3/3 | **5** | 12 | 40 | 185 |

hupA breaks only when the gate is removed entirely, because **hupA's own minimum lit fraction is
0.9959** — the same mechanism v9's entry records. That band is far wider than v10's (hupA clean
only for $k \ge 0.60$, two points tested) and wider than v9's 0.60–0.90.

**`min_frames` is a genuinely new axis** — no swept duration gate appears anywhere earlier in this
document. It is also what lets the gate be relaxed from v9's 0.90 to 0.95. At gate 0.95:

| `min_frames` | events | mazF | chpS | v6 exact |
|---|---|---|---|---|
| 1 | 789 | 18 | 51 | 106 |
| **2** | **469** | **9** | **34** | **89** |
| 3 | 353 | 6 | 24 | 81 |
| 4 | 262 | 3 | 18 | 67 |

**Two new axes were tried and bought nothing. Both are recorded so they are not re-tried.**
`fiji_frac` at the peak had never been thresholded by any definition — §1 notes the three FIJI
columns share one positive set, which is true only until a threshold is applied. Applied, it is
inert: `frac_min` 0.2 removes 37 events (469 → 432) and changes the controls by **nothing at
all** (mazF 9, chpS 34 either way). `sparse_admit` — whether a trench with fewer than 5 lit
cell-frames admits all of them or is skipped — gives **identical event counts at every swept
setting**, because `min_frames` ≥ 2 already removes those trenches.

Link tolerance is inherited from v9 and was checked, not tuned: $(\Delta t, \Delta y)$ of
$(1, 1.0)$ / $(2, 1.5)$ / $(3, 2.5)$ gives 601 / 469 / 402 events with v6 exact frames 85 / 89 /
89. The count moves by 1.5×, the frame agreement barely.

#### What it is

**469 events over 174 trenches, REG 3/3, hupA 0, mazF 9, chpS 34.** Against v10:

| | v10 | v11 |
|---|---|---|
| events | 352 | **469** (+33%) |
| trenches | 202 | 174 |
| events per trench (median / mean) | 1 / 1.74 | **2 / 2.70** |
| hupA / mazF / chpS | 0 / 13 / 35 | 0 / **9** / **34** |
| v6 events matched at the exact frame | 91 / 156 | 89 / 156 |
| v6 events matched within ±5 frames | 0.692 | **0.763** |
| events in trenches whose own max < 1 count | 54 | **26** |
| mazF rank by rate | 7 of 10 | **9 of 10** |
| chpS rank by rate | **3 of 10** | 5 of 10 |

Per gene: baeS 108, gfcE 104, gldA 55, alkA 49, araB 43, lacZ 43, chpS 34, murQ 24, mazF 9,
hupA 0. Seed-cluster duration: median 4 frames, p90 12, max 39. 219 of v10's 352 events survive
at the exact frame.

v11 finds a third more events at two per trench instead of one, with better controls and better
±5 coverage, and it costs two exact-frame matches and 28 trenches. The trenches it drops are
those whose only excursions are single-frame or below their own trench's bar.

**Be honest about what is doing the work.** hupA is 0 because the `lit_gate` **discards all six
of its trenches wholesale**, not because the bar discriminates inside them — the same blunt
mechanism v9 uses. mazF (9) and chpS (34) are **not** zero, and chpS still ranks 5th of ten by
rate, so **v9 remains the only FIJI-values definition under which all three constitutive genes
rank last**. v11 is neither a superset nor a subset of v9 or v10; §7's warning against putting
FIJI-values definitions on a `burst_rate_comparison` axis with v4–v7 applies to it unchanged.

Events in `data/bursts_v11_090926.csv`; the viewer union, which replaced v10's layer, in
`data/bursts_viewer_v11_090926.csv`. The v11-vs-v10 adjudication worklist — every event either
finds, tagged `both` / `v11_only` / `v10_only`, with each peak's percentile within its own
trench's lit values and the absolute viewer frame — is `data/burst_worklist_v11_090926.csv`.

---

## 8. The v5-local variant (evaluated, not shipped)

Since trenches vary so much, why not scale to each **trench** instead of each (gene, experiment)?
The premise checks out: within one (gene, experiment) the median trench brightness spans a median
of **2.11×**, and up to **10.99×** (mazF).

Run v5 against a per-trench median and MAD — floored at 25% of the gene's, or a quiet trench sets
its bar to nothing and the control breaks (mazF goes to 0.19) — and you get **181 events**, with
coverage up from 43% to 59% and both hard controls still at zero.

Against shipped v5 that is **+77 gained, −14 lost**, and the mechanism is exactly right: gained
events sit in trenches *quieter* than their gene (median local-to-global scale ratio **0.63**,
75 of 77 below 1.0), lost ones in *noisier* trenches (median **1.19**).

**Why it is not shipped:**

1. It **loses** the chpS 010/1188 case — that trench's own MAD is larger than its gene's, so the
   same event drops from 5.38 to 3.66 MAD-scales, under the bar. Five trenches go completely dark.
2. It does **not** reach the class it was proposed for — the alkA case needs the fold route, not a
   re-scoped baseline.
3. Some gains are only bursts relative to their trench. One araB event sits at 4.77 *local*
   MAD-scales but **−0.25 global** — below its gene's median.
4. **v6 recovers 9 of the 77 with a gene-scoped rule** (§7), which retrospectively removes those
   9 from the evidence for re-scoping at all: they were never a baseline problem, only a
   too-strict ratio. The genuinely local-only set is **68**, and that is the number this
   variant now has to justify itself on.

The mechanism that actually handles trench-to-trench variation is the fold route, because a ratio
is already immune to where a trench sits. It is kept as a viewer layer so the remaining 68 can be
judged by eye.

### The distribution view: a quartile rule reaches what no height bar does

The per-trench question can also be asked without any peak detection: histogram a kymograph's own
cell-frames and threshold on that distribution. `figures.kymo_threshold_table` /
`figures.plot_fluor_histogram_scoped`, and
`notebooks/analysis_notebooks/trackastra_figures_v2_09032026.ipynb`. Everything below is on **raw
`intensity_mean_fluor`**: exposure is one constant per experiment, so within a kymograph or a
(gene, experiment) the per-ms division is an exactly affine rescale — measured, **0 verdict
disagreements** across all 255 kymographs on every count column.

**A height bar re-scoped to the trench is not the answer, and neither is a percentile.** Re-scoping
the `median + k·s` bar reproduces this section's coverage numbers (43% → 59%) and nothing more: it
reaches *neither* documented near-miss, and at the cell-frame level it admits **17 hupA and 5 mazF**
cell-frames, so on height alone the hard controls do not score zero. A trench's own **p95** is worse
still — it flags ~5% of every trench by construction (measured 0.0488–0.0909, median 0.0513), so it
cannot rank trenches or pass the control at all. It is kept as the default *drawn* line because
"here is the top 5% of this trench" is what makes a histogram readable, not because it decides
anything.

**Tukey's fence does discriminate.** Per kymograph, `Q3 + 1.5·IQR` on raw fluorescence; the flagged
fraction depends on distribution shape rather than being pinned. Median flagged fraction per gene:
lacZ 0.0569, gldA 0.0364, gfcE 0.0345, alkA 0.0286, araB 0.0245, **chpS 0.0236, mazF 0.0200, hupA
0.0191**, murQ 0.0181, baeS 0.0118 — the three constitutive controls land **6th, 7th and 8th of
10**, with lacZ top and baeS bottom.

**And it reaches both near-misses this section says are out of reach of a re-scoped baseline:**

| trench | documented peak | gene bar | own MAD bar | Tukey fence |
|---|---|---|---|---|
| gfcE 013/153 | frame **31** | 0 of 300 | 0 of 300 | **2 frames: 31, 39** |
| alkA 001/1192 | frame **9** | 0 of 275 | 0 of 275 | **4: 9, 10, 11, 30** |

Not magic: for 013/153 the fence sits at 14.89 a.u. against a MAD bar of 16.63, i.e. `1.5·IQR`
above Q3 is 7.99 above the median where `4·s` is 9.73. It is simply a lower bar on a trench whose
bulk is tight — but lower in a **shape-aware** way that keeps the control ordering, which is what
distinguishes it from lowering `k`.

**The unit-free companion.** Per gene, the **max** `p99/p50` separates outright: all five bursty
genes reach ≥ 3.51, all three controls stop at ≤ 2.85 — band **(2.85, 3.51]**, directly analogous
to v5's fold band (3.10, 3.60]. It needs no denominator floor, because raw `intensity_mean_fluor`
is strictly positive here (min 0.346; every per-kymograph median ≥ 1.50) where `fold_left` had to
floor at 0.5·m.

**It agrees with v6 event by event.** Of v6's 156 events, **144 (92%)** have a fence-flagged
cell-frame in the same trench within ±5 frames — by route, absolute 97/99, fold 18/19, fold_left
29/38, the ratio route agreeing least, as expected for events with no height signal. Spearman rank
correlation between v6's per-100-cell rate and the Tukey median fraction is **0.839**. Two routes
with no shared machinery, which is the evidence standard §10 sets.

**Three limits, so this is not mistaken for a definition.**

1. **A distribution has no notion of a rise**, and prominence is the load-bearing criterion (§4).
2. **Flagged cell-frames are not events** — araB 003/1598 flags a contiguous run over frames 37–66,
   one burst plus its dilution decay.
3. **The fence flags something in 220 of 255 trenches**, so a per-trench binary is near-vacuous;
   only the fraction and the ranking carry information, and per trench the controls are not at zero.

A **v7 `tukey` route**, OR'd in as `fold` and `fold_left` were, is the natural follow-up — with the
same discipline: a sweep on the IQR multiplier (1.5 is the textbook value, not a measured one), the
hupA/mazF acceptance test at every setting, and a strict-superset check against v6. It is blocked
on one gap: **`burst_candidates` has no `baseline_cols` parameter**, so it cannot audit candidates
under anything but the (gene, experiment) baseline.

### The per-time-frame baseline (evaluated, not shipped)

The other re-scoping worth trying, and the one proposed before v6 was designed: take each
kymograph's **average fluorescence per time frame** and score every cell against the average of
its own frame and the neighbouring ones,

$$B(t) = \frac{1}{2w+1}\sum_{t'=t-w}^{t+w} L(t'), \qquad
L(t) = \operatorname{mean}_{c \text{ at } t} F_c(t), \qquad
\text{score} = \frac{F_c(t)}{B(t)}$$

It has every property one would want — unit-free, per-trench, and blind to slow drift.
**Measured, it cannot separate the target event from the constitutive controls:**

| denominator | the target | best control | margin |
|---|---|---|---|
| frame level, cell included | 1.561 | 2.415 (chpS) | 0.65 |
| frame level, the cell's whole **track** removed | 1.846 | 2.662 (chpS) | 0.69 |
| trench quiet level (20th pct of frame medians) | 2.895 | 3.564 (chpS) | 0.81 |

Under the first two it scores *below hupA's own maximum* (1.749 / 1.915). Sweeping the window
($w = 3, 5, 8, 12$) moves nothing, and smoothing the ratio along each track to demand temporal
coherence ($m = 3, 5, 7$) holds the margin at 0.67–0.69 — because chpS's high-contrast cells
are **also** temporally coherent. Local contrast is simply not the axis that separates a burst
from constitutive cell-to-cell scatter.

The structural reason is the denominator. The median trench-frame holds 5 cells, but **25.4%
hold ≤ 2** and 34.3% hold ≤ 3; the target trench holds exactly 2 during its event, so the frame
average is half the bursting cell — or, leaving it out, is one neighbour.

**One trap, recorded so nobody loses an afternoon.** A "leave-one-out" baseline built per
*frame* is not a change at all: with a mean aggregator the mean over cells of their
leave-one-out means is *identically* the frame mean,
$\frac{1}{n}\sum_c \frac{S - F_c}{n-1} = \frac{S}{n}$. Only removing the whole **track** is
a real leave-one-out.

---

## 9. The acceptance test

hupA, mazF and chpS are constitutive. They must score ~0, and they are also the *brightest*
genes, so this is not trivial to pass.

| definition | measures | hupA | mazF | mazF rank |
|---|---|---|---|---|
| v3 | $F$ | 0.00 | 0.00 | 10 of 10 |
| v4 | $F$ | 0.00 | 0.00 | 10 of 10 |
| v5 | $F$ | 0.00 | 0.00 | 10 of 10 |
| v5local | $F$ | 0.00 | 0.00 | 10 of 10 |
| v6 | $F$ | 0.00 | 0.00 | 10 of 10 |
| **v7** | $F$ | **1 event** | 0.00 | 10 of 10 |
| **v9** | `fiji_mean` | 0.00 | **9 events** | 9 of 10 |
| **v10** | `fiji_mean` | 0.00 | **13 events** | **7 of 10** |
| **v11** | `fiji_mean` | 0.00 | **9 events** | 9 of 10 |

**Through v6, no definition ever scored a hupA or mazF event** — and under v5-local mazF's
eligibility rises from 0% to 23% of its trenches, so it became *able* to fire and still did not,
which means the control was passing on merit rather than by structural exclusion.

**That clean record ends at v7, and the FIJI-values definitions break it further.** v7 scores one
hupA event on the argument that a constitutive gene can still burst (§7, and `tukey_iqr_k=4.4`
is the control-clean variant). v9, v10 and v11 all hold hupA at zero but score mazF, and **v10
moves chpS to 3rd of ten** — see §11. v11 pulls chpS back to 5th and mazF to 9th, but **v9 is
still the only FIJI-values definition under which all three constitutive genes rank last**. Read
this table as the record of a bar that has been consciously lowered twice, not as a claim that
every definition passes.

**And read every hupA 0.00 in the `fiji_mean` rows for what it is.** v9 and v11 hold hupA at zero
by *excluding all six of its trenches* on the lit gate, not by discriminating events inside them
— hupA is lit in 99.9% of its cell-frames, so the gate is a structural exclusion of exactly the
kind §8 credits v5-local for *not* needing. A zero earned that way is weaker evidence than
v3–v6's.

The two axes are also not comparable: v3–v7 measure $F$ on the raw image, v9–v11 measure the
transformed image of §1. A control failure on one says nothing about the other.

---

## 10. Reading the outputs

Events per gene:

| gene | v1 | v2 | v3 | v4 | v5 | v6 | v5local |
|---|---|---|---|---|---|---|---|
| lacZ | 29 | 33 | 33 | 35 | 36 | 38 | 48 |
| araB | 18 | 19 | 19 | 21 | 22 | 22 | 36 |
| gfcE | 10 | 14 | 14 | 15 | 20 | 28 | 40 |
| alkA | 7 | 9 | 9 | 9 | 13 | 23 | 18 |
| gldA | 3 | 7 | 7 | 7 | 15 | 24 | 21 |
| baeS | 3 | 4 | 4 | 7 | 7 | 14 | 10 |
| chpS \* | 2 | 2 | 2 | 4 | 4 | 5 | 6 |
| murQ | 1 | 1 | 1 | 1 | 1 | 2 | 2 |
| hupA \*, mazF \* | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **total** | **73** | **89** | **89** | **99** | **118** | **156** | **181** |

v6 and v5local are **not nested** — 113 shared, 43 v6-only, 68 v5local-only. They re-scope
different things: v5local moves the *baseline*, v6 changes the *base of the ratio*.

\* constitutive control.

Counts are not comparable between genes — sample sizes differ ~10×. The headline rate is

$$\text{bursts per cell cycle} = \frac{\text{burst episodes}}{\text{observed divisions}}$$

One division is one cell cycle watched start to finish, so this reads as **the chance of seeing a
burst per generation**. Under v5: lacZ 0.052, gldA 0.044, gfcE 0.029, alkA 0.028, araB 0.021,
chpS 0.011, murQ 0.009, baeS 0.006, hupA 0.000, mazF 0.000. In words, **a burst is seen roughly
once in 15 generations for lacZ and once in 180 for baeS.**

Under v6: **gldA 0.070, lacZ 0.055**, alkA 0.049, gfcE 0.040, araB 0.021, murQ 0.019, chpS
0.014, baeS 0.011, hupA 0.000, mazF 0.000.

Under v7: **gldA 0.073, lacZ 0.057**, alkA 0.049, gfcE 0.041, murQ 0.028, araB 0.024, chpS 0.014,
**hupA 0.011**, baeS 0.011, mazF 0.000. Only 8 events separate it from v6, and hupA is the one
that matters — see §11.

Counts are small, so all rates carry **exact Poisson (Garwood) 95% intervals**, not
$r \pm 1.96\sqrt{n}/N$ — the normal approximation gives $0 \pm 0$ at $n = 0$, a claim of certainty
from no data. hupA's 0 of 88 divisions admits a rate up to 0.042 per cell cycle, which *overlaps
gfcE's point estimate*. **Zero observed bursts is not evidence of a zero rate.**

---

## 11. What to be careful about

- **lacZ was top under every definition through v5; under v6 it is not.** gldA takes first place
  (0.070 vs lacZ's 0.055), because gldA is the gene most starved by an absolute bar (32% trench
  eligibility) and so gains most from a second ratio route. The intervals overlap (gldA
  0.045–0.104, lacZ 0.039–0.076), so this is **not** a resolved reordering — but do not repeat
  the old claim without saying which definition it was true of.
- **The three constitutive genes ranked last of ten under v9; under v10 they do not.** Per cell
  cycle v10 puts **chpS 3rd of ten** (0.0983) and **mazF 7th** (0.0544), with only hupA at a hard
  zero. Under v9 all three were the bottom three (chpS 0.0449, mazF 0.0377, hupA 0.0000). This is
  the clearest cost of preferring v10's frame-level agreement with v6 over v9's control
  cleanliness, and it is why v9 is kept runnable rather than deleted. **Do not quote a
  "controls rank last" claim without naming the definition.**
- **baeS/hupA/mazF were bottom under every definition through v6. Under v7 they are not.** v7's
  single hupA event (012/1165 frame 81 — bright at 7.73 against a 6.17 bar, but not prominent)
  takes hupA from a tied-last 0.0000 to 0.0114, 8th of 10 and a hair above baeS's 0.0112. The
  intervals overlap completely (hupA 0.0003–0.0633 on one event over 88 divisions), so this is not
  evidence that hupA bursts more than baeS — but do not repeat the old claim without naming the
  definition. It is the price of κ = 4.0 over the control-clean 4.4, and it is the only ranking
  claim v7 breaks.
- **The middle of the ranking is not resolved.** gldA doubles from v4 to v5 (0.020 → 0.044),
  entirely from the fold route, because its trenches are the least able to clear an absolute bar
  (32% eligible). Its confidence interval overlaps gfcE, alkA and araB. Whether that is a
  correction or an over-correction is open.
- **Per-gene rates are partly a statement about coverage.** With 57% of trenches unable to clear
  their bar, and unevenly so by gene, a low rate mixes "does not burst" with "could not have been
  seen to burst". Always read the eligibility column alongside.
- **The numerator counts episodes, the denominator counts divisions.** An inherited burst counts
  once, but both daughters' later divisions land in the denominator, so a gene with long bursts
  relative to its cell cycle scores lower at identical biology.
- **A burst here is a concentration excursion, not a synthesis event.** $F$ is a per-pixel mean,
  so a gene that raised total protein steadily in step with growth would register nothing.
- **No physical scale exists.** All lengths are camera pixels; nothing here can be quoted in µm.
- **Frame interval mixes 5 and 10 min**, and a burst must clear threshold within one interval.
  The per-cell-cycle rate absorbs most of this; per-100-cells does not.

---

## 12. Where the code is

| what | where |
|---|---|
| definitions, detector, audit tools | `mmtrack/bursts.py` |
| `BURST_DEF_V1…V11`, `BURST_DEFS`, `ROUTE_RANK` | same; `figures.BURST_DEF` forwards to them |
| the FIJI transform itself | `imagej_median`, `remove_speckle`, `to_kymograph` in `mmtrack/fluor_filters.py` |
| the FIJI per-cell extractor | `compute_fiji_fluor`, `fiji_display_floor` in `mmtrack/track_displacement.py` |
| v10's baseline, v9's gate | `attach_fiji_scale`, `attach_fiji_gate` in `mmtrack/bursts.py` |
| v11's baseline and detector | `attach_lit_baseline` / `attach_lit_baseline_for`, `detect_bursts_seeded` in `mmtrack/bursts.py` |
| the (t, y) linking both v9 and v11 use | `_link_ty_components`, same file — shared so they cannot drift |
| plots, loaders, per-gene figures | `mmtrack/figures.py` |
| the distribution view: `plot_fluor_histogram_scoped`, `kymo_threshold_table`, `select_scope`, `scoped_label`, `load_cells` | same |
| the narrative changelog | notebook cell 17 — **append only** |
| the audit and every measurement quoted here | notebook sections 8a–8k, 10, 11 |
| the distribution view and its worklist | `trackastra_figures_v2_09032026.ipynb` §3–4 |
| event tables | `data/bursts_090526.csv` (v3/v4/v5/v6/**v7**/v5local + positions + `route`) |
| the v6-era table | `data/bursts_090426.csv` (v3/v4/v5/v5local/v6) |
| the v5-era table | `data/bursts_090326.csv` (v3/v4/v5/v5local) |
| the frozen regression fixture | `data/bursts_082026.csv` — **do not regenerate** |
| the FIJI per-cell values | `data/fiji_fluor_nospeckle_090926.csv` (`fiji_mean`/`max`/`frac` + the per-trench floor) |
| v11 events | `data/bursts_v11_090926.csv` |
| the v11-vs-v10 adjudication worklist | `data/burst_worklist_v11_090926.csv` — every event either finds, tagged `both`/`v11_only`/`v10_only`, with each peak's within-trench lit percentile and the **absolute viewer frame** |
| v10 events | `data/bursts_v10_090926.csv` |
| v9 events | `data/bursts_v9_nospeckle_090926.csv` |
| the viewer's render-only union | `data/bursts_viewer_v11_090926.csv` — keys, `gene`, `centroid_y_rel` and the `burst_*` booleans only |

`BURST_DEF` currently points at **v3**, so the shipped tables are the 89-event set. v4, v5 and v6
are candidates pending review.

**The viewer draws two definitions, not twelve** — **v7** and **v12**, because more than three
outline layers cannot be read on one kymograph and these two are the live decision boundary: the
most precise rule (P 0.838) beside the classifier that beats every rule on recall (R 0.739).
Everything else is nested inside them or superseded — v5 is a strict subset of v7, v4 and v6 are
the versions v5 and v7 are supersets of, v1–v3 are the frozen fixture, v9/v10 are superseded by
v11 and v11 by v12, and v10 stays runnable from `data/bursts_v10_090926.csv`. All eleven rules
remain in `BURST_DEFS` (v12 is a classifier and is deliberately absent from it). Adding or
removing a layer is **six** coordinated edits in `generate_kymograph_viewer.py`, and the sixth
is the new ctrl-group's own closing tag: omitting it leaves `.controls` unclosed, which silently
disables both the sticky control bar and horizontal kymograph scrolling. `_assert_controls_balanced`
now fails the build on that.

**The dated viewer HTMLs are gone.** Only the most recent build is kept; `data/` and `*.html` are
both git-ignored, so the older ones were never recoverable and are now deleted. Older notebooks
still name specific files ("Visually QC'd against `data/kymograph_viewer_082026.html`") — read
those as a **historical record of what was done**, not an instruction to open a file. Any viewer
is regenerable from the burst CSVs above in a few minutes, with `/Volumes/mcovert` mounted.

**One trap that has cost real time.** A frame number read off the kymograph viewer is *not* a
`time_frame`. The viewer labels its axis with `start + column`; every table stores the 0-based
column. Use `mmtrack.figures.viewer_frame_to_time_frame()`.

---

## 13. Hand-labelled ground truth

Every count in this document — 73, 89, 99, 118, 156, 164, 352, 469 — is reported **without a
denominator**. §9's acceptance test catches gross failures, but three constitutive controls and
three regression trenches cannot express recall or precision. v11 made the gap concrete: 469
events against v10's 352, and the honest summary was "+33% events" with no way to say how many
are real, or how many real bursts no definition finds.

`mmtrack/burst_labels.py` closes it. **Nothing in it detects anything** — it is not a
definition, it takes no part in the v1–v11 sequence, and it must never be added to
`BURST_DEFS`.

### The table

Seeded from **v7 ∪ v11**, which already exists on disk: because v5 ⊂ v6 ⊂ v7 strictly,
`data/bursts_viewer_v11_090926.csv`'s 541 rows *are* that union (164 + 469 − 92 shared). The
template is `data/burst_labels_TEMPLATE_091026.csv`, 541 rows over 187 trenches, sorted by
gene → experiment → FOV → trench → frame. The sort is not cosmetic: locating a card in a 145 MB
viewer is the slow step, so grouping a trench's events turns 541 rows into ~190 visits.

The labeller fills two columns, `label` ∈ {`yes`, `no`, `unsure`} and `notes`. **Blank means
unreviewed, not "no"** — only labelled rows enter any statistic, which is what makes a 541-row
table tractable: stop whenever, and the measurement is smaller but still valid. To record a
burst the definitions missed, append a row with `experiment_name`, `FOV`, `trench_id`,
`track_id`, `viewer_frame` and a label, leaving `row_key` blank.

### `row_key` is the authority, and the reason is Excel

A spreadsheet round-trip turns FOV `'003'` into `3` and reformats a numeric-looking
`trench_id` — the same corruption `figures.READ_DTYPES` exists to prevent on the way in, after
which every join silently matches nothing. One text field, `exp|FOV|trench|track|time_frame`,
survives it. The split-out columns are for the labeller to read and are **re-derived from
`row_key` on ingest rather than trusted**; a de-padded FOV inside the key is repaired by
`zfill(3)`, not rejected.

### Four silent failures the ingest exists to catch

Each is a real mechanism in this codebase, not a hypothetical:

| failure | why it is silent |
|---|---|
| trench absent from the time dict | `figures.trench_start_frame` returns **0** rather than raising, so a typo'd FOV yields a plausible-looking wrong `time_frame` — and the viewer's `burst_by_trench.get(...)` then drops the row without counting or printing it |
| `viewer_frame` out of range | `figures.viewer_frame_to_time_frame` raises below `start` but **never consults `end`** |
| cell-frame does not exist | `build_burst_csv` dies later on `assert out[c].notna().all(), 'a peak row has no centroid_x'` **without naming the row** |
| burst picked on a degenerate mask | absent from `clean`, so it fails the same check — correct, but the message has to say why |

`read_labels` reports every offender by **CSV line number** and raises by default, because a
label table that silently drops a third of its rows is worse than no table.

**`end` is `None` for 86 of the 295 time-dict entries**, meaning "run to the end of the movie".
Coercing that to 0 gives the window `[start, 0]`, against which every real frame looks out of
range — it would have rejected every appended row in 29% of trenches. `trench_window` returns
`None` for the upper bound and callers must treat it as unbounded; the cell-frame existence
check is the real authority on which frames exist.

### What it can and cannot measure

**Precision is measurable.** Every seeded row is an event some definition claimed, so the
fraction of a definition's *labelled* events marked `yes` is exactly its precision. Unlabelled
rows are unreviewed, never negative.

**Recall over all rows is an upper bound, and must be quoted as one.** The table was seeded
with these definitions' own output, so a burst they all miss is present only if the labeller
volunteered it. `sweep_trench` marks a gene-stratified sample of **15 trenches** (43 seeded
rows) for *exhaustive* review; recall over those is unbiased, and it is the number to quote.
`evaluate` returns both scopes as separate rows and never averages them.

Recall is also reported twice — at the exact frame and within ±5 frames — and the two are
never merged, for the reason §7 gives for v9: 96% trench-level coverage against 16/156 exact
rows, where quoting either as "agreement" was wrong. Proportions carry **Wilson** intervals,
not `p ± 1.96·√(p(1−p)/n)`, which returns `0 ± 0` at zero events — the same objection §9
raises against the normal approximation for burst rates.

### The viewer change this required

The labeller's identifier is the `track_id`, and it **was not readable**. Through 090926
`render_labels_svg` mirrored napari's `track_labels` layer: one number per *lineage, at its
first appearance only*, so learning the id of a cell bursting at frame 80 meant tracing its
polyline back through division connectors to wherever that lineage started. It now draws one
label per **cell-frame**. The `napari_tracks_*.npy` array is `[track_id, t, y, x]` with exactly
one row per cell-frame — verified, 300 rows against 300 cell-frames on
`DUMM_giTG060_064_121425` 013/153 — and its ids are the ctc mask labels, so a number read off
the screen joins directly to the cell tables. Verified end to end: on that trench all four v11
events show a label equal to their `track_id`, and the mask label under the printed number
agrees.

Font dropped 10 → 8 px and the anchor is now centred, because frames are 20 px wide at
`--downscale 1.0` and a 3-digit id at 10 px overruns into the neighbouring time point.

### Where the code is

| what | where |
|---|---|
| template writer, ingest, scoring | `mmtrack/burst_labels.py` |
| the template to edit | `data/burst_labels_TEMPLATE_091026.csv` |
| the validated ingest | `data/burst_labels_091026.csv` — **NEVER WRITTEN, see below** |
| the per-cell-frame id layer | `render_labels_svg` in `generate_kymograph_viewer.py` |

**⚠ THIS ROUTE WAS BUILT AND NEVER USED. Read this before trusting anything above.**
`data/burst_labels_TEMPLATE_091026.csv` exists, with all 541 seeded rows — and **every one of
its `label` cells is blank**. `data/burst_labels_091026.csv` was never written and is not on
disk. So `write_label_template` / `read_labels` / `evaluate` have never been run against real
verdicts, and no precision or recall figure anywhere in this document comes from them.

The ground truth that *does* exist arrived by a simpler route and is the subject of §14:
`data/bursts_lacZ_gfcE_091026.csv`, bursts read straight off the viewer into a spreadsheet of
`track_id, frame, experiment_name, FOV, trench_id, gene, start, end`. Every v12 number is
matched against that file, via `burst_training.load_hand_labels` / `match_events` / `pr_curve`,
not via this module.

`burst_labels.py` is therefore working but unexercised machinery. It is the better-designed of
the two routes — `row_key` survives Excel, blank means unreviewed, `sweep_trench` gives an
unbiased recall scope, proportions carry Wilson intervals — so it is kept rather than deleted.
But the first person to use it should expect to be the first person to use it.

---

## 14. The trained classifier (v12)

Every definition up to v11 was a hand-built rule judged on three controls and three regression
trenches. §13 built the machinery for ground truth; `data/bursts_lacZ_gfcE_091026.csv` is the
first of it — **211 bursts identified by eye** in lacZ and gfcE, over 60 trenches reviewed
**exhaustively**, so an unmarked cell-frame in one of them is a real negative.

That changes what can be said. For the first time there are precision and recall numbers, and
they are not flattering to the rules.

### What the labels are

Measured, not assumed:

- **`frame` is the absolute viewer frame.** 203 of 211 rows resolve under
  `time_frame = frame − start`; only 110 resolve reading it directly.
- **The labelled frame is the peak** — median offset 0 from its own track's brightest
  `fiji_mean` frame, 77% within ±1, 91% within ±3, symmetric. Same convention as v11.
- **60 reviewed trenches, 23,388 cell-frames, 203 positives = 0.87%.**
- Two spreadsheet defects repaired on ingest: 4 rows carried Excel fill-handle increments
  (`…061126`–`…061129` → `…061125`, all FOV 008 trench 1488) and `FOV` was de-padded
  throughout. 8 rows do not resolve and are held out in
  `data/burst_labels_recheck_091026.csv` for re-checking rather than snapped to a neighbour —
  snapping would invent ground truth.

### How the rules actually score

Greedy **one-to-one** matching, ±1 frame, restricted to the reviewed trenches. One-to-one
matters: counting "any label within ±k" lets one event claim several labels and inflates both
numbers, which is how the first pass at this table was done before being corrected.

| definition | events | TP | precision | recall |
|---|---|---|---|---|
| v7 | 68 | 57 | **0.838** | 0.281 |
| v10 | 82 | 61 | 0.744 | 0.300 |
| v11 | 147 | 87 | 0.592 | 0.429 |
| v9 | 279 | 49 | 0.176 | 0.241 |
| **classifier @0.90** | 191 | 125 | 0.654 | 0.616 |
| **classifier @0.75** (shipped) | 247 | 150 | 0.607 | **0.739** |
| classifier @0.60 | 283 | 159 | 0.562 | 0.783 |

**The classifier strictly dominates v11** — higher precision *and* 1.7× the recall. It does
**not** reach v7's precision; v7 remains the most precise rule in the collection, at a quarter
of the recall. All classifier numbers are **out-of-fold**: no trench was ever scored by a model
that had seen it.

### The model

A `HistGradientBoostingClassifier` (sklearn 1.7.2, already pinned — xgboost and lightgbm are
not installed and this is the same algorithm family) over **43 per-cell-frame features**, all
within-trench normalised because the FIJI clip floor is per-trench and raw fluorescence spans
283× across genes.

Cell-frame out-of-fold: **AUC 0.968, PR-AUC 0.342** against a 0.87% base rate. Shuffled labels
give AUC 0.472 / PR-AUC 0.010, i.e. the base rate — the sanity floor holds.

**Cross-validation is grouped by trench** (`StratifiedGroupKFold`), asserted per fold. This is
not optional: cell-frames in a trench share a clip floor, a baseline and usually a lineage, so
a random split trains and tests on the same burst. The GNN's existing split is time-based and
puts every trench in both halves; it was not copied.

#### The temporal block is what earns the gain

Nothing in this codebase computed a rolling window on fluorescence before — `bdef['window']`
was only a `find_peaks` half-width. Adding lineage-aware temporal features is the single
biggest contributor:

| feature set | n | AUC | PR-AUC | event max F1 |
|---|---|---|---|---|
| within-trench percentile alone | 1 | 0.932 | 0.329 | 0.598 |
| + the rest of the normalisation block | 11 | 0.964 | 0.289 | 0.600 |
| + raw / shape / trench context | 33 | 0.963 | 0.296 | 0.635 |
| **+ temporal (shipped)** | **43** | **0.968** | **0.342** | **0.667** |

Read the event max F1 column, not AUC: the normalisation block saturates AUC almost
immediately, and every subsequent gain shows up in the number that actually matters.

The top feature by permutation importance is `intensity_mean_fluor__fold_back5` — the causal
fold against the trailing median along the lineage, i.e. the per-cell-frame analogue of v6's
`fold_left`, which until now existed only at peak level inside `burst_candidates`.

"Along the lineage" is load-bearing: a track ends at every division, so a plain groupby-shift
loses the signal exactly where a burst is inherited by two daughters.

#### Three constants, each with its band

**`mask_window = 8`.** The labeller's own note is that neighbouring frames still show signal,
because the decay is dilution by growth rather than protein loss. Cell-frames within ±W frames
of a positive — on its track *or any ancestor or descendant* — are excluded from the loss
entirely: neither positive nor negative. Precision at fixed recall 0.65: W=0 → 0.494,
W=3 → 0.544, W=5 → 0.559, **W=8 → 0.592**, W=12 → 0.572. Bracketed on both sides.

**`link_dt = 3`** for collapsing scored cell-frames into events. At W=8: dt=2 → max F1 0.614,
**dt=3 → 0.634**, dt=5 → 0.624.

**The event linker is NOT v9/v11's**, and that is a correction, not a preference. Their
`_link_ty_components` tolerates `1.5 · cell_length · Δt` — a *product*, so at Δt=3 it reaches
4.5 cell lengths, most of a trench. A rule emits sparse peaks and survives that; a classifier
lights up whole neighbourhoods and does not. Measured on `DUMM_giTG059_060_061125` 008/1488,
it linked two cells **109.6 px apart** — three cell lengths, opposite ends of the trench —
because the tolerance at Δt=2 was 110.2 px, a margin of **0.6 px**, and it absorbed two
hand-labelled bursts into their neighbours' events. That trench reported 3 of its 5 labels
instead of 5.

The product is also the wrong physics: cells advect ~4.7 px/frame (median, 17 at p10), so the
displacement to tolerate is a drift term plus fixed slack, not a multiplication.
`_link_ty_capped` uses `1.5 · cell_length + 10 px · Δt`. On the shipped model it is better on
every axis — **5/5 labels in that trench and event F1 0.731 against 0.716** — and out of fold
the two are tied within noise (0.661 vs 0.667). `bursts._link_ty_components` is left untouched
so v9 and v11 stay bit-reproducible.

**A trap this exposed, worth keeping.** The shipped event table is produced by the FINAL model,
which is *in-sample* on the 60 reviewed trenches and therefore more confident there. Higher
scores mean more cell-frames clear the threshold, which means larger components, which means
*more* merging — so an over-permissive linker damages the shipped artifact more than it damages
the out-of-fold metrics. The metrics looked fine (5/5 in that trench out of fold) while the
delivered file was losing events.

**`FOLD_FLOOR_FRAC = 0.01`.** A fold denominator is floored at 1% of the trench's own max, not
at an epsilon. This is v6's `fold_base_floor_frac` lesson and it is not optional — unfloored,
a near-zero trailing median manufactures arbitrary fold, and a depth-3 tree fitted on that
version split at `fold_back3 <= 3179`, which is noise. Swept, and bracketed on both sides:

| floor | GBM max F1 | rule max F1 | largest fold seen |
|---|---|---|---|
| 0.000 | 0.640 | 0.543 | 1,000,000,000 |
| 0.005 | 0.657 | 0.594 | 200 |
| **0.010** | **0.667** | 0.592 | 100 |
| 0.020 | 0.658 | 0.590 | 50 |
| 0.050 | 0.635 | 0.589 | 20 |

#### Relative features alone cannot see an empty trench

Every normalised feature is *relative*, and that is a trap in its own right: in a trench with
no burst at all, the brightest cell still scores percentile 1.0, so a purely normalised model
must invent an event in every trench. Measured on the certified burst-free trench
`DUMM_giTG064_Glucose_022625` 015/298 — without absolute-magnitude features the classifier
fired once there at threshold 0.92 (top cell-frame score 0.939); with them, zero, and the top
score fell to 0.853. Max F1 rose 0.634 → 0.665 at the same time. Absolute magnitude is the only
thing that can say *nothing here is bright enough to be anything*.

### The simple rule, for the record

A depth-3 tree on six features reaches **max F1 0.589 (P 0.638, R 0.547)** against the full
model's 0.667. It splits on `fiji_mean__fold_back3`, `fiji_mean__lit_pct`,
`fiji_mean__rise3`, `fiji_mean__over_tmax` and `trench_fiji_max` — a rise term, a
within-trench rank, and an absolute floor, which is a fair summary of what the full model is
doing. Quote it when a rule is needed and the classifier is not available; it still beats v11
on recall.

### Two things it does not do, stated plainly

**It fires in the certified burst-free trench at the shipped threshold.** 0 events at 0.90,
**2 at 0.75**, 3 at 0.60. v7 and v11 are clean there; v10 fires once and v9 five times. This
is a known failure of the shipped operating point, not a rounding error.

**It is much worse than the rules on the constitutive controls**: hupA 25, mazF 57, chpS 94
events, against v11's 0 / 9 / 34. This is expected and was predicted — the model has **never
seen a control gene**, at the labeller's direction, so it has no example of "uniformly bright
is not a burst". These counts are reported, not gated. **The single highest-return next step
is labelling a handful of hupA/mazF/chpS trenches**; nothing else in the pipeline will fix
this, because the information simply is not in the training set.

Transfer between the two labelled genes is nevertheless good, and better than the pooled
cross-validation: training on gfcE and testing on lacZ gives AUC 0.985 / max F1 0.694, and the
reverse gives AUC 0.956 / max F1 0.655. That is the only transfer evidence that exists, and it
is a two-fold test on two genes — do not read it as proof the model generalises to the five
unlabelled non-control genes.

### Does v12 have a physical definition?

**Not a closed-form one, and it should not be claimed to.** v1–v11 each reduce to a sentence
with a threshold in it. v12 is 300 boosted trees over 43 features, none of which carries a cut;
two cells with identical brightness can be judged differently because something else differs.
That is the price of the recall.

**But the decision is effectively three-dimensional, which is a physical statement.** Ranking
features by local occlusion — replacing one feature with its dataset median and measuring the
drop in predicted probability — the top driver across all 1,091 events is one of only three
quantities:

| share of events | quantity |
|---|---|
| **48.9%** | percentile within its own trench (FIJI-transformed) |
| **41.1%** | rank among the cells present in the same frame (1.0 = brightest) |
| **10.0%** | fold above its own lineage's trailing 5-frame median |

So the honest one-line gloss is: **a burst is a cell that is the brightest in its frame, in the
top few percent of its own trench, and raised above its own lineage's recent past.** All three
axes are within-trench, which is what makes them comparable across a dataset whose brightness
spans 283×. That gloss is also why the distilled depth-3 rule gets within 0.08 F1 of the full
model — it is fitting the same three axes.

What the gloss is *not*: a threshold. The occlusion ranking says what mattered for a given
call, not where the boundary lies.

**Per event, use `explain_events`.** `data/bursts_v12_explained_091026.csv` carries one row per
event with the physical quantities in readable units — `raw_fluor_au`, `fiji_mean`,
`fold_over_own_past_5f`, `pct_within_trench`, `rank_within_frame`, `n_frames` — plus a `why`
sentence. For `DUMM_giTG059_060_061125` 008/1488 track 11 at frame 20:

> raw 9.29 a.u., 2.58× its own lineage's trailing 5-frame median, 95th percentile of its
> trench, brightest cell in its frame, elevated for 3 frames.

Attribution is local occlusion, not a Shapley value (`shap` is not installed, and adding it
would mean editing a fully pinned `pyproject.toml`). Drops do not sum to the score and can be
negative — read them as a ranking.

**For a methods section**, three defensible options, in descending order of how much recall they
keep: quote v12 with its training set as the definition ("an event an expert marked on the
kymograph viewer; reproduced at precision 0.607 / recall 0.739 on held-out trenches, grouped by
trench"); quote the distilled depth-3 rule and cite v12 as the screen; or quote v7/v11, which
are closed-form, and accept the recall loss. The one thing not to do is describe v12 as though
it had a threshold.

### Where the code is

| what | where |
|---|---|
| label ingest, features, model, event collapsing | `mmtrack/burst_training.py` |
| per-event explanation | `explain_events`, `FEATURE_MEANING`, same file |
| the explained event table | `data/bursts_v12_explained_091026.csv` |
| the hand labels | `data/bursts_lacZ_gfcE_091026.csv` |
| the 8 unresolved labels, with diagnoses | `data/burst_labels_recheck_091026.csv` |
| the fitted model + its feature list | `models/burst_classifier_091026.joblib` |
| per-cell-frame scores | `data/burst_scores_091026.csv` |
| events at the shipped threshold | `data/bursts_v12_091026.csv` (1,091 events, 240 trenches) |

### Reproducing the shipped v12 scores

Measured, and worth writing down because it is not obvious: `data/burst_scores_091026.csv` is
reproducible to **1.1e-16 over all 81,799 cell-frames**, but only from the exact input it was
made from. The recipe is

```python
cells = figures.load_cells()                                   # drop_failed=True
fiji  = pd.read_csv('data/fiji_fluor_nospeckle_090926.csv', dtype=figures.READ_DTYPES)
j     = cells.merge(fiji, on=CELL_KEY, how='inner')             # 81,920 rows
j     = j[~j['is_degenerate_mask']]                             # 81,799 -- DO NOT SKIP
d     = burst_training.featurize(j)                             # 43 features
```

**The `is_degenerate_mask` drop is load-bearing and easy to miss.** It removes only 121 rows
(0.15%), but they are spread over **67 trenches**, and every feature in block A is a
*within-trench* rank or maximum — so those 121 rows perturb the features of their trench-mates
and 1,460 cell-frames come out with different scores, up to 0.75 different. A 0.15% input
difference producing a 1.8% output difference is the whole reason the filter is not optional.
`bursts.drop_degenerate_masks(j, MIN_CELL_AREA_FRAC, exp_median_area)` is the equivalent call
and returns the identical row set.

**The same trap sinks the v1/v2/v3 regression**, and it is worth stating as a general rule
because it has now cost two separate measurements. Run `run_burst_detection_any` on the raw
`load_cells()` table and you get **79 / 94 / 94** — a strict *superset* of the fixture, nothing
missing, 6/5/5 spurious. Drop the degenerate masks first (the notebook calls that view `clean`)
and it is exactly **73 / 89 / 89, set-identical**. v5 118; v11 469.

> **RULE: every burst measurement in this document starts from `cells[~cells['is_degenerate_mask']]`.**
> A 1-px mask is not a cell, and because every baseline here is a *within-trench* statistic, one
> bad mask contaminates its whole trench rather than just itself. The filter is §6's "step zero"
> and it is not optional. If a number comes out too HIGH and nothing is missing, this is the
> first thing to check.
