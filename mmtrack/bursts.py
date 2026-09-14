"""Burst detection: the versioned definitions, the detector, and the audit tools.

This code used to live only in cells 55/56/75 of
``notebooks/analysis_notebooks/trackastra_figures_082026.ipynb``, which is closed. That made
the shipped ``data/bursts_082026.csv`` unreproducible from the current summary notebook and
made every definition change a copy-paste. It is hoisted here unchanged in behaviour: the
regression test is that :func:`run_burst_detection_any` returns exactly **73 / 89 / 89**
events for v1 / v2 / v3, set-identical to the ``burst_v1/v2/v3`` columns of that CSV.

**The primitives are imported, not re-copied.** ``compute_baselines``,
``trench_max_envelope`` and ``lineage_family_roots`` already existed in :mod:`mmtrack.figures`,
byte-identical to the notebook's copies. They are imported from there so there is one
implementation rather than three. (``build_lineage_traces`` lives there too, but detection runs
on the max envelope, so only the plots use it.)

Import direction is one-way: this module imports :mod:`mmtrack.figures`, never the reverse.
``figures`` reaches the definition dicts through a module-level ``__getattr__`` so
``figures.BURST_DEF`` keeps working without an import cycle.

What a burst is
---------------
A frame on a max-fluorescence envelope is a burst when it (1) clears
``median + k * mad_scale`` for its (gene, experiment), (2) is the maximum within +/-``window``
frames, and (3) has topographic **prominence** >= ``k_prom * mad_scale``. Prominence is the
load-bearing criterion: a flat-high constitutive trace clears any height bar but has
prominence ~0, which is what keeps hupA and mazF at zero. Do not replace it with a one-sided
or ``min``-based rise -- measured, both score hupA's brightest wobble at 6.855 MAD-scales,
*above* a real chpS burst at 6.061, so both invert the control. **That applies to the
prominence in MAD units and only to it**; the unit-free ratio does not invert, which is what
v6 relies on.

From **v5** there is a second, OR'd route in: a peak whose ``peak / max(left_base,
right_base)`` reaches ``fold_prom`` qualifies even when it never clears the height bar. That
exists because the height bar is absolute-within-gene, so a trench sitting below its gene's
median can never register a burst however large the excursion inside it -- 146 of 255 trenches
are in that position. The ratio is unit-free, and a flat-high constitutive trace scores ~1.0,
which is why it does not cost the control. Every event carries ``route``.

From **v6** there is a third: the same ratio taken against the ``left_base`` alone, with the
denominator floored at ``fold_base_floor_frac * median``. ``max(left, right)`` binds on the
right whenever a burst's decay has not returned to its pre-burst floor -- the normal case,
since the decay is dilution by growth rather than protein loss -- so the max-base rule
systematically penalises long-decay events. v6 is a strict superset of v5: 99 absolute + 19
fold + 38 fold-left = **156**, with hupA and mazF still at zero.

Then an **episode merge** collapses a slow decay to one event, and (v2 onward) a
**concurrent merge** collapses one burst seen in two adjacent lineage families.

Units
-----
``fluor_col`` defaults to ``fluor_per_ms`` (a.u./ms). Detection is invariant under the
exposure rescale: baselines are per (gene, experiment) and exposure is constant within an
experiment, so median, MAD, height and prominence all scale by the same 1/E.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from mmtrack.figures import (
    FLUOR_COL, KYMO_KEY,
    compute_baselines, lineage_family_roots, trench_max_envelope,
)

__all__ = [
    'BURST_DEF', 'BURST_DEF_V1', 'BURST_DEF_V2', 'BURST_DEF_V3', 'BURST_DEF_V4',
    'BURST_DEF_V5', 'BURST_DEF_V6',
    'BURST_DEF_V7', 'BURST_DEFS', 'ROUTE_RANK', 'FAMILY_SEP', 'MIN_CELL_AREA_FRAC',
    'attach_kymograph_fence',
    'detect_bursts', 'add_lineage_family', 'add_spatial_subgroup',
    'merge_concurrent_bursts', 'run_burst_detection_any', 'drop_degenerate_masks',
    'build_burst_csv', 'burst_candidates', 'detect_sustained_rises',
    'add_relative_y', 'attach_kymograph_baselines', 'burst_coverage', 'prom_multiplier',
    'BURST_DEF_V11', 'attach_lit_baseline', 'detect_bursts_seeded',
]

# ---------------------------------------------------------------------------------------
# Versioned definitions -- append only
#
# Add a new dict and repoint BURST_DEF; never edit an old one in place. Every output frame
# carries `def_version` so a saved result traces back to the rule that produced it. The
# narrative changelog lives in the summary notebook.
# ---------------------------------------------------------------------------------------

FAMILY_SEP = '#'          # encodes the partition into trench_id so detect_bursts groups by it

BURST_DEF_V1 = {
    'version': 'v1',
    'k': 4.0,                       # MAD multiplier for the HEIGHT threshold
    'k_prom': None,                 # required prominence multiplier; None -> fall back to k
    'window': 5,                    # +/- frames for the local-max check / peak separation
    'min_prominence': None,         # absolute override; None -> k_prom * mad_scale
    'fluor_col': FLUOR_COL,
    'signal': 'per-trench max envelope',
    'baseline_keys': ('gene', 'experiment_name'),
    'constitutive_genes': ('mazF', 'hupA', 'chpS'),
}

# v2 detects on per-LINEAGE-FAMILY envelopes so two bursts running at once in one trench are
# not masked by whichever is brighter, then merges concurrent adjacent events back together.
BURST_DEF_V2 = {
    **BURST_DEF_V1,
    'version': 'v2',
    'signal': 'per-lineage-family max envelope',
    'merge_window': 5,                 # frames; peaks this close together are concurrent
    'merge_gap_cell_lengths': 1.5,     # centroid gap below this = one burst, not two
    'merge_keep': 'earliest',          # report burst onset, not the brighter neighbour
}

# v3 subdivides a lineage family that is spatially discontiguous. Measured: a no-op on this
# dataset (identical 89 events to v2), because identity-based linking re-joins whatever a
# position split separates -- cells in a trench are physically contiguous.
BURST_DEF_V3 = {
    **BURST_DEF_V2,
    'version': 'v3',
    'signal': 'per-lineage-family x spatial-cluster max envelope',
    'split_gap_cell_lengths': 1.5,   # cut a family where its cells are this far apart in y
}

# v4 decouples the two uses of k. In v1-v3 one k set the height bar AND the required
# prominence, and they had never been tested apart. Measured on the 20-experiment frame:
# lowering only the prominence bar has a plateau at k_prom in [3.6, 3.8] that catches genuine
# near-misses while leaving hupA and mazF at exactly 0.00 (hupA's best candidate is 3.586, so
# 3.5 is where the control breaks). 3.7 is the middle of that plateau. 89 -> 99 events.
BURST_DEF_V4 = {
    **BURST_DEF_V3,
    'version': 'v4',
    'k': 4.0,            # height bar only, from here on
    'k_prom': 3.7,
}

# v5 adds a second ROUTE rather than moving a bar. v1-v4 gate on absolute brightness -- a peak
# must clear median + k*s -- so a trench whose whole envelope sits below its gene's median can
# never register a burst however dramatic the excursion inside it is. Measured, 146 of 255
# trenches are in that position, and 36 of them contain a >= 2.5x internal fold rise: gldA 42%
# of its trenches, gfcE 27%, alkA 17%. `fold_prom` catches those on a unit-free ratio.
#
# 3.3 is the midpoint of the window that admits the motivating case (3.60x) while excluding
# every constitutive gene's best candidate (hupA 1.76x, mazF 2.94x, chpS 3.10x). That ~7%
# guard band on each side is the honest limit of the criterion -- quote it with the number.
BURST_DEF_V5 = {
    **BURST_DEF_V4,
    'version': 'v5',
    'fold_prom': 3.3,
    'fold_height_floor_s': 1.0,
}

# v6 changes which BASE the fold ratio divides by, and adds that as a third route rather
# than altering v5's. `fold` uses max(left_base, right_base); that max binds on the RIGHT
# whenever a burst's decay does not return to its pre-burst floor -- which is the normal case
# here, because the decay is dilution by growth, not protein loss (see the module docstring
# and bursts.md section 1). So max-base systematically penalises exactly the long-decay
# events. Anchoring on the left base asks "did it rise from where it was?", which is the
# question a burst poses, and a flat-high trace still scores ~1.0, so the constitutive
# control survives.
#
# This is NOT the left-base variant that bursts.md section 4 rejects. That measurement was on
# ABSOLUTE prominence in MAD units, where left-only inverted the control (hupA's best wobble
# 6.855 vs a real chpS burst 6.061). As a unit-free RATIO it does not invert: hupA tops out at
# 3.372 and the motivating case reaches 3.787.
#
# `fold_base_floor_frac` is load-bearing. Unfloored, a near-zero left base manufactures an
# arbitrary ratio -- chpS 010/46 has left_base 0.006 and scores 30.0x. Flooring the
# denominator at 0.5 * the gene-experiment median takes the controls to hupA/mazF 3.372 and
# chpS 4.22 while leaving the motivating case untouched at 3.787.
#
# 3.6 sits mid-plateau: hupA and mazF stay at zero candidates for every bar >= 3.4, and the
# motivating case is lost above 3.79. Band [3.4, 3.79], ~5% each side -- narrower than v5's,
# and its upper edge is set by a SINGLE event. Quote the band with the number.
BURST_DEF_V6 = {
    **BURST_DEF_V5,
    'version': 'v6',
    'fold_left_prom': 3.6,
    'fold_base_floor_frac': 0.5,
}

# v7 adds a fourth route on a completely different statistic: the trench's own Tukey fence,
# `Q3 + iqr_k*IQR` over its cell-frame distribution. Every route up to here is a location+scale
# statement about the (gene, experiment) baseline -- height, or a ratio against a peak's own
# flanks. A quartile fence is a SHAPE statement about one trench, and measured on cell-frames it
# reaches both documented near-misses that no height bar reaches at any scope (gfcE 013/153
# frame 31, alkA 001/1192 frame 9) while ranking hupA/mazF 8th/7th of 10 by flagged fraction.
#
# THE MULTIPLIER IS 4.0, NOT THE TEXTBOOK 1.5. Measured sweep (events / added over v6's 156 /
# control-gene events), all strict supersets of v6 with zero losses:
#
#   iqr_k  events  added   hupA  mazF  chpS   regression trenches
#     1.5     523   +367      5    16    43   BROKEN
#     2.5     247    +91      2     3     9   BROKEN
#     3.0     196    +40      2     2     7   BROKEN
#     3.4     175    +19      2     1     6   ok
#     4.0     164     +8      1     0     5   ok      <- shipped
#     4.4     160     +4      0     0     5   ok      <- control-clean
#     4.8     159     +3      0     0     5   ok
#     5.0     156     +0      0     0     5   ok      (adds nothing = v6)
#
# Read that table honestly, because it says something the earlier routes did not have to admit:
# **there is no multiplier at which this route both adds events and leaves the constitutive
# control untouched by more than a hair.** At 1.5 it is not close -- mazF alone takes 16 events
# and all three regression trenches break. The control-clean band is [4.4, 4.8], and it buys
# only 4 events.
#
# 4.0 is shipped as a deliberate call, not a plateau midpoint: it doubles the yield to +8 at a
# cost of exactly ONE hupA event (hupA 012/1165 frame 81), on the argument that a constitutive
# gene can still burst and a hard zero is a convention rather than a law. Set
# `tukey_iqr_k=4.4` for the control-clean variant. Note what that one event is, because it is
# the textbook failure mode: fluor 7.73 against a gene bar of 6.17, so it is BRIGHT but not
# prominent -- Denver, in bursts.md section 4's terms. The fence cannot see a rise, and this is
# what that costs.
#
# At 4.0: gfcE 013/153 keeps route='fold_left' and alkA 001/1192 keeps route='fold', so v7
# relabels nothing v6 found. chpS and mazF are unchanged from v6 entirely.
BURST_DEF_V7 = {
    **BURST_DEF_V6,
    'version': 'v7',
    'tukey_iqr_k': 4.0,
}

# v8 -- NOT SHIPPED. A tested and REJECTED definition, kept so the negative result is not
# re-derived. It is deliberately absent from the viewer's BURST_VERSIONS and from
# figures.VERSION_COLOR, and no event CSV was written for it.
#
# MEASURED FAILURE: swept fiji_fold {2,3,4,6,10} x fiji_base_floor_frac {0.05..0.30} x
# fiji_max_activity {0.02,0.05,0.10,0.15} over both fiji_mean and fiji_max -- 34 settings --
# and NONE is both constitutive-clean and able to find the three regression trenches. The
# tradeoff is structural, not a tuning problem:
#   * every setting that finds 3/3 regression trenches admits 9-17 mazF events;
#   * every setting that holds mazF at 0 finds at most 1 of the 3;
#   * v6 recall peaks at 54.5%, so v8 is NOT a superset of v6 -- it misses half of v6's 156
#     events while adding hundreds of new ones. Each version being a strict superset of the
#     last is what makes the definitions comparable at all, and this breaks it.
#
# WHY, diagnosed: on the three regression trenches the fiji envelope maxima are 2.06, 0.73
# and 6.48 counts against display spans of 29, 22 and 25. `fiji_mean` averages over the WHOLE
# cell, so a burst lighting part of a cell is diluted to a couple of counts, and any
# denominator floor expressed as a fraction of the display span (a per-PIXEL range) is the
# same size as the peak itself -- dimensionally inconsistent, and the ratio can never reach
# the bar. `fiji_max` avoids the dilution and does reach 3/3, but only at floors loose enough
# to let mazF through.
#
# WHAT IS STILL GOOD, and why the values were kept: within a trench the signal ranks bursts
# almost perfectly -- 21/21 sampled v6 events are nonzero at a median percentile rank of
# 0.993. The measurement is informative; it is the absolute-threshold rule built on it that
# fails. A rank- or envelope-relative rule (peak as a fraction of the trench's OWN fiji max,
# rather than of the display span) is the untried direction.
#
# The per-cell values live in data/fiji_fluor_082026.csv via
# track_displacement.compute_fiji_fluor, verified bit-identical to the viewer's layer.

# v8 -- detection on the FIJI-TRANSFORMED images, a different measurement rather than a
# different rule on the same one. `fluor_col` is fiji_mean from
# track_displacement.compute_fiji_fluor: per-cell mean of (clip at 0.25*trench max ->
# ImageJ median r2 on the assembled kymograph -> small-object removal).
#
# It does NOT reuse the v1-v7 routes, and cannot. Those all lean on the (gene, experiment)
# median + MAD, and this column is ~99% zeros, so that baseline collapses: measured, three
# of four gene-experiment groups give median 0 and mad_scale 1e-6 (the clip floor), i.e.
# height ~ 4e-6, where every nonzero blip clears the bar. The fold routes fail the other
# way (`base > 0` and `den_floor = 0.5*med` both go to zero) and a Tukey fence degenerates
# identically (Q3 = IQR = 0). So v8 runs its own detector, `detect_bursts_fiji`, and
# detect_bursts is untouched -- which is also what keeps the frozen v1/v2/v3 event sets safe.
#
# `k`/`k_prom`/`min_prominence` are kept only so the dict stays shape-compatible with the
# hard-indexed keys in run_burst_detection_any; the fiji detector ignores them.
BURST_DEF_V8 = {
    **BURST_DEF_V7,
    'version': 'v8',
    'fluor_col': 'fiji_mean',
    'tukey_iqr_k': None,           # no fence: Q3 and IQR of a mostly-zero column are 0
    'fiji_floor': 0.0,             # a candidate must clear its trench's own display floor
    'fiji_fold': 3.0,              # peak / floored left base: unit-free, replaces MAD prominence
    'fiji_base_floor_frac': 0.05,  # floor the denominator at this fraction of the display
                                   # span; an eps-only floor makes the ratio inert
    'fiji_max_activity': 0.05,     # reject a trench whose MEDIAN cell already sits this high
                                   # above its floor (as a fraction of the display span) --
                                   # that is a constitutive trace, not a burst. Swept.
}

# v9 -- SUPERSEDED BY v10, kept for provenance and still runnable. Any-signal detection with
# (time, y) deduplication; v10 replaced it because a sharp-RISE rule on the same values agrees
# with v6 at the frame level far better (91 of 156 exact rows vs 16), v9 reporting burst onset
# where v6 reports the peak. v9 is still the better CONTROL performer (mazF 9 vs 13, and a
# wider hupA-clean band), so it is worth re-running rather than deleting.
#
# v9 -- ANY-SIGNAL detection on the FIJI-transformed values, deduplicated in (time, y).
#
# The rule shape v8 should have been. The transform has already done the thresholding: clip
# at 25% of the trench's own max, ImageJ median r2 on the assembled kymograph, drop sub-10px
# components. Whatever survives IS the candidate set, so there is no baseline, no MAD height,
# no prominence and no fold ratio -- the three things that all collapse on a ~99%-zero column
# and sank v8.
#
# A candidate is any cell-frame with a nonzero value. That needs no choice of statistic:
# fiji_mean > 0, fiji_max > 0 and fiji_frac > 0 are the IDENTICAL set of 32,979 cell-frames,
# verified, because all three read the same above-floor pixels.
#
# Deduplication is spatial-temporal connected components, not the concurrent merge: link two
# lit cell-frames when |dt| <= link_dt and |dy| <= link_dy_cell_lengths * median cell length
# * |dt|, then each component is one event. The |dt| scaling is not cosmetic -- a track
# drifts -4.7 px/frame (median, -17 at p10), so a fixed tolerance fragments any advecting
# burst. The cell length is the TRENCH's own median axis_major_length, which keeps the
# tolerance scale-free for the ~1/3 linear scale DUMM_giTG66_Glucose_012325.
#
# `lit_gate` is mandatory, and it is what the per-trench floor makes it. Because the floor is
# 25% of each trench's OWN max, a uniformly bright trench lights up entirely: hupA is 99.9%
# lit across all 6 of its trenches and mazF 85.1%, against araB 4.9% and lacZ 13.6%. The lit
# fraction runs INVERSELY to burstiness, so rejecting trenches above 60% lit takes hupA to
# exactly zero. It is not free: v6 coverage drops 96% -> 76% because some >60%-lit trenches
# hold real v6 events, and chpS falls to 4 despite being a known burster.
#
# `lit_gate` is 0.90, RAISED from 0.60 on 090926 when small-object removal was turned off.
# Dropping that step raises the lit fraction (40.3% -> 48.9% of cell-frames, and 86 -> 117 of
# 255 trenches above 60% lit), so a gate tuned on the speckled distribution rejected too much
# -- murQ, the top gene by rate, went from 0 of 4 sampled trenches rejected to 3 of 4.
#
# 0.90 dominates rather than trades. Swept 0.60/0.70/0.80/0.90: hupA is 0 at EVERY value (it
# cannot break until 0.994, its own minimum lit fraction) and mazF is FLAT at 9 from 0.70 up,
# because mazF's lit fractions jump from 0.31 straight to 0.98+ so the same single low-lit
# trench is the only one admitted anywhere in that range. Raising the gate therefore costs
# nothing on the controls, and buys chpS 6 -> 16 (a known burster the 0.60 gate cut to 4) and
# v6 coverage 78% -> 96%.
#
# Measured at the shipped settings: 802 events over 208 of 255 trenches, 3/3 regression
# trenches, hupA 0, mazF 9, v6 coverage 96%. The previous with-speckle settings
# (min_size=10, lit_gate=0.60) gave 671 events / 145 trenches / hupA 0 / mazF 11 / 76%, and
# are reproducible from data/fiji_fluor_082026.csv + data/bursts_v9_090826.csv. Event duration median 3 frames, p90 20,
# max 87 -- treat the top end as unreliable, it is an over-merged lit trench rather than a
# burst. 265 events (39%) are single cell-frames; `min_cell_frames` filters them and ships
# at 1 (keep), because whether a one-frame excursion is an event is a biological call.
BURST_DEF_V9 = {
    **BURST_DEF_V8,
    'version': 'v9',
    'signal': 'per-trench lit cluster',   # startswith('per-trench') -> no family partition
    'fluor_col': 'fiji_mean',
    'fiji_fold': None,                    # no ratio route; the transform IS the threshold
    'lit_gate': 0.90,
    'link_dt': 2,
    'link_dy_cell_lengths': 1.5,
    'min_cell_frames': 1,
}

# v10 -- SUPERSEDES v9 as the definition on the FIJI-transformed values.
#
# v9 asked "is anything lit at all"; v10 asks "is there a sharp RISE", which is what the
# filtered mean-fluorescence traces actually show. It is burst v5's rule shape verbatim --
# same per-lineage-family x spatial-cluster max envelope, same +/-5 frame window, same
# episode merge -- with ONE substitution: the baseline. See attach_fiji_scale for why the
# (gene, experiment) median + MAD cannot be used on this column, and why the trench's own
# brightest cell-mean is the right scale. `k` and `k_prom` are therefore FRACTIONS, not MAD
# multipliers, which is why they look nothing like v5's 4.0 / 3.7.
#
# `fold_prom` is off: the fold route needs `max(left_base, right_base) > 0` and these traces
# rise from exactly 0, so it is inert here (measured: identical event sets with and without).
#
# Measured at the shipped settings: 352 events, 3/3 regression trenches, hupA 0, mazF 13,
# chpS 35, and 91 of 156 v6 events matched at the EXACT frame -- against v9's 16, because
# keeping v5's envelope-and-peak machinery means it reports the same representative frames
# v6 does.
#
# CAVEAT, stated because it is narrower than v9's: hupA is 0 only for k >= 0.60 at
# k_prom = 0.40 (two points tested in that band, not a fine grid), and mazF/chpS are not
# zero. v10 wins on frame-level agreement with v6, not on control cleanliness. v9 remains
# reproducible from data/bursts_v9_nospeckle_090926.csv.
BURST_DEF_V10 = {
    **BURST_DEF_V5,
    'version': 'v10',
    'fluor_col': 'fiji_mean',
    'k': 0.60,          # peak >= 0.60 x this trench's brightest cell-mean
    'k_prom': 0.40,     # and rises by >= 0.40 x it  <- the "sharp increase"
    'fold_prom': None,  # inert on this column: bases are 0
}

# v11 -- seed on the trench's own LIT distribution, cluster in (t, y), report the peak.
#
# The problem v11 exists for, measured on the shipped sets: v10's bar is 0.60 x the trench's
# single brightest cell-mean, so it returns a median of ONE event per trench (mean 1.74, max
# 8 over 202 trenches) and a trench's second and third excursions are invisible by
# construction. In 114 of 236 trenches that bar sits ABOVE the trench's own lit Tukey fence
# -- i.e. across half the dataset v10 is stricter than "an outlier among this trench's own
# lit values". v9 has the opposite problem: no bar at all, 39% singletons, and it reports
# onset so it matches 16 of v6's 156 events where v10 matches 91.
#
# v11 = v9's candidate structure + a bar estimated on the LIT values only + peak reporting.
# The bar is the repair of v8's diagnosis rather than a repeat of it: v8 collapsed because
# median and MAD were taken over a zero-inflated column, and conditioning on lit removes the
# zero inflation while staying dimensionally matched (a cell-mean bar for a cell-mean
# measurement, unlike the rejected per-pixel display span).
#
# Constants are set by the sweep in the summary notebook; quote each with its measured band.
BURST_DEF_V11 = {
    **BURST_DEF_V9,
    'version': 'v11',
    'signal': 'per-trench lit fence cluster',
    'seed_rule': 'mad',         # 'tukey' | 'mad' | 'otsu'
    'iqr_k': 1.5,               # unused under seed_rule='mad'
    'k': 4.0,                   # median + k * 1.4826 * MAD, over the LIT values only
    'min_lit': 5,               # below this many lit cell-frames a quantile is meaningless
    'sparse_admit': True,       # ... and those trenches admit every lit cell-frame instead
    'lit_gate': 0.95,
    'min_frames': 2,            # seed-cluster duration in distinct frames -- a NEW axis
    'frac_min': 0.0,            # measured INERT; see the sweep note below
}
# Why these, and the band for each (bursts.md §7 v11 has the tables):
#
#   seed_rule='mad'  The lit TUKEY fence fails the regression outright -- 1 of 3 trenches at
#       every multiplier from 1.5 to 6.0. On a trench where the burst is a large share of the
#       lit values it inflates Q3 and the IQR together: 003/1598 is 5% lit, so its fence
#       (2.080) lands ABOVE the burst itself (2.062), where median+4*MAD sits at 0.935.
#       'otsu' passes 3/3 and is knob-free, but costs controls at matched recall (at gate
#       0.95 / min_frames 2 it gives 542 events with mazF 13 / chpS 44 against mad's 469 with
#       9 / 34). Kept selectable; not shipped.
#   k=4.0  INHERITED, not fitted -- it is the same multiplier v1-v7 use for the height bar.
#       There is no plateau here: at gate 0.95 / min_frames 2, k = 3.0/3.5/4.0/4.5/5.0 gives
#       550/496/469/446/410 events and chpS 49/39/34/29/25, monotone. So k is an honest
#       recall/control trade knob, and it is set by precedent rather than tuned.
#   lit_gate=0.95  hupA holds at 0 for EVERY gate up to 0.995 and breaks only when the gate
#       is removed (1.01 -> 5 events), because hupA's own minimum lit fraction is 0.9959.
#       That band is far wider than v10's (hupA clean only for k >= 0.60, two points tested)
#       and wider than v9's 0.60-0.90. 0.95 sits inside it with margin.
#   min_frames=2  A NEW axis: bursts.md records no swept duration gate anywhere. It is the
#       minimum statement of persistence -- "seen in at least two frames" -- and it is what
#       lets the lit gate be relaxed from v9's 0.90 to 0.95. At gate 0.95, min_frames
#       1/2/3/4 gives 789/469/353/262 events and chpS 51/34/24/18.
#   frac_min=0.0  Measured INERT. At gate 0.95 / min_frames 2, frac_min 0.2 removes 37 events
#       (469 -> 432) and changes the controls by nothing at all (mazF 9, chpS 34 either way).
#       `fiji_frac` had never been thresholded by any definition; now it has been, and it
#       buys nothing. Kept as a parameter, shipped off.
#   sparse_admit=True  Also measured inert: identical event counts either way at every swept
#       setting, because min_frames >= 2 already removes every trench with fewer than 5 lit
#       cell-frames. Kept because it documents a real choice.
#   link_dt=2, link_dy_cell_lengths=1.5  INHERITED from v9, not re-tuned. Sensitivity at the
#       shipped setting: (1,1.0) 601 events / (2,1.5) 469 / (3,2.5) 402, with v6 exact frames
#       85/89/89 -- so the choice moves the count by 1.5x and the frame agreement barely.

BURST_DEF = BURST_DEF_V3          # shipped default; bursts_082026.csv was made with this
BURST_DEFS = {'v1': BURST_DEF_V1, 'v2': BURST_DEF_V2, 'v3': BURST_DEF_V3,
              'v4': BURST_DEF_V4, 'v5': BURST_DEF_V5, 'v6': BURST_DEF_V6,
              'v7': BURST_DEF_V7, 'v8': BURST_DEF_V8, 'v9': BURST_DEF_V9,
              'v10': BURST_DEF_V10, 'v11': BURST_DEF_V11}

# Route precedence, strongest first. Used to pick an episode's representative peak and the
# survivor of a concurrent merge. A confirmed bright peak must always outrank a ratio-only
# one, or a fold candidate on a rising shoulder absorbs the real peak behind it.
ROUTE_RANK = {'absolute': 0, 'fold': 1, 'fold_left': 2, 'tukey': 3, 'fiji': 4,
              'lit': 5, 'lit_fence': 6}
# 'tukey' ranks LAST deliberately: it is the weakest evidence, so it is always the
# candidate dropped when it collides with a v6-route event in a concurrent merge, and
# never the one that represents a mixed episode. That is what lets v7 add without
# relabelling anything v6 already found.


def prom_multiplier(bdef):
    """The prominence multiplier a definition asks for, falling back to `k` when unset.

    Keeps v1-v3 bit-identical under the v4 signature: they leave `k_prom` None, so the single
    `k` still drives both bars.
    """
    kp = bdef.get('k_prom')
    return bdef['k'] if kp is None else kp


# ---------------------------------------------------------------------------------------
# The input filter -- applied before ANY definition runs
# ---------------------------------------------------------------------------------------

# Not part of a version's rule: the same status as the failed-run exclusion.
#
# RELATIVE, not an absolute px^2 threshold. DUMM_giTG66_Glucose_012325 was recorded at ~1/3
# the linear pixel scale of every other run (cells, widths AND trenches all ~0.34x; median
# cell area 33 px^2 against 263 px^2 elsewhere), and its masks are geometrically sound --
# area/(L*W) = 0.783 vs 0.767, both matching pi/4 for an ellipse. An absolute 50 px^2 floor
# deleted 749 of its 844 good cell-frames; the relative floor drops 121 across the dataset, of
# which 1. Nothing in the pipeline records pixel size, so the threshold has to be scale-free,
# exactly as delta_y_cell_lengths is.
MIN_CELL_AREA_FRAC = 0.05          # 5% of the experiment's own median cell area


def drop_degenerate_masks(df, frac=MIN_CELL_AREA_FRAC, exp_median_area=None, verbose=True):
    """Drop cell-frames whose mask is below `frac` of their experiment's median cell area.

    Sorted burst-cell areas were `[1, 1, 1, 2, 2, 117, 131, ...]`: only 0.11% of cell-frames
    are <=5 px^2, yet 5 of 95 v2 bursts landed on them -- a 48x enrichment. On a 1-pixel mask
    the mean intensity is a single raw pixel with no averaging, which is exactly how a false
    peak clears a threshold built from properly-averaged cells.

    Args:
        exp_median_area: Series indexed by experiment_name. Pass the medians computed on the
            FULL table, so the floor does not move when `df` is scoped to a subset. Computed
            from `df` when omitted.

    The annotated exports already carry the equivalent verdict as `is_degenerate_mask`; using
    that column is preferred, because it was computed against the full 23-experiment medians.
    """
    if exp_median_area is None:
        exp_median_area = df.groupby('experiment_name')['area'].median()
    floor = frac * df['experiment_name'].map(exp_median_area)
    assert floor.notna().all(), 'an experiment is missing from exp_median_area'
    keep = df['area'] >= floor
    if verbose:
        print(f"area floor {frac:.0%} of each experiment's own median: "
              f'{len(df):,} -> {int(keep.sum()):,} cell-frames '
              f'({int((~keep).sum()):,} dropped, {100 * (~keep).mean():.2f}%)')
    return df[keep].copy()


# ---------------------------------------------------------------------------------------
# v1 core -- the detector every version calls
# ---------------------------------------------------------------------------------------

def _link_ty_components(frames, ys, tol, link_dt):
    """Connected components of cell-frames in (time, y). Returns one label per row.

    Two cell-frames join when they are within `link_dt` frames AND within `tol` pixels in y,
    that y tolerance SCALED by the frame gap because cells advect down the trench (-4.7
    px/frame median, -17 at p10) -- a fixed tolerance fragments any drifting burst.

    `gap != 0` is deliberate: two cells in the SAME frame never link directly, only through a
    shared neighbour in an adjacent frame. Two bursts running side by side in one trench are
    different events.

    `frames` must be sorted ascending; the inner loop stops at the first row beyond `link_dt`.
    Shared by v9's any-signal clustering and v11's seed clustering so the two cannot drift.
    """
    n = len(frames)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        j = i + 1
        # `frames` is sorted, so stop as soon as the gap exceeds link_dt
        while j < n and frames[j] - frames[i] <= link_dt:
            gap = frames[j] - frames[i]
            if gap != 0 and abs(ys[j] - ys[i]) <= tol * gap:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
            j += 1

    return [find(i) for i in range(n)]


def detect_bursts_lit(df, fluor_col='fiji_mean', lit_gate=0.60, link_dt=2,
                      link_dy_cell_lengths=1.5, min_cell_frames=1,
                      constitutive_genes=None):
    """Any-signal burst detection on the FIJI-transformed values, deduplicated in (time, y).

    A candidate is any cell-frame with `fluor_col` > 0 -- i.e. some of the cell survived the
    0.25*max clip AND the small-object removal. Candidates in one kymograph are linked into
    events by connected components: two lit cell-frames join when they are within `link_dt`
    frames AND within `link_dy_cell_lengths` cell lengths in y, that y tolerance SCALED by
    the frame gap because cells advect (-4.7 px/frame median, -17 at p10). Each component is
    one event, reported at its earliest frame -- onset, matching merge_keep='earliest'.

    The cell length is the trench's own median `axis_major_length`, so the tolerance is
    scale-free; an absolute pixel figure would mis-merge the ~1/3-scale
    DUMM_giTG66_Glucose_012325 (median cell area 34 px^2 against 263-398 elsewhere).

    `lit_gate` rejects a trench where more than this fraction of its cell-frames are lit.
    Mandatory, not tuning: the floor is 25% of each trench's own max, so a uniformly bright
    trench lights up completely and the lit fraction runs inversely to burstiness (hupA
    99.9%, mazF 85.1%, araB 4.9%). At 0.60 hupA goes to exactly zero. See the BURST_DEF_V9
    comment for what that costs.

    `min_cell_frames` drops events made of fewer than this many lit cell-frames; 39% of
    events are singletons, so 1 (the default) keeps them and >1 is a real biological filter.

    Emits `detect_bursts`' columns so add_relative_y and build_burst_csv work unchanged, with
    route='lit'. `threshold` is 0.0 because the real threshold lives inside the transform,
    as the trench's own display floor -- carried separately as `fiji_vmin` where available.
    """
    constitutive_genes = constitutive_genes or set()
    records = []

    for (exp, fov, trench), df_trench in df.groupby(KYMO_KEY):
        gene = df_trench['gene'].iloc[0]
        lit = df_trench[df_trench[fluor_col] > 0]
        if lit.empty:
            continue
        if len(lit) / len(df_trench) > lit_gate:
            continue                      # uniformly lit: constitutive, not a burst

        cell_len = float(df_trench['axis_major_length'].median())
        if not np.isfinite(cell_len) or cell_len <= 0:
            continue
        tol = link_dy_cell_lengths * cell_len

        d = lit.sort_values('time_frame').reset_index(drop=True)
        d['_ev'] = _link_ty_components(d['time_frame'].to_numpy(),
                                       d['centroid_y'].to_numpy(), tol, link_dt)
        for _, ev in d.groupby('_ev', sort=False):
            if len(ev) < min_cell_frames:
                continue
            ev = ev.sort_values('time_frame')
            onset = ev.iloc[0]                              # report burst onset
            peak = ev.loc[ev[fluor_col].idxmax()]
            records.append({
                'experiment_name': exp, 'FOV': fov, 'trench_id': trench,
                'track_id': int(onset['track_id']),
                'predicted_lineage': onset['predicted_lineage'],
                'node_id': int(onset['node_id']),
                'time_frame': int(onset['time_frame']),
                'fluor': float(onset[fluor_col]), 'gene': gene,
                'threshold': 0.0, 'route': 'lit',
                'n_cell_frames': int(len(ev)),
                'n_frames': int(ev['time_frame'].max() - ev['time_frame'].min() + 1),
                'peak_fluor': float(peak[fluor_col]),
                'peak_time_frame': int(peak['time_frame']),
                'lit_frac': float(len(lit) / len(df_trench)),
            })

    cols = ['experiment_name', 'FOV', 'trench_id', 'track_id', 'predicted_lineage',
            'node_id', 'time_frame', 'fluor', 'gene', 'threshold', 'route',
            'n_cell_frames', 'n_frames', 'peak_fluor', 'peak_time_frame', 'lit_frac',
            'is_constitutive']
    if not records:
        return pd.DataFrame(columns=cols)
    bursts = pd.DataFrame(records)
    bursts['is_constitutive'] = bursts['gene'].isin(constitutive_genes)
    return bursts.reset_index(drop=True)


def detect_bursts_seeded(df, fluor_col='fiji_mean', thresh_col='_lit_thresh',
                         lit_gate=0.90, link_dt=2, link_dy_cell_lengths=1.5,
                         min_frames=1, frac_min=0.0, frac_col='fiji_frac',
                         constitutive_genes=None):
    """v11: seed on a trench's own lit distribution, cluster in (t, y), report the PEAK.

    v9 accepts every lit cell-frame and reports each cluster's onset. v10 accepts a peak that
    reaches 0.60 of the trench's single brightest cell-mean. Both are answers to the wrong
    question. v9's bar is so low that 39% of its events are single cell-frames with a peak
    around 0.1 counts; v10's is anchored to one order statistic, so it finds a median of ONE
    event per trench and a trench's second and third excursions cannot register -- measured,
    in 114 of 236 trenches v10's bar sits ABOVE that trench's own lit Tukey fence.

    v11 keeps v9's structure and changes the two things that are actually wrong with it:

      1. SEEDS, not every lit cell-frame. A seed clears `thresh_col`, the bar
         attach_lit_baseline estimates on that trench's lit values only.
      2. The event is reported at its PEAK cell-frame, not its onset. v9's `time_frame` and
         `peak_time_frame` agree on only 51% of its rows, which is most of why it matched
         16 of v6's 156 events where v10 matched 91.

    Events are connected components of the SEED set under v9's linking rule (`_link_ty_components`,
    shared so the two cannot drift). Seeding first is what avoids v9's over-merge: its longest
    event spans 87 frames because a whole lit trench is one component, whereas a lit blob here
    yields as many events as it holds seed clusters.

    The surrounding lit extent is measured by linking the LIT set as well and reporting, per
    event, the lit component its seeds fall in (`n_lit_cell_frames`, `lit_first`, `lit_last`).
    Reported, never gating -- duration is a property to look at, not yet a filter with a
    measured band.

    `lit_gate` is carried over from v9 and is doing the constitutive control's work: hupA is
    lit in 99.7% of its cell-frames, so the gate discards all seven of its trenches wholesale
    rather than discriminating events inside them. That is blunt, and it is the honest
    limitation of this route -- the fence itself flags hupA 2.20%, mazF 2.78% and chpS 2.63%
    of cell-frames against araB's 0.36%, i.e. the controls sit mid-pack, much as bursts.md
    §8 records for the raw-F Tukey fence.

    `min_frames` gates on how many distinct frames a seed cluster spans. bursts.md records no
    swept duration gate anywhere, so this is a genuinely new axis rather than a re-tread.
    `frac_min` gates on `fiji_frac` at the peak -- the fraction of the cell's own pixels above
    the floor, which separates a coherent bright cell from a speckle and which no definition
    has ever thresholded.
    """
    constitutive_genes = constitutive_genes or set()
    records = []

    for (exp, fov, trench), df_trench in df.groupby(KYMO_KEY):
        gene = df_trench['gene'].iloc[0]
        lit = df_trench[df_trench[fluor_col] > 0]
        if lit.empty:
            continue
        lit_frac = len(lit) / len(df_trench)
        if lit_frac > lit_gate:
            continue                      # uniformly lit: constitutive, not a burst

        assert df_trench[thresh_col].nunique(dropna=False) <= 1, (
            f'{thresh_col} varies within {exp} {fov}/{trench}; it must be per-trench')
        thresh = df_trench[thresh_col].iloc[0]
        if not np.isfinite(thresh):
            continue                      # too few lit cell-frames to place a bar

        cell_len = float(df_trench['axis_major_length'].median())
        if not np.isfinite(cell_len) or cell_len <= 0:
            continue
        tol = link_dy_cell_lengths * cell_len

        # The lit extent, for reporting. Same linking rule, run on the lit set.
        litd = lit.sort_values('time_frame').reset_index(drop=True)
        litd['_lc'] = _link_ty_components(litd['time_frame'].to_numpy(),
                                          litd['centroid_y'].to_numpy(), tol, link_dt)
        lit_ext = {c: (len(gg), int(gg['time_frame'].min()), int(gg['time_frame'].max()))
                   for c, gg in litd.groupby('_lc', sort=False)}

        seeds = litd[litd[fluor_col] > thresh]
        if frac_min > 0 and frac_col in seeds.columns:
            seeds = seeds[seeds[frac_col] >= frac_min]
        if seeds.empty:
            continue

        d = seeds.sort_values('time_frame').reset_index(drop=True)
        d['_ev'] = _link_ty_components(d['time_frame'].to_numpy(),
                                       d['centroid_y'].to_numpy(), tol, link_dt)

        # The trench's own lit values, for the peak's within-trench percentile.
        litvals = np.sort(lit[fluor_col].to_numpy(dtype=float))

        for _, ev in d.groupby('_ev', sort=False):
            n_frames = int(ev['time_frame'].nunique())
            if n_frames < min_frames:
                continue
            peak = ev.loc[ev[fluor_col].idxmax()]
            n_lit, l0, l1 = lit_ext.get(peak['_lc'], (len(ev), int(ev['time_frame'].min()),
                                                      int(ev['time_frame'].max())))
            records.append({
                'experiment_name': exp, 'FOV': fov, 'trench_id': trench,
                'track_id': int(peak['track_id']),
                'predicted_lineage': peak['predicted_lineage'],
                'node_id': int(peak['node_id']),
                'time_frame': int(peak['time_frame']),           # the PEAK, not the onset
                'fluor': float(peak[fluor_col]), 'gene': gene,
                'threshold': float(thresh), 'route': 'lit_fence',
                'n_cell_frames': int(len(ev)),
                'n_frames': n_frames,
                'peak_fluor': float(peak[fluor_col]),
                'peak_time_frame': int(peak['time_frame']),
                'lit_frac': float(lit_frac),
                'seed_first': int(ev['time_frame'].min()),
                'seed_last': int(ev['time_frame'].max()),
                'n_lit_cell_frames': int(n_lit),
                'lit_first': int(l0), 'lit_last': int(l1),
                'peak_frac': float(peak[frac_col]) if frac_col in ev.columns else np.nan,
                'peak_pct': float(np.searchsorted(litvals, peak[fluor_col], 'right')
                                  / len(litvals)),
            })

    cols = ['experiment_name', 'FOV', 'trench_id', 'track_id', 'predicted_lineage',
            'node_id', 'time_frame', 'fluor', 'gene', 'threshold', 'route',
            'n_cell_frames', 'n_frames', 'peak_fluor', 'peak_time_frame', 'lit_frac',
            'seed_first', 'seed_last', 'n_lit_cell_frames', 'lit_first', 'lit_last',
            'peak_frac', 'peak_pct', 'is_constitutive']
    if not records:
        return pd.DataFrame(columns=cols)
    bursts = pd.DataFrame(records)
    bursts['is_constitutive'] = bursts['gene'].isin(constitutive_genes)
    return bursts.reset_index(drop=True)


def attach_fiji_scale(df, fluor_col='fiji_mean', med_col='_base_med',
                      scale_col='_base_scale'):
    """Baseline columns that express v5's thresholds as fractions of the TRENCH'S OWN peak.

    Attaches `med = 0` and `scale = max(fluor_col)` per kymograph, for
    `detect_bursts(baseline_cols=...)`. `k` and `k_prom` then read as "peak must reach k x
    the brightest cell-mean in this trench" and "and rise by k_prom x it".

    Why not the (gene, experiment) median + MAD that v1-v7 use: on the FIJI-transformed
    column it COLLAPSES. Measured, 11 of 28 (gene, experiment) groups give median 0 with
    mad_scale floored at 1e-6 -- so height ~ 0 and every local maximum qualifies. Trenches
    in a collapsed group fire on 74.1% of their local maxima against 42.9% for healthy ones,
    and the collapsed groups are exactly the dim interesting genes (all of araB, all of lacZ,
    all of gfcE). A per-kymograph median+MAD does not rescue it either: ~35k rows still have
    zero scale and the controls get worse.

    Why the trench's own MAX and not the display span or a p90: dimensions. `fiji_mean` is
    averaged over a cell (~2-6 counts) while `fiji_vmax - fiji_vmin` is a per-PIXEL range
    (~25), so a span-scaled bar is the same size as the peak -- the mismatch that sank v8,
    and it yields 0 workable settings. The p90 cell-mean gives only 1 of 3 regression
    trenches. The max is the only scale tried that is both non-collapsing and dimensionally
    matched.

    Trenches whose max is 0 get NaN and are skipped by detect_bursts' isfinite guard.
    """
    out = df.copy()
    peak = out.groupby(KYMO_KEY, dropna=False, sort=False)[fluor_col].transform('max')
    out[med_col] = 0.0
    out[scale_col] = peak.where(peak > 0)
    return out


def attach_lit_baseline(df, fluor_col='fiji_mean', rule='tukey', iqr_k=1.5, k=4.0,
                        min_lit=5, sparse_admit=True, thresh_col='_lit_thresh',
                        frac_col='_lit_frac', n_col='_lit_n', tag_col='_lit_rule'):
    """A per-trench bar estimated on that trench's LIT cell-frames only -- v11's baseline.

    This is the repair of v8's diagnosed cause of death, not a re-run of it. v8 collapsed
    because it estimated location and scale over a ZERO-INFLATED column: 62.6% of cell-frames
    are exactly 0, so the median is 0 and the MAD floors at 1e-6, giving a bar of ~4e-6 that
    every blip clears. Conditioning on `fluor_col > 0` removes the zero inflation, and unlike
    the rejected `fiji_vmax - fiji_vmin` scale the result is dimensionally matched -- a
    cell-mean bar for a cell-mean measurement.

    It is also not v7's fence. attach_kymograph_fence computes Q3 and the IQR over ALL of a
    trench's cell-frames, which on this column are both 0; and it runs on raw `fluor_per_ms`,
    a different measurement entirely.

    Three rules, all swept:
      'tukey'  Q3 + iqr_k * IQR of the lit values      (shape-aware, iqr_k is the knob)
      'mad'    median + k * 1.4826 * MAD of the lit values
      'otsu'   1-D Otsu on log1p of the lit values     (KNOB-FREE)

    Otsu runs on log1p because a trench's lit values span orders of magnitude and Otsu
    maximises between-class variance, which on a linear scale is dominated by the single
    brightest cell -- the same failure that limits v10 to one event per trench.

    `min_lit` / `sparse_admit`: a quantile is meaningless on a handful of points, but a
    trench with three lit cell-frames in an otherwise dark movie is the clearest burst
    evidence in the dataset. So below `min_lit` the bar is set to 0 (admit every lit
    cell-frame) rather than NaN. Set `sparse_admit=False` to skip those trenches instead;
    both arms are swept. A trench with NO lit cell-frames always gets NaN and is skipped by
    the detector's isfinite guard.

    Attached as COLUMNS, for the same reason attach_kymograph_fence is: a statistic computed
    inside a detector would be per-subgroup, not per-trench.
    """
    out = df.copy()
    thresh = pd.Series(np.nan, index=out.index, dtype=float)
    lit_frac = pd.Series(np.nan, index=out.index, dtype=float)
    lit_n = pd.Series(0, index=out.index, dtype=int)

    for _, g in out.groupby(KYMO_KEY, dropna=False, sort=False):
        x = g[fluor_col].to_numpy(dtype=float)
        v = x[np.isfinite(x) & (x > 0)]
        lit_frac.loc[g.index] = len(v) / len(g) if len(g) else np.nan
        lit_n.loc[g.index] = len(v)
        if len(v) == 0:
            continue                                  # nothing lit -> no bar, trench skipped
        if len(v) < min_lit:
            if sparse_admit:
                thresh.loc[g.index] = 0.0             # so few lit frames that each is the news
            continue
        if rule == 'tukey':
            q1, q3 = np.percentile(v, [25, 75])
            t = q3 + iqr_k * (q3 - q1)
        elif rule == 'mad':
            m = np.median(v)
            t = m + k * 1.4826 * np.median(np.abs(v - m))
        elif rule == 'otsu':
            from skimage.filters import threshold_otsu
            lv = np.log1p(v)
            t = 0.0 if lv.min() == lv.max() else float(np.expm1(threshold_otsu(lv)))
        else:
            raise ValueError(f'unknown rule {rule!r}; use tukey, mad or otsu')
        thresh.loc[g.index] = t

    out[thresh_col] = thresh
    out[frac_col] = lit_frac
    out[n_col] = lit_n
    # A tag, so run_burst_detection_any can refuse a frame built for a different rule. The
    # bar lives in a column, so nothing else would notice that the definition said 'tukey'
    # and the attached numbers were 'mad' -- it would just silently report the wrong set.
    out[tag_col] = _lit_tag(rule, iqr_k, k, min_lit, sparse_admit)
    return out


def _lit_tag(rule, iqr_k, k, min_lit, sparse_admit):
    knob = iqr_k if rule == 'tukey' else k if rule == 'mad' else None
    return f'{rule}|{knob}|{min_lit}|{bool(sparse_admit)}'


def attach_lit_baseline_for(df, bdef):
    """attach_lit_baseline with the constants a definition asks for. Use this, not the raw
    call, so the dict is the single source of truth for the bar."""
    return attach_lit_baseline(df, fluor_col=bdef['fluor_col'],
                               rule=bdef['seed_rule'], iqr_k=bdef.get('iqr_k', 1.5),
                               k=bdef.get('k', 4.0), min_lit=bdef.get('min_lit', 5),
                               sparse_admit=bdef.get('sparse_admit', True))


def attach_fiji_gate(df, fluor_col='fiji_mean', vmin_col='fiji_vmin',
                     vmax_col='fiji_vmax', col='_fiji_activity'):
    """Per-kymograph 'how much of this trench is already lit', for the v8 constitutive gate.

    `activity = median(fluor_col) / (vmax - vmin)` over the trench's cell-frames, i.e. where
    the TYPICAL cell sits above its own display floor as a fraction of the display span.
    Unit-free, so one bar serves the dataset's ~283x spread of gene brightness.

    Attached as a COLUMN rather than computed inside the detector, for the same reason
    attach_kymograph_fence is: by the time detection runs, `trench_id` has been rewritten to
    trench#partition, so a statistic computed there would be per-FAMILY, not per-trench.

    This is what keeps the constitutive control at zero BY MECHANISM. hupA is uniformly
    bright, so every cell clears its trench's 0.25*max floor (measured: 0.0% zeros, median
    356 of a 1312 span -> activity 0.27) and a naive "nonzero = candidate" rule would fire on
    all of it. A bursting trench has a dark median instead (008/1488: median 0 -> activity 0).
    Note the discriminator is the MEDIAN, not the nonzero FRACTION: 008/1488 is 52.9% nonzero,
    so a 0.5 fraction gate would reject one of the three regression trenches.
    """
    out = df.copy()
    span = (out[vmax_col] - out[vmin_col]).replace(0, np.nan)
    g = out.groupby(KYMO_KEY, dropna=False, sort=False)
    med = g[fluor_col].transform('median')
    out[col] = med / span
    # The display span, for the fold denominator floor. Same per-trench-column reasoning as
    # the gate: by detection time `trench_id` is trench#partition.
    out['_fiji_span'] = span
    return out


def detect_bursts_fiji(df, fluor_col='fiji_mean', window=5, fiji_fold=3.0, fiji_floor=0.0,
                       fiji_max_activity=0.05, fiji_base_floor_frac=0.05,
                       gate_col='_fiji_activity', span_col='_fiji_span',
                       constitutive_genes=None):
    """Burst detection on the FIJI-transformed per-cell values.

    Same envelope and same episode-merge shape as `detect_bursts`, but every threshold is
    per-trench or unit-free, because the (gene, experiment) median+MAD baseline collapses on
    a ~99%-zero column (see the BURST_DEF_V8 comment).

    A candidate is a local maximum on the trench max-envelope that
      (a) clears `fiji_floor` -- i.e. is nonzero, meaning some of the cell survived the
          0.25*max clip AND the small-object removal, and
      (b) reaches `fiji_fold` x its own LEFT base, where that base is FLOORED at
          `fiji_base_floor_frac` x the trench's display span. Left rather than
          max(left, right) for the v6 reason: the decay here is dilution by growth, not
          protein loss, so the right base has not returned to the pre-burst floor and
          max-base penalises long events.

    Flooring the denominator is not optional, and unlike v6 it cannot be a fraction of the
    gene median, which is 0 here. An eps-only floor makes the ratio test INERT: with a left
    base of 0 every nonzero peak scores peak/1e-9 and passes, which is why an early version
    of this returned ~1000 events and barely responded to `fiji_fold` at all. Same trap
    bursts.md records for v6 (chpS 010/46 scoring 30x on a left_base of 0.006).
    and its whole trench must pass the `gate_col` activity gate.

    Prominence is a RATIO because the MAD form is unavailable -- with a baseline of 0 there
    are no MAD units to express a rise in. The episode reset is the same ratio, matching how
    the fold routes reset in detect_bursts.

    Emits the same columns as `detect_bursts` so merge_concurrent_bursts and build_burst_csv
    work unchanged. `threshold` carries the trench's floor, not a gene bar -- the
    `baseline_scope` column records that.
    """
    constitutive_genes = constitutive_genes or set()
    records = []
    eps = 1e-9

    for (exp, fov, trench), df_trench in df.groupby(['experiment_name', 'FOV', 'trench_id']):
        gene = df_trench['gene'].iloc[0]
        if gate_col in df_trench.columns:
            assert df_trench[gate_col].nunique(dropna=False) == 1, (
                f'{gate_col} varies within {exp} {fov}/{trench}; the gate is per-trench')
            activity = df_trench[gate_col].iloc[0]
            if np.isfinite(activity) and activity > fiji_max_activity:
                continue                      # constitutive / uniformly lit: not a burst
        span = np.nan
        if span_col in df_trench.columns:
            span = df_trench[span_col].iloc[0]
        den_floor = fiji_base_floor_frac * span if np.isfinite(span) else 0.0

        env = trench_max_envelope(df_trench, fluor_col=fluor_col)
        sig = env[fluor_col].to_numpy()
        if len(sig) < 3:
            continue

        fpk, _ = find_peaks(sig, distance=window)
        if not len(fpk):
            continue
        _, lb, _rb = peak_prominences(sig, fpk)
        cand = {}
        for i, pi in enumerate(fpk):
            if sig[pi] <= fiji_floor:
                continue
            if sig[pi] < sig[max(0, pi - window):min(len(sig), pi + window + 1)].max():
                continue
            base_l = max(sig[lb[i]], den_floor, eps)
            if sig[pi] / base_l >= fiji_fold:
                cand[int(pi)] = 'fiji'

        episodes, cur_peak, last_i, reset = [], -np.inf, None, fiji_floor
        for pi in sorted(cand):
            if (last_i is None
                    or sig[last_i:pi + 1].min() < reset
                    or sig[pi] > cur_peak * fiji_fold):
                episodes.append([])
                cur_peak = sig[pi]
                # The ratio analogue of "returned to baseline", as for the fold routes:
                # fall back by the factor that qualified the peak. Never `fiji_floor` alone,
                # which a decaying hump sits above for many frames and would reopen an
                # episode on every wobble.
                reset = max(sig[pi] / fiji_fold, fiji_floor)
            episodes[-1].append(pi)
            last_i = pi

        for ep in episodes:
            pi = min(ep, key=lambda p: (ROUTE_RANK[cand[p]], p))
            row = env.iloc[pi]
            records.append({
                'experiment_name': exp, 'FOV': fov, 'trench_id': trench,
                'track_id': int(row['track_id']),
                'predicted_lineage': row['predicted_lineage'],
                'node_id': int(row['node_id']), 'time_frame': int(row['time_frame']),
                'fluor': float(row[fluor_col]), 'gene': gene,
                'threshold': float(fiji_floor), 'route': cand[pi],
            })

    cols = ['experiment_name', 'FOV', 'trench_id', 'track_id', 'predicted_lineage',
            'node_id', 'time_frame', 'fluor', 'gene', 'threshold', 'route',
            'is_constitutive']
    if not records:
        return pd.DataFrame(columns=cols)
    bursts = pd.DataFrame(records)
    bursts['is_constitutive'] = bursts['gene'].isin(constitutive_genes)
    return bursts.reset_index(drop=True)


def detect_bursts(df, k=4.0, window=5, min_prominence=None, fluor_col=FLUOR_COL,
                  constitutive_genes=None, k_prom=None, baseline_cols=None,
                  fold_prom=None, fold_height_floor_s=1.0, fold_left_prom=None,
                  fold_base_floor_frac=0.5, fence_col=None, tukey_iqr_k=None):
    """Detect fluorescence bursts on the per-group max-fluorescence envelope.

    For each (experiment, FOV, trench) the brightest cell at each frame forms one continuous
    signal (`trench_max_envelope`). Candidate peaks satisfy: (a) height > the gene x
    experiment threshold `median + k * mad_scale`, (b) local maximum within +/- `window`
    frames, and (c) prominence >= the required rise. Prominence is why flat-high constitutive
    genes score ~0. Using the max envelope means a daughter's inherited fluorescence is part
    of a decaying hump, so division carryover is not miscounted.

    Episode merge (the key rule): a single burst that decays over many frames must count
    ONCE, not once per wobble on the way down. Walking candidate peaks in time, a later peak
    only opens a NEW burst when either the envelope first drops back below threshold (a reset
    to baseline, then re-rises) OR it climbs to a genuinely higher new maximum
    (> previous burst peak + the required prominence). Otherwise it is the same
    decaying/plateauing burst and is skipped.

    Callers pass the partition through `trench_id` -- see `run_burst_detection_any` -- so this
    function is unchanged from v1 and needs no knowledge of families or spatial clusters.

    Args:
        df: per-cell-per-frame DataFrame (needs gene, experiment_name, FOV, trench_id,
            track_id, time_frame, node_id, predicted_lineage, fluor_col).
        k: robust z-score multiplier for the absolute height threshold.
        window: +/- frames for the local-maximum check and minimum peak separation.
        min_prominence: absolute required rise; overrides `k_prom` when given.
        fluor_col: fluorescence column (default a.u./ms).
        constitutive_genes: optional set; only tags an 'is_constitutive' column.
        k_prom: multiplier for the required prominence. None -> `k`, which is what v1-v3 do.
        baseline_cols: ``(median_col, scale_col)`` to read the baseline from `df` instead of
            recomputing it per (gene, experiment). This is how an alternative baseline SCOPE
            is evaluated -- see `attach_kymograph_baselines`. The columns must be constant
            within each detection group, which is asserted, because the threshold has to be
            one number per envelope.
        fold_prom: enables the v5 FOLD route as a second way in -- a peak whose
            ``peak / max(left_base, right_base)`` reaches this ratio qualifies even when it
            never clears `height`. Unit-free, so one bar serves a 283x spread of gene
            brightness. None (v1-v4) leaves the absolute rule as the only route.
        fold_height_floor_s: on either ratio route, the peak must also sit this many
            MAD-scales above the gene median. Guards the ratio's denominator.
        fold_left_prom: enables the v6 FOLD-LEFT route -- the same ratio taken against the
            LEFT base alone. ``max(left, right)`` binds on the right whenever a burst's decay
            has not returned to its pre-burst floor, which is the normal case here (the decay
            is dilution by growth, not protein loss), so the max-base rule systematically
            penalises long-decay events: of the 84 peaks v3 rejected on prominence, the right
            base decides 53. None (v1-v5) leaves this route off.
        fold_base_floor_frac: floors the fold-left denominator at this fraction of the
            baseline median. NOT optional: unfloored, a near-zero left base manufactures an
            arbitrary ratio (chpS 010/46 scores 30x on a left_base of 0.006). Bracketed on
            both sides at 0.5 -- 0.25 leaves the controls loose, 0.75 rises above the
            motivating case's own left base and loses it.

    The routes are OR'd, and every row carries `route` ('absolute', 'fold' or 'fold_left')
    recording which admitted it. Neither ratio route substitutes for the absolute rule:
    measured, fold alone finds 67 events and fails all three regression trenches, because it
    misses the bright canonical bursts. They only ever add, which is what makes each version
    a strict superset of the last.

    Returns:
        DataFrame with one row per burst event: experiment_name, FOV, trench_id, track_id,
        predicted_lineage, node_id, time_frame, fluor, gene, threshold, route,
        is_constitutive.
        (track_id / node_id / predicted_lineage identify the brightest cell at the peak.)
    """
    baselines = None if baseline_cols else compute_baselines(df, fluor_col=fluor_col)
    constitutive_genes = constitutive_genes or set()
    kp = k if k_prom is None else k_prom
    records = []

    for (exp, fov, trench), df_trench in df.groupby(['experiment_name', 'FOV', 'trench_id']):
        gene = df_trench['gene'].iloc[0]
        if baseline_cols:
            med_col, scale_col = baseline_cols
            med = df_trench[med_col].iloc[0]
            scale = df_trench[scale_col].iloc[0]
            # <= 1, not == 1: an all-NaN group (a trench with no usable scale) gives
            # nunique() == 0 and used to trip this assert BEFORE the isfinite guard below
            # could skip it. Varying values still give >= 2 and still raise.
            assert df_trench[med_col].nunique() <= 1 and df_trench[scale_col].nunique() <= 1, (
                f'{baseline_cols} vary within {exp} {fov}/{trench}; a detection group must '
                'have one threshold')
            if not np.isfinite(med) or not np.isfinite(scale):
                continue
        else:
            if (gene, exp) not in baselines.index:
                continue
            med = baselines.loc[(gene, exp), 'median']
            scale = baselines.loc[(gene, exp), 'mad_scale']
        height = med + k * scale
        prom = kp * scale if min_prominence is None else min_prominence

        # The trench's own Tukey fence, for the v7 route. Read as a COLUMN rather than
        # recomputed here, for the same reason `baseline_cols` is: by the time this loop runs,
        # `trench_id` has been rewritten to trench#partition (run_burst_detection_any), so
        # `df_trench` is a lineage-family x spatial-cluster subgroup and a fence computed from
        # it would be per-FAMILY, not per-trench. A pre-attached column is the only way to get
        # a genuine per-kymograph number in here. Constant within the group by construction,
        # since every partition subgroup is a subset of one trench -- asserted anyway.
        fence = None
        if fence_col is not None and tukey_iqr_k is not None:
            assert df_trench[fence_col].nunique() == 1, (
                f'{fence_col} varies within {exp} {fov}/{trench}; a detection group must have '
                'one fence')
            fence = df_trench[fence_col].iloc[0]
            if not np.isfinite(fence):
                fence = None

        env = trench_max_envelope(df_trench, fluor_col=fluor_col)
        sig = env[fluor_col].to_numpy()
        if len(sig) < 3:
            continue

        peaks, _ = find_peaks(sig, height=height, distance=window, prominence=prom)
        # explicit +/- window local-maximum check on the envelope
        cand = {int(pi): 'absolute' for pi in peaks
                if sig[pi] >= sig[max(0, pi - window):min(len(sig), pi + window + 1)].max()}

        # --- the fold route (v5) -------------------------------------------------------
        # A second way in, for a peak that is large RELATIVE to its own surroundings but
        # small in absolute terms because its whole trench sits below the gene median. Such
        # a trench can never clear `height`, so the absolute route discards it before
        # prominence is ever evaluated -- see the v5 changelog entry for the worked case.
        # v6 adds a third route on the same peaks, dividing by the LEFT base only. Order
        # matters and is the whole mechanism: `setdefault` never overwrites, so trying
        # 'fold' before 'fold_left' labels a peak with the strongest route that admits it.
        if fold_prom is not None or fold_left_prom is not None or fence is not None:
            fpk, _ = find_peaks(sig, distance=window)
            if len(fpk):
                _, lb, rb = peak_prominences(sig, fpk)
                floor = med + fold_height_floor_s * scale
                # Flooring the denominator is what keeps the left-base route safe: an
                # unfloored near-zero base turns a noise wobble into an arbitrary ratio.
                den_floor = fold_base_floor_frac * med
                for i, pi in enumerate(fpk):
                    if sig[pi] < sig[max(0, pi - window):min(len(sig), pi + window + 1)].max():
                        continue
                    # --- the tukey route (v7) ---------------------------------------------
                    # Evaluated HERE, above the `floor` test, and that placement is
                    # load-bearing. `floor` is `med + fold_height_floor_s*scale`: a HEIGHT
                    # condition on the gene baseline. The entire claim for the fence is that it
                    # reaches peaks no height bar reaches, so inheriting the floor would defeat
                    # it -- measured, it costs exactly the two near-misses v7 exists for.
                    # `setdefault` is called after the fold routes below, so a peak they admit
                    # keeps the stronger label; this only records the route, it does not gate.
                    _tukey_hit = fence is not None and sig[pi] > fence
                    if sig[pi] < floor:
                        if _tukey_hit:
                            cand.setdefault(int(pi), 'tukey')
                        continue
                    # The floor guards the denominator: fold divides by `base`, and a noise
                    # wobble in a near-zero trench would otherwise manufacture a huge ratio.
                    if fold_prom is not None:
                        base = max(sig[lb[i]], sig[rb[i]])
                        if base > 0 and sig[pi] / base >= fold_prom:
                            cand.setdefault(int(pi), 'fold')
                    if fold_left_prom is not None:
                        base_l = max(sig[lb[i]], den_floor)
                        if base_l > 0 and sig[pi] / base_l >= fold_left_prom:
                            cand.setdefault(int(pi), 'fold_left')
                    # Last, so 'absolute' < 'fold' < 'fold_left' < 'tukey' all keep priority.
                    if _tukey_hit:
                        cand.setdefault(int(pi), 'tukey')

        # episode merge: collapse a decaying burst to a single event.
        #
        # Two passes: group the candidates into episodes, then choose one representative
        # frame per episode. v1-v4 have a single route, so the ROUTE_RANK ordering below is
        # uniformly 0 and "strongest, then earliest" is just "earliest" -- bit-identical to
        # the original single-pass loop.
        episodes, cur_peak, last_i, reset = [], -np.inf, None, height
        for pi in sorted(cand):
            route = cand[pi]
            if (last_i is None                              # first candidate
                    or sig[last_i:pi + 1].min() < reset     # returned to baseline -> reset
                    or sig[pi] > cur_peak + prom):          # climbed to a higher maximum
                episodes.append([])
                cur_peak = sig[pi]
                # Route-appropriate reset floor. A ratio route's envelope never reaches
                # `height`, so reusing it would make the reset test always true and reopen a
                # new burst on every wobble of the decay. Requiring the envelope to fall
                # back by the factor that qualified the peak is the ratio analogue of
                # "returned to baseline", so each route resets on its own divisor. Measured
                # INERT on this dataset (same 118 events for v5 either way), like `window`;
                # kept because a long dim decaying hump is exactly what it guards, and that
                # is a data property, not a rule property.
                if route == 'tukey':
                    # A fence has no qualifying FACTOR to fall back by, unlike fold /
                    # fold_left, so the analogue of "returned to baseline" is "dropped back
                    # below the trench's own fence" -- the same statement as `reset = height`
                    # for the absolute route, with the trench fence substituted for the gene
                    # bar. Leaving this out would fall through to `reset = height`, which a
                    # tukey peak never reaches, making the reset test always true and reopening
                    # a burst on every wobble of a decay: silently inflated counts, no error.
                    reset = fence
                else:
                    _div = {'fold': fold_prom, 'fold_left': fold_left_prom}.get(route)
                    reset = height if _div is None else sig[pi] / _div
            episodes[-1].append(pi)
            last_i = pi

        for ep in episodes:
            # Represent the episode by its STRONGEST route, then the earliest frame
            # (ROUTE_RANK: absolute < fold < fold_left). Without this, a ratio candidate on
            # the rising shoulder opens the episode and then absorbs the genuine bright peak
            # behind it -- in alkA 001/807 v4 reported frame 4 at 0.3583 and v5 would have
            # reported frame 3 at 0.2457 instead, downgrading a real burst and mislabelling
            # its route. This is what keeps **v5 a strict superset of v4 and v6 of v5**,
            # which is the property that makes the definitions comparable at all.
            pi = min(ep, key=lambda p: (ROUTE_RANK[cand[p]], p))
            row = env.iloc[pi]
            records.append({
                'experiment_name': exp, 'FOV': fov, 'trench_id': trench,
                'track_id': int(row['track_id']),
                'predicted_lineage': row['predicted_lineage'],
                'node_id': int(row['node_id']), 'time_frame': int(row['time_frame']),
                'fluor': float(row[fluor_col]), 'gene': gene, 'threshold': float(height),
                'route': cand[pi],
            })

    cols = ['experiment_name', 'FOV', 'trench_id', 'track_id', 'predicted_lineage',
            'node_id', 'time_frame', 'fluor', 'gene', 'threshold', 'route',
            'is_constitutive']
    if not records:
        return pd.DataFrame(columns=cols)
    bursts = pd.DataFrame(records)
    bursts['is_constitutive'] = bursts['gene'].isin(constitutive_genes)
    return bursts.reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# v2 -- lineage families and the concurrent-burst merge
# ---------------------------------------------------------------------------------------

def add_lineage_family(df):
    """Add a `lineage_family` column (root track_id within the trench) to a cell table."""
    df = df.sort_values(KYMO_KEY + ['track_id', 'time_frame']).reset_index(drop=True)
    fam = np.empty(len(df), dtype=object)
    for _, g in df.groupby(KYMO_KEY, observed=True, sort=False):
        roots = lineage_family_roots(g)
        fam[g.index] = [roots[int(t)] for t in g['track_id']]
    df['lineage_family'] = fam
    return df


def merge_concurrent_bursts(bursts, bdef=None):
    """Collapse events that are the same burst seen in two lineage families.

    Per-family detection separates genuinely concurrent bursts, but it also splits a single
    burst across two ADJACENT cells the tracker put in different families -- e.g. tracks 44
    and 45 of trench 1488, which are neighbours 1.41 cell lengths apart. Two events are the
    same burst when their peaks are within `merge_window` frames AND their centroids are
    within `merge_gap_cell_lengths` of each other.

    The gap is measured in CELL LENGTHS, not pixels, for the same reason flag_jumps does it
    (mmtrack/track_displacement.py): a cell-length unit is independent of magnification and
    cell size. It is also an insensitive choice here -- 1.5, 2.0 and 2.5 all give the same
    total, because the concurrent pairs in this dataset sit at <=1.06 or >=2.77 cell lengths.

    `merge_keep='earliest'` reports the ONSET of the merged burst, so timing is not biased
    late by whichever neighbour happened to peak brighter.

    Returns (merged_bursts, audit) where `audit` has one row per concurrent pair considered.
    """
    bdef = BURST_DEF if bdef is None else bdef
    W = bdef['merge_window']
    GAP = bdef['merge_gap_cell_lengths']
    if bdef.get('merge_keep', 'earliest') != 'earliest':
        raise ValueError("only merge_keep='earliest' is implemented")

    drop, audit = set(), []
    for _, g in bursts.groupby(KYMO_KEY, observed=True):
        # Earliest first, brighter first on ties: the survivor of a merge is the onset.
        #
        # From v5, an ABSOLUTE event outranks a fold one regardless of timing. Two routes can
        # fire on neighbouring cells in different lineage families, and then plain
        # onset-ordering keeps whichever came first -- in alkA 001/807 that meant a dim fold
        # shoulder (frame 3, 0.2457) survived and the confirmed bright burst behind it
        # (frame 4, 0.3583) was dropped, downgrading a real event and mislabelling its route.
        # This ordering keeps **v5 a strict superset of v4**, and v6 of v5. It is
        # bit-identical for v1-v4, where every event is 'absolute' and the rank is uniformly 0.
        ordered = sorted(g.itertuples(),
                         key=lambda r: (ROUTE_RANK[getattr(r, 'route', 'absolute')],
                                        r.time_frame, -r.fluor))
        for x, y in itertools.combinations(ordered, 2):
            if abs(x.time_frame - y.time_frame) > W:
                continue                       # not concurrent -- nothing to decide
            length = np.nanmean([x.axis_major_length, y.axis_major_length])
            gap_cl = abs(x.centroid_y - y.centroid_y) / length
            # Already-dropped events must not absorb further ones, or a chain of neighbours
            # would collapse transitively into a single event.
            live = x.Index not in drop and y.Index not in drop
            merged = bool(gap_cl < GAP and live)
            audit.append({
                'gene': x.gene, 'experiment_name': x.experiment_name, 'FOV': x.FOV,
                'trench_id': x.trench_id,
                'frames': f'{int(x.time_frame)}/{int(y.time_frame)}',
                'families': f'{x.lineage_family}/{y.lineage_family}',
                'gap_px': round(abs(x.centroid_y - y.centroid_y), 1),
                'gap_cell_lengths': round(gap_cl, 2),
                'merged': 'yes' if merged else 'no',
                'kept_frame': int(x.time_frame) if merged else '-',
            })
            if merged:
                drop.add(y.Index)

    cols = ['gene', 'experiment_name', 'FOV', 'trench_id', 'frames', 'families',
            'gap_px', 'gap_cell_lengths', 'merged', 'kept_frame']
    audit = (pd.DataFrame(audit, columns=cols).sort_values('gap_cell_lengths')
             if audit else pd.DataFrame(columns=cols))
    return bursts.drop(index=drop), audit.reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# v3 -- spatial clusters inside a lineage family
# ---------------------------------------------------------------------------------------

def _spatial_clusters_one_frame(sub, split_gap):
    """Cluster one family's cells at one frame by gaps along y. Returns cluster index per row.

    Sorted by centroid_y; a new cluster starts wherever the gap to the previous cell exceeds
    `split_gap` cell lengths. In trench 1488 the separator is a genuinely dark cell sitting
    between the two bright regions, so the gap that splits them is already in the data.
    """
    order = np.argsort(sub['centroid_y'].to_numpy())
    y = sub['centroid_y'].to_numpy()[order]
    L = np.nanmean(sub['axis_major_length'].to_numpy())
    L = L if L > 1 else 1.0
    gaps = np.diff(y) / L
    cl = np.concatenate([[0], np.cumsum(gaps > split_gap)])
    out = np.empty(len(sub), dtype=int)
    out[order] = cl
    return out


def add_spatial_subgroup(df, bdef=None):
    """Add `spatial_group`: a lineage family subdivided by position, coherent through time.

    Per (trench, family): cluster by y-gap within each frame, then join clusters in
    CONSECUTIVE frames that SHARE A CELL -- same track_id, or a parent/daughter pair.
    Union-find over (frame, cluster) nodes gives groups that persist as the cells drift.

    Linking on cell identity, not on cluster position. An earlier version matched cluster mean
    centroids within 1 cell length and fragmented badly: clusters split and merge as cells
    divide, so the MEAN jumps far more than any individual cell does. That shattered the
    reference burst into 5 groups and left 52% of units under 3 frames (unusable by
    find_peaks). Shared identity is invariant to that.

    Consecutive-frames-only is the safeguard against fusing two bursts that occupy the same y
    at different times: there is no chain of links between them because the region went dark.
    """
    bdef = BURST_DEF_V3 if bdef is None else bdef
    split_gap = bdef['split_gap_cell_lengths']
    df = df.sort_values(KYMO_KEY + ['lineage_family', 'time_frame']).reset_index(drop=True)
    grp = np.empty(len(df), dtype=object)

    for (exp, fov, tr, fam), g in df.groupby(KYMO_KEY + ['lineage_family'], sort=False):
        node_of_row, members, parents_of = {}, {}, {}
        for fr, gf in g.groupby('time_frame', sort=True):
            cl = _spatial_clusters_one_frame(gf, split_gap)
            for idx, c in zip(gf.index, cl):
                node_of_row[idx] = (fr, int(c))
            tids = gf['track_id'].to_numpy()
            pars = gf['parent_track_id'].to_numpy()
            for c in np.unique(cl):
                members[(fr, int(c))] = set(tids[cl == c].tolist())
                parents_of[(fr, int(c))] = {int(p) for p in pars[cl == c] if pd.notna(p)}

        parent = {n: n for n in members}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        frames = sorted({f for f, _ in members})
        by_frame = {f: [n for n in members if n[0] == f] for f in frames}
        for f0, f1 in zip(frames, frames[1:]):
            if f1 - f0 != 1:
                continue                       # a frame gap breaks the chain, by design
            for a in by_frame[f0]:
                for b in by_frame[f1]:
                    # same cell carried over, or a daughter of a cell in `a`
                    if members[a] & members[b] or members[a] & parents_of[b]:
                        union(a, b)

        for idx in g.index:
            r = find(node_of_row[idx])
            grp[idx] = f'{fam}.{r[0]}_{r[1]}'

    df['spatial_group'] = grp
    return df


# ---------------------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------------------

def run_burst_detection_any(df, bdef, baseline_cols=None, fence_col=None):
    """Detect bursts under any definition, returning (bursts, merge_audit).

    Dispatches on the partition the definition asks for -- trench (v1), lineage family (v2),
    or family x spatial cluster (v3/v4) -- by temporarily encoding the partition into
    `trench_id`, which is what detect_bursts groups by. `trench_id` is restored before
    returning, so select_trench keeps working on the result.
    """
    assert tuple(bdef['baseline_keys']) == ('gene', 'experiment_name'), (
        'compute_baselines groups by (gene, experiment_name); a different baseline_keys in '
        'the definition would not be honoured. Change the function, not just the dict.')
    # A definition asking for the tukey route with no fence attached would silently return the
    # previous version's event set -- the same class of silent no-op the baseline_keys assert
    # above exists to prevent.
    assert not (bdef.get('tukey_iqr_k') is not None and fence_col is None), (
        f"{bdef['version']} asks for the tukey route but no fence_col was passed; it would "
        'silently reproduce v6. Call attach_kymograph_fence first and pass its column.')
    # v8 measures a different image, so it runs its own detector; detect_bursts and the
    # frozen v1-v3 sets are untouched by it. Everything downstream (partition encoding,
    # merge, audit) is shared.
    fiji_mode = bdef.get('fiji_fold') is not None
    # v9: any-signal + (time, y) clustering. Its own detector, and it returns before the
    # lineage-family partition and merge_concurrent_bursts -- the clustering IS the
    # deduplication, and running the concurrent merge on top would dedupe twice.
    lit_mode = bdef.get('lit_gate') is not None
    # v11: seeded clustering. Checked BEFORE lit_mode, because v11 carries a `lit_gate` too --
    # it inherits v9's dict. `seed_rule` is what distinguishes them.
    seed_mode = bdef.get('seed_rule') is not None
    sig = bdef['signal']
    # Which baseline the `threshold` column refers to. Load-bearing once one CSV carries
    # definitions run under different scopes: a per-trench bar and a per-gene bar are not
    # comparable numbers, and without this the column silently mixes them.
    scope = 'kymograph' if baseline_cols else 'gene_experiment'
    common = dict(k=bdef['k'], k_prom=bdef.get('k_prom'), window=bdef['window'],
                  min_prominence=bdef['min_prominence'], fluor_col=bdef['fluor_col'],
                  constitutive_genes=set(bdef['constitutive_genes']),
                  baseline_cols=baseline_cols,
                  fold_prom=bdef.get('fold_prom'),
                  fold_height_floor_s=bdef.get('fold_height_floor_s', 1.0),
                  fold_left_prom=bdef.get('fold_left_prom'),
                  fold_base_floor_frac=bdef.get('fold_base_floor_frac', 0.5),
                  fence_col=fence_col, tukey_iqr_k=bdef.get('tukey_iqr_k'))

    if seed_mode:
        assert '_lit_thresh' in df.columns and '_lit_rule' in df.columns, (
            f"{bdef['version']} needs attach_lit_baseline_for(df, bdef) first")
        _want = _lit_tag(bdef['seed_rule'], bdef.get('iqr_k', 1.5), bdef.get('k', 4.0),
                         bdef.get('min_lit', 5), bdef.get('sparse_admit', True))
        _got = df['_lit_rule'].iloc[0]
        assert _got == _want, (
            f"{bdef['version']} asks for the {_want!r} bar but the frame carries {_got!r}; "
            'it would silently report a different definition\'s events. '
            'Rebuild with attach_lit_baseline_for(df, bdef).')
        common = dict(fluor_col=bdef['fluor_col'], lit_gate=bdef['lit_gate'],
                      link_dt=bdef['link_dt'],
                      link_dy_cell_lengths=bdef['link_dy_cell_lengths'],
                      min_frames=bdef.get('min_frames', 1),
                      frac_min=bdef.get('frac_min', 0.0),
                      constitutive_genes=set(bdef['constitutive_genes']))
        scope = 'lit_fence_trench'
    elif lit_mode:
        common = dict(fluor_col=bdef['fluor_col'], lit_gate=bdef['lit_gate'],
                      link_dt=bdef['link_dt'],
                      link_dy_cell_lengths=bdef['link_dy_cell_lengths'],
                      min_cell_frames=bdef.get('min_cell_frames', 1),
                      constitutive_genes=set(bdef['constitutive_genes']))
        scope = 'lit_trench'
    elif fiji_mode:
        common = dict(fluor_col=bdef['fluor_col'], window=bdef['window'],
                      fiji_fold=bdef['fiji_fold'],
                      fiji_floor=bdef.get('fiji_floor', 0.0),
                      fiji_max_activity=bdef.get('fiji_max_activity', np.inf),
                      fiji_base_floor_frac=bdef.get('fiji_base_floor_frac', 0.05),
                      constitutive_genes=set(bdef['constitutive_genes']))
        scope = 'fiji_trench'
    _detect = (detect_bursts_seeded if seed_mode
               else detect_bursts_lit if lit_mode
               else detect_bursts_fiji if fiji_mode else detect_bursts)

    if sig.startswith('per-trench'):
        b = _detect(df, **common)
        b['def_version'] = bdef['version']
        b['baseline_scope'] = scope
        b['lineage_family'] = np.nan
        return b, pd.DataFrame()

    cells = add_lineage_family(df)
    part = 'lineage_family'
    if sig.startswith('per-lineage-family x spatial'):
        cells = add_spatial_subgroup(cells, bdef)
        part = 'spatial_group'

    keyed = cells.copy()
    keyed['trench_id'] = keyed['trench_id'].astype(str) + FAMILY_SEP + keyed[part].astype(str)
    b = _detect(keyed, **common)
    if b.empty:
        b['def_version'] = bdef['version']
        b['baseline_scope'] = scope
        return b, pd.DataFrame()

    split = b['trench_id'].str.rsplit(FAMILY_SEP, n=1, expand=True)
    b['trench_id'] = split[0]
    b['partition'] = split[1]
    b['lineage_family'] = b['partition'].astype(str).str.split('.').str[0].astype(int)

    # Position and size at the peak, for the merge. centroid_y is the true along-trench axis;
    # centroid_x is kymograph-space and must never be used for distance (see CLAUDE.md).
    pos = KYMO_KEY + ['track_id', 'time_frame']
    b = b.merge(cells[pos + ['centroid_y', 'axis_major_length']], on=pos, how='left')
    assert b['centroid_y'].notna().all(), 'a burst peak failed to match a cell row'
    b, audit = merge_concurrent_bursts(b, bdef)
    b['def_version'] = bdef['version']
    b['baseline_scope'] = scope
    return b.reset_index(drop=True), audit


def attach_kymograph_baselines(df, fluor_col=FLUOR_COL, scope='kymograph',
                               scale_floor_frac=0.25, med_col='_base_med',
                               scale_col='_base_scale'):
    """Attach a per-row baseline under an alternative SCOPE, for `detect_bursts`.

    The shipped rule centres and scales per (gene, experiment). That is what makes thresholds
    comparable across a 283x range of gene brightness -- but it is a POPULATION threshold
    applied to a per-trench phenomenon, and measured, only **109 of 255 trenches (43%)** have
    a brightest-ever cell-frame that reaches their own bar. The rest cannot register a burst
    at all, very unevenly: 87% of lacZ trenches are eligible against 25% of baeS and 0% of
    mazF, so the per-gene ranking is partly a statement about eligibility.

    Scopes, all measured at k = 4.0 on the 20-experiment frame:

    ==========================  =======  ========  =====================================
    scope                       events   hupA/mazF notes
    ==========================  =======  ========  =====================================
    ``'gene_experiment'``            89  0.00/0.00 the shipped rule
    ``'kymograph'``                 156  0.00/0.19 **breaks mazF** -- a quiet trench gets a
                                                   tiny MAD and its own noise clears the bar
    ``'kymograph_global_scale'``     89  0.00/0.00 local centre, global scale: no change
    ``'kymograph'`` + floor 0.25    153  0.00/0.00 coverage 43% -> 59%, control intact
    ==========================  =======  ========  =====================================

    So the floor is not cosmetic: it is what stops the local scale collapsing on a trench
    that never does anything. **None of these catch the two documented near-misses**, which
    fail on prominence and shape rather than on the height offset -- a local scale larger
    than the global one actually raises the prominence bar.

    Args:
        scope: ``'kymograph'`` (centre and scale both per trench),
            ``'kymograph_global_scale'`` (centre per trench, scale per gene x experiment), or
            ``'gene_experiment'`` (the shipped rule, attached as columns).
        scale_floor_frac: floor the local scale at this fraction of the (gene, experiment)
            scale. 0 disables the floor.

    Returns a copy of `df` with the two baseline columns added.
    """
    glob = compute_baselines(df, fluor_col=fluor_col)
    out = df.copy()
    gkey = list(zip(out['gene'], out['experiment_name']))
    g_med = np.array([glob['median'].get(k, np.nan) for k in gkey])
    g_scale = np.array([glob['mad_scale'].get(k, np.nan) for k in gkey])

    if scope == 'gene_experiment':
        out[med_col], out[scale_col] = g_med, g_scale
        return out

    def _robust(x):
        m = np.median(x)
        return pd.Series({'median': m,
                          'mad_scale': max(1.4826 * np.median(np.abs(x - m)), 1e-6)})

    local = (out.groupby(KYMO_KEY, observed=True)[fluor_col]
                .apply(lambda x: _robust(x.to_numpy())).unstack())
    idx = pd.MultiIndex.from_arrays([out[c] for c in KYMO_KEY])
    l_med = local['median'].reindex(idx).to_numpy()
    l_scale = local['mad_scale'].reindex(idx).to_numpy()

    out[med_col] = l_med
    if scope == 'kymograph_global_scale':
        out[scale_col] = g_scale
    elif scope == 'kymograph':
        out[scale_col] = (np.maximum(l_scale, scale_floor_frac * g_scale)
                          if scale_floor_frac else l_scale)
    else:
        raise ValueError(f'unknown scope {scope!r}')
    return out


def attach_kymograph_fence(df, fluor_col=FLUOR_COL, iqr_k=4.0, fence_col='_tukey_fence'):
    """Attach each row's own KYMOGRAPH's Tukey fence, `Q3 + iqr_k*IQR`, for `detect_bursts`.

    Default `iqr_k` is **4.0**, matching `BURST_DEF_V7` -- not the textbook 1.5, which on this
    data gives 523 events and breaks all three regression trenches. See V7's sweep table.

    The v7 route's statistic. Same shape as `attach_kymograph_baselines` -- a per-trench number
    broadcast onto every row of that trench -- because `detect_bursts` sees family x cluster
    subgroups, not trenches, and so cannot compute a per-trench quantity itself.

    **Why a quartile fence rather than another location+scale bar.** Every gate in v1-v6 is a
    statement about the (gene, experiment) median: a height, or a ratio against a peak's own
    flanks. Re-scoping that median to the trench was evaluated four times and shipped none of
    them (`bursts.md` section 8). A fence is a statement about the SHAPE of one trench's own
    distribution instead, and measured on cell-frames it reaches both documented near-misses
    that no height bar reaches at any scope, while ranking the constitutive controls
    hupA/mazF/chpS 8th/7th/6th of 10 by flagged fraction.

    **Units are a trap here.** The fence must be computed on the SAME column the detector runs
    on. Exposure is one constant per experiment, so raw and per-ms fences give identical
    verdicts -- but the fences themselves differ by 150-500x, so a fence built on
    `intensity_mean_fluor` and compared against `fluor_per_ms` is wrong by up to 500x and
    produces no error. The default matches `detect_bursts`' default; pass `bdef['fluor_col']`
    explicitly when driving it from a definition.

    No degeneracy guard is needed on this dataset: measured, 0 of 255 kymographs have a zero
    IQR, so the fence is always finite. `np.percentile`'s linear interpolation, and the caller
    compares with strict `>`, both matching `figures._quantile_stats` so v7 and the notebook's
    `kymo_threshold_table` cannot drift.

    Returns a copy of `df` with `fence_col` added.
    """
    out = df.copy()

    def _fence(x):
        q1, q3 = np.percentile(x, [25, 75])
        return q3 + iqr_k * (q3 - q1)

    local = out.groupby(KYMO_KEY, observed=True)[fluor_col].apply(
        lambda x: _fence(x.to_numpy(dtype=float)))
    idx = pd.MultiIndex.from_arrays([out[c] for c in KYMO_KEY])
    out[fence_col] = local.reindex(idx).to_numpy()
    assert out[fence_col].notna().all(), 'a row did not get a fence'
    return out


def burst_coverage(df, bdef=None, baseline_cols=None):
    """Per-trench eligibility: can this trench's brightest frame ever reach its height bar?

    A trench whose envelope never crosses the bar contributes only to the DENOMINATOR of
    every burst rate, so the rates mix "does not burst" with "could not have been seen to
    burst". `headroom_s` is how far the brightest frame sits above the bar in MAD-scales:
    negative means structurally ineligible.

    Returns one row per trench, with `n_families` so the trenches where v2/v3 degenerate to
    v1 (22% of them are a single family) are visible too.
    """
    bdef = BURST_DEF if bdef is None else bdef
    fluor_col = bdef['fluor_col']
    glob = None if baseline_cols else compute_baselines(df, fluor_col=fluor_col)
    rows = []
    for (exp, fov, trench), tr in df.groupby(KYMO_KEY, observed=True):
        gene = tr['gene'].iloc[0]
        if baseline_cols:
            med, scale = (tr[baseline_cols[0]].iloc[0], tr[baseline_cols[1]].iloc[0])
        else:
            if (gene, exp) not in glob.index:
                continue
            med, scale = glob.loc[(gene, exp), ['median', 'mad_scale']]
        if not np.isfinite(med) or not np.isfinite(scale):
            continue
        height = med + bdef['k'] * scale
        fams = lineage_family_roots(tr)
        tagged = tr.assign(_fam=[fams[int(t)] for t in tr['track_id']])
        env_max = max(trench_max_envelope(g, fluor_col=fluor_col)[fluor_col].max()
                      for _, g in tagged.groupby('_fam'))
        rows.append({'gene': gene, 'experiment_name': exp, 'FOV': fov,
                     'trench_id': trench, 'env_max': float(env_max),
                     'bar': float(height), 'eligible': bool(env_max >= height),
                     'headroom_s': float((env_max - height) / scale),
                     'n_families': len(set(fams.values()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Audit: every candidate the detector looked at, and why it was rejected
# ---------------------------------------------------------------------------------------

def burst_candidates(df, bdef=None, fluor_col=None):
    """Every height-clearing local maximum on the envelopes, with the reason for its verdict.

    `detect_bursts` returns only what passed. This returns the whole candidate set, so
    "underperforming for some events" becomes a countable claim rather than an impression.
    Measured on the 20-experiment frame under v3: 178 candidates, 94 pass, and **84 are
    rejected by prominence alone** -- 34 of them at >=75% of the required prominence.

    Prominence is computed three ways so the rule itself can be audited:

    ==============  =====================================================================
    ``prom_s``      the shipped rule: peak - max(left_base, right_base), in MAD-scales
    ``prom_min_s``  peak - min(left_base, right_base)
    ``prom_left_s`` peak - left_base, i.e. a causal "did it rise" reading
    ==============  =====================================================================

    Both alternatives look more permissive and are worse: measured, they score hupA's
    brightest flat-high wobble at 6.855 MAD-scales, *above* a real chpS burst at 6.061. Keep
    the max-base rule and move the bar (`k_prom`) instead.

    `frames_to_end` is carried because a peak near the end of a recording has a truncated
    right flank, so its prominence is measured against a plateau rather than a baseline.

    Returns one row per candidate, sorted by descending `prom_ratio`.
    """
    bdef = BURST_DEF if bdef is None else bdef
    fluor_col = bdef['fluor_col'] if fluor_col is None else fluor_col
    k, kp, window = bdef['k'], prom_multiplier(bdef), bdef['window']
    fold_bar = bdef.get('fold_prom')
    fold_left_bar = bdef.get('fold_left_prom')
    den_frac = bdef.get('fold_base_floor_frac', 0.5)
    floor_s = bdef.get('fold_height_floor_s', 1.0)
    baselines = compute_baselines(df, fluor_col=fluor_col)

    rows = []
    for (exp, fov, trench), tr in df.groupby(KYMO_KEY, observed=True):
        gene = tr['gene'].iloc[0]
        if (gene, exp) not in baselines.index:
            continue
        med, s = baselines.loc[(gene, exp), ['median', 'mad_scale']]
        height, req = med + k * s, kp * s
        tr = tr.sort_values('time_frame')
        fams = lineage_family_roots(tr)
        tagged = tr.assign(_fam=[fams[int(t)] for t in tr['track_id']])

        for fam, g in tagged.groupby('_fam'):
            env = trench_max_envelope(g, fluor_col=fluor_col)
            sig = env[fluor_col].to_numpy()
            if len(sig) < 3:
                continue
            # No height gate when a fold route is active: the whole point of that route is
            # peaks that never clear `height`, and an audit that filtered them out could not
            # show why they were or were not admitted.
            peaks, _ = (find_peaks(sig) if (fold_bar or fold_left_bar)
                        else find_peaks(sig, height=height))
            if not len(peaks):
                continue
            # scipy's own bases, so the audit and the detector agree by construction
            proms, lb, rb = peak_prominences(sig, peaks)
            for i, pi in enumerate(peaks):
                lo = max(0, pi - window)
                hi = min(len(sig), pi + window + 1)
                is_locmax = bool(sig[pi] >= sig[lo:hi].max())
                left, right = sig[lb[i]], sig[rb[i]]
                r = env.iloc[pi]
                base = max(left, right)
                fold = float(sig[pi] / base) if base > 0 else np.nan
                # v6's left-base ratio, on the SAME floored denominator the detector uses --
                # unfloored it is reported too, because that column is what shows why the
                # floor is needed (chpS 010/46 scores 30x on a left_base of 0.006).
                base_l = max(left, den_frac * med)
                fold_left_raw = float(sig[pi] / left) if left > 0 else np.nan
                fold_left = float(sig[pi] / base_l) if base_l > 0 else np.nan
                over_floor = bool(sig[pi] >= med + floor_s * s)
                ok_abs = is_locmax and sig[pi] > height and proms[i] >= req
                ok_fold = bool(fold_bar and is_locmax and np.isfinite(fold)
                               and fold >= fold_bar and over_floor)
                ok_fold_left = bool(fold_left_bar and is_locmax and np.isfinite(fold_left)
                                    and fold_left >= fold_left_bar and over_floor)
                passed = ok_abs or ok_fold or ok_fold_left
                rows.append({
                    'gene': gene, 'experiment_name': exp, 'FOV': fov, 'trench_id': trench,
                    'lineage_family': fam, 'time_frame': int(r['time_frame']),
                    'track_id': int(r['track_id']), 'fluor': float(sig[pi]),
                    'threshold': float(height), 'required_prominence': float(req),
                    'height_s': float((sig[pi] - med) / s),
                    'prom_s': float(proms[i] / s),
                    'prom_min_s': float((sig[pi] - min(left, right)) / s),
                    'prom_left_s': float((sig[pi] - left) / s),
                    'prom_ratio': float(proms[i] / req) if req > 0 else np.nan,
                    'fold': fold, 'fold_bar': fold_bar if fold_bar else np.nan,
                    'fold_left': fold_left, 'fold_left_unfloored': fold_left_raw,
                    'fold_left_bar': fold_left_bar if fold_left_bar else np.nan,
                    'route': ('absolute' if ok_abs else 'fold' if ok_fold
                              else 'fold_left' if ok_fold_left else ''),
                    'left_base': float(left), 'right_base': float(right),
                    'binding_side': 'right' if right > left else
                                    ('left' if left > right else 'tie'),
                    'n_frames': int(len(sig)), 'frames_to_end': int(len(sig) - 1 - pi),
                    'is_local_max': is_locmax,
                    # Which bar blocked it. With a ratio route active, 'height' alone is
                    # misleading -- a peak under `height` may still have been rejected on
                    # fold, so say 'height+fold' when a fold route was live and also failed.
                    'verdict': ('burst' if passed
                                else 'not local max' if not is_locmax
                                else ('height+fold' if (fold_bar or fold_left_bar)
                                      else 'height') if sig[pi] <= height
                                else 'prominence'),
                })
    cols_empty = ['gene', 'experiment_name', 'FOV', 'trench_id', 'time_frame', 'verdict']
    if not rows:
        return pd.DataFrame(columns=cols_empty)
    out = pd.DataFrame(rows).sort_values('prom_ratio', ascending=False)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Locating an event's cell along the trench
# ---------------------------------------------------------------------------------------

def add_relative_y(events, cells, track_col='track_id', frame_col='time_frame',
                   y_col='centroid_y'):
    """Add a scale-free along-trench position to any event table, so its cell is findable.

    Applied to **every** event table regardless of how the event was decided, because
    `centroid_y` alone does not identify a cell to a reader: it is raw camera pixels, and a
    trench occupies a different pixel band in every acquisition.

    Three columns, deliberately overlapping, because they fail differently:

    ==============================  =====================================================
    ``centroid_y_rel``              0 at the OPEN end, 1 at the closed end. Denominator is
                                    this trench's own occupied extent (see below), so it
                                    maps straight onto the viewer's y-axis as a fraction.
    ``cell_index_from_open_end``    0-based rank of this cell among the cells present in
                                    that frame, sorted by `centroid_y`. Denominator-free,
                                    so it survives a bad extent estimate.
    ``n_cells_in_frame``            how many cells that rank is out of -- "cell 3 of 8".
    ==============================  =====================================================

    **What the denominator is, precisely.** Per kymograph, over all cell-frames:
    `y_lo = min(centroid_y - L/2)` and `y_hi = max(centroid_y + L/2)`, using cell bodies
    rather than centroids so a trench whose end cell is long is not clipped. That is the
    **occupied extent**, not the image height -- nothing in the tables records image height,
    and the masks live on the microscopy share. In a packed mother-machine trench the two are
    close, but `centroid_y_rel` is a fraction of what cells occupied, so do not read it as a
    fraction of the trench's physical length.

    **Orientation is measured, not assumed.** Cells advect toward LOW `centroid_y` in this
    dataset -- in chpS 010/1188 the same cell tracks 270 -> 10 px over the movie -- so low y
    is the open end and high y is the closed (mother) end. `centroid_y_rel` near 0 therefore
    means "about to be washed out", near 1 means "at the mother".

    Args:
        events: any table with the trench key plus a track and a frame column.
        cells: the cell table the events were detected on, for the extent and the ranks.
        track_col, frame_col, y_col: the event table's names for those three. Sustained rises
            anchor on `peak_track_id` / `peak_frame` / `peak_centroid_y`.

    Returns a copy of `events` with the three columns added, plus `trench_span_px` so the
    denominator that produced `centroid_y_rel` is auditable rather than implicit.
    """
    out = events.copy()
    if out.empty:
        for c in ('centroid_y_rel', 'cell_index_from_open_end', 'n_cells_in_frame'):
            out[c] = np.nan
        return out

    half = cells['axis_major_length'].to_numpy() / 2.0
    extent = (cells.assign(_lo=cells['centroid_y'].to_numpy() - half,
                           _hi=cells['centroid_y'].to_numpy() + half)
                   .groupby(KYMO_KEY, observed=True)
                   .agg(y_lo=('_lo', 'min'), y_hi=('_hi', 'max')))

    # Rank within (trench, frame): 0 = nearest the open end.
    ranked = cells[KYMO_KEY + ['track_id', 'time_frame', 'centroid_y']].copy()
    grp = ranked.groupby(KYMO_KEY + ['time_frame'], observed=True)['centroid_y']
    ranked['cell_index_from_open_end'] = grp.rank(method='first').astype(int) - 1
    ranked['n_cells_in_frame'] = grp.transform('size').astype(int)
    ranked = ranked.rename(columns={'track_id': track_col, 'time_frame': frame_col})

    out = out.merge(extent, on=KYMO_KEY, how='left')
    out = out.merge(ranked.drop(columns='centroid_y'),
                    on=KYMO_KEY + [track_col, frame_col], how='left')

    span = out['y_hi'] - out['y_lo']
    if y_col not in out.columns:
        raise ValueError(f'{y_col!r} is not in the event table; pass y_col=')
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = (out[y_col] - out['y_lo']) / span.where(span > 0)
    out['centroid_y_rel'] = rel.clip(0.0, 1.0).round(4)
    out['trench_span_px'] = span.round(1)
    return out.drop(columns=['y_lo', 'y_hi'])


# ---------------------------------------------------------------------------------------
# A second event class: sustained rises
# ---------------------------------------------------------------------------------------

def detect_sustained_rises(df, w=8, z_min=3.0, fluor_col=FLUOR_COL, min_group=0,
                           return_all=False, verbose=True):
    """Trenches whose expression steps up to a NEW LEVEL rather than spiking.

    A burst is a peak; prominence is designed to reject a monotone ramp, and correctly so.
    But some trenches genuinely shift baseline -- `DUMM_giTG066_063_120925 002/1979` rises
    1.62x across the movie, with its envelope still climbing at the last frame. That is a
    real observation the burst table cannot represent, so it gets its own table.

    Statistic: ``log2(median(last w frames) / median(first w frames))`` per (trench, lineage
    family) envelope, robust-z'd **within (gene, experiment)**. The z is the whole point --
    an absolute fold bar does not work, because constitutive genes drift too: measured, a
    fold >= 1.5 rule ranks hupA **1st of 10**. Scoring against how much this gene's own
    trenches normally drift puts hupA and mazF back at 0.00 for every z >= 2.5.

    `peak_over_last` = ``env.max() / median(last w)`` separates a ramp from a burst that
    happens to end high: at z >= 3.0, 1 of 3 hits is a trench that already bursts, and that
    column is what identifies it. Filter on it rather than assuming the two classes are
    disjoint.

    **Read the z, not the fold.** baeS trenches span 0.56x-1.9x (10th-90th percentile of
    per-trench fold), so trench 1979's 2.62x is only z = 1.42 -- real, but not separable from
    that gene's ordinary trench-to-trench spread on this dataset. A bar low enough to include
    it flags ~10% of all trenches.

    Args:
        w: frames per comparison window. A family needs >= 3*w frames to be judged.
        z_min: robust-z bar. Only positive rises are returned.
        min_group: skip (gene, experiment) groups with fewer than this many envelopes, whose
            MAD is not estimable. 0 keeps everything (what the reported numbers used). Check
            the returned `group_n` before trusting any single z.
        return_all: return EVERY scored envelope instead of the hits -- rises and falls
            alike, no z bar, no one-per-trench collapse. This is the frame to plot the drift
            distribution from. The hits frame is filtered to `logfold > 0` and to one family
            per trench, so its spread is not the null distribution and must not be quoted as
            one.

    Returns:
        DataFrame. With `return_all=False`, one row per trench (its strongest family), sorted
        by descending z. With `return_all=True`, one row per (trench, family) envelope.
    """
    rows = []
    for (exp, fov, trench), tr in df.groupby(KYMO_KEY, observed=True):
        gene = tr['gene'].iloc[0]
        tr = tr.sort_values('time_frame')
        fams = lineage_family_roots(tr)
        tagged = tr.assign(_fam=[fams[int(t)] for t in tr['track_id']])
        for fam, g in tagged.groupby('_fam'):
            env = trench_max_envelope(g, fluor_col=fluor_col)
            sig = env[fluor_col].to_numpy()
            if len(sig) < 3 * w:
                continue
            first, last = np.median(sig[:w]), np.median(sig[-w:])
            if first <= 0 or last <= 0:
                continue
            imax = int(np.argmax(sig))
            rows.append({
                'gene': gene, 'experiment_name': exp, 'FOV': fov, 'trench_id': trench,
                'lineage_family': fam, 'n_frames': int(len(sig)),
                'first': float(first), 'last': float(last),
                'fold': float(last / first), 'logfold': float(np.log2(last / first)),
                'env_max': float(sig[imax]),
                'peak_frame': int(env['time_frame'].iloc[imax]),
                'peak_track_id': int(env['track_id'].iloc[imax]),
                'peak_centroid_y': float(env['centroid_y'].iloc[imax]),
                'peak_over_last': float(sig[imax] / last),
            })

    cols_empty = ['gene', 'experiment_name', 'FOV', 'trench_id', 'fold', 'z']
    if not rows:
        return pd.DataFrame(columns=cols_empty)
    d = pd.DataFrame(rows)

    grp = d.groupby(['gene', 'experiment_name'])['logfold']
    sizes = grp.transform('size')

    def _z(x):
        m = x.median()
        return (x - m) / max(1.4826 * (x - m).abs().median(), 1e-6)

    d['z'] = grp.transform(_z)
    d['group_n'] = sizes
    if min_group:
        d = d[d['group_n'] >= min_group]
    if return_all:
        return d.sort_values('z', ascending=False).reset_index(drop=True)

    hits = d[(d['z'] >= z_min) & (d['logfold'] > 0)]
    # One row per trench: the family with the strongest rise.
    hits = (hits.sort_values('z', ascending=False)
                .drop_duplicates(KYMO_KEY).reset_index(drop=True))
    # Anchored on the peak cell, so a rise is locatable in the viewer like a burst is.
    hits = add_relative_y(hits, df, track_col='peak_track_id', frame_col='peak_frame',
                          y_col='peak_centroid_y')
    if verbose:
        print(f'sustained rises: {len(d)} envelopes with >= {3 * w} frames across '
              f'{d.groupby(KYMO_KEY).ngroups} trenches -> {len(hits)} at z >= {z_min}')
    return hits


# ---------------------------------------------------------------------------------------
# Export for the kymograph viewer
# ---------------------------------------------------------------------------------------

GEO_COLS = ['gene', 'centroid_x', 'centroid_y', 'axis_major_length',
            'axis_minor_length', 'area']


def build_burst_csv(burst_sets, cells, path, raw_csv=None):
    """One row per burst PEAK across the union of all definitions, one boolean per definition.

    The viewer joins on (experiment_name, FOV, trench_id, track_id, time_frame) and cuts the
    outline contour from `ctc_masks[time_frame] == track_id`, so those five columns are the
    load-bearing ones; the geometry columns let the CSV be validated without opening a mask.

    `time_frame` is the 0-based index within the trimmed stack, i.e. the kymograph COLUMN
    index. It must NOT be offset by the time_dict `start` -- the viewer uses `start` only for
    the printed tick labels, so adding it would slide every outline sideways. (That offset is
    also why a frame number read off the viewer does not equal `time_frame` for any trench
    with a non-zero `start`.)

    Every row also gets `centroid_y_rel`, `cell_index_from_open_end` and `n_cells_in_frame`
    from `add_relative_y`, so a burst can be tied to a cell without opening a mask.

    Args:
        burst_sets: {version: bursts DataFrame}. One boolean column per key, so passing only
            the definitions under comparison keeps the CSV to those columns. The viewer reads
            whatever `burst_*` columns it finds and defaults the rest to False.
        cells: the cell table detection ran on, for the geometry join and the position columns.
        raw_csv: path to the un-annotated export, read only for columns `cells` lacks
            (`centroid_x`, `axis_minor_length`), for just these few rows.
    """
    key = KYMO_KEY + ['track_id', 'time_frame']
    frames = []
    for v, b in burst_sets.items():
        f = b[key].astype({'track_id': int, 'time_frame': int}).assign(**{f'burst_{v}': True})
        frames.append(f)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=key, how='outer')
    for v in burst_sets:
        out[f'burst_{v}'] = out[f'burst_{v}'].eq(True)   # NaN -> False, no object downcast

    have = [c for c in GEO_COLS if c in cells.columns]
    out = out.merge(cells[key + have], on=key, how='left')

    missing = [c for c in GEO_COLS if c not in have]
    if missing:
        if raw_csv is None:
            raise ValueError(f'{missing} are absent from `cells`; pass raw_csv to fetch them')
        want = set(key + missing)
        extra = pd.read_csv(raw_csv, usecols=lambda c: c in want,
                            dtype={'FOV': str, 'trench_id': str,
                                   'experiment_name': str, 'gene': str})
        extra['FOV'] = extra['FOV'].str.zfill(3)
        out = out.merge(extra, on=key, how='left')

    # fluor / threshold / route / baseline_scope from whichever definition detected the
    # peak. Filled in `burst_sets` order and only where still null, so the FIRST version that
    # found a peak wins -- fluor and threshold are identical across versions run under the
    # same baseline, since baselines do not depend on the partition, and any peak an older
    # definition found cleared the height bar, so every version that sees it agrees the route
    # is 'absolute'.
    #
    # `baseline_scope` is what makes this safe when the sets were run under DIFFERENT
    # baselines: a row found only by a per-kymograph run carries that trench's local bar in
    # `threshold`, and the scope column says so instead of leaving the number ambiguous.
    # Order `burst_sets` with the gene-scoped definitions first so shared rows report the
    # gene bar.
    # The v9 extras ride along only when EVERY set has them, which the all() already
    # guarantees -- so a v9-only export keeps them and a mixed v4-v7 export is unchanged.
    carry = [c for c in ('fluor', 'threshold', 'route', 'baseline_scope',
                         'n_cell_frames', 'n_frames', 'peak_fluor', 'peak_time_frame',
                         'lit_frac', 'seed_first', 'seed_last', 'n_lit_cell_frames',
                         'lit_first', 'lit_last', 'peak_frac', 'peak_pct')
             if all(c in b.columns for b in burst_sets.values())]
    for i, v in enumerate(burst_sets):
        src = (burst_sets[v][key + carry]
               .astype({'track_id': int, 'time_frame': int}))
        out = out.merge(src, on=key, how='left', suffixes=('', f'_{v}'))
        if i:
            for c in carry:
                out[c] = out[c].fillna(out.pop(f'{c}_{v}'))

    assert not out['trench_id'].astype(str).str.contains(FAMILY_SEP).any(), (
        f'a partition id leaked into trench_id via {FAMILY_SEP!r}; the viewer would match '
        'no card')
    assert out['FOV'].astype(str).str.len().eq(3).all(), 'FOV must stay zero-padded to 3'
    for c in ('centroid_x', 'centroid_y', 'axis_major_length', 'area'):
        assert out[c].notna().all(), f'a peak row has no {c}'
    assert len(out) == len(out.drop_duplicates(key)), 'duplicate peak rows'

    # Unconditional, not opt-in: the point of these columns is that EVERY burst carries a
    # locatable position no matter which definition found it, so the viewer and any future
    # definition get them for free.
    out = add_relative_y(out, cells)
    assert out['cell_index_from_open_end'].notna().all(), (
        'a peak row did not match a cell-frame, so it has no within-frame rank')

    cols = (key + [f'burst_{v}' for v in burst_sets] + GEO_COLS
            + ['centroid_y_rel', 'cell_index_from_open_end', 'n_cells_in_frame',
               'trench_span_px'] + carry)
    out = out[cols].sort_values(KYMO_KEY + ['time_frame', 'track_id']).reset_index(drop=True)
    out.to_csv(path, index=False)
    print(f'wrote {path}  ({len(out)} rows)')
    print('rows per definition: '
          + ' | '.join(f'{v}={int(out[f"burst_{v}"].sum())}' for v in burst_sets))
    if 'route' in carry:
        print('by route: '
              + ' | '.join(f'{r}={n}' for r, n in out['route'].value_counts().items()))
    return out
