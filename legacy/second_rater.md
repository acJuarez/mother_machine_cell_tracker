# The second labeller

Moved out of `bursts.md` with `compare_raters.py` and `bursts.load_second_rater`.
The study is finished; nothing in the live pipeline reads it.

`compare_raters.py` reproduces every number here and asserts the counts. It needs the
repo root on `sys.path` and `data/subgen_kymograph_heena.csv`.

**A second labeller** (`bursts.SECOND_RATER`, araB / lacZ / chpS only; `compare_raters.py`
reproduces every number here and asserts the counts). On 92 trenches, A has 257 labels and the
second labeller has 265.
- **Counts agree, so the rates are labeller-robust.** Per-trench Spearman 0.82, Wilcoxon p = 0.81,
  bursts per division within each other's Poisson CI for all three genes.
- **Timing does not agree.** A marks the **peak** (52% exactly at the local max). The second
  labeller marks the **rise** (24% at the peak, median 1 frame early, at 83% of peak brightness).
  Raw F1 between the two is 0.41 at ±1. Snapped to the peak it is **0.71 [0.67, 0.76]**, which
  is the human ceiling. That leaves 168 shared events, 78 marked only by A and 57 only by the
  second labeller.
- **Certified-empty depends on the labeller.** Empty-vs-not κ = 0.63: 14 empty trenches by A, 23
  by the second labeller, mostly araB.
- **v2 encodes labeller A.** v2 agrees with A at 0.77, better than a second human does (+0.06
  [+0.004, +0.11]). It recalls 85% of A's labels but 60% of the second labeller's.
- **v2 > v3 holds only against A's raw frames** (+0.088 at ±1). Against the second labeller,
  either set snapped to the peak, the consensus, or the union, v2 − v3 is within ±0.04 of zero.
  **Quote v2 against v3 only on a labeller-neutral reference**, and do not chase F1 against A
  above ~0.7.
- `data/rater_events_102426.csv` lists every peak-snapped event with `cls` ∈ {both, A_only,
  H_only}, for adjudication.
