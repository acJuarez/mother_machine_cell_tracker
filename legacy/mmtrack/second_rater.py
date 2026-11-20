"""The second labeller's labels. Used only by legacy/compare_raters.py.

Same columns as the first labeller plus `confidence` / `comment`, which are ignored.
Covers araB, lacZ and chpS only. See legacy/second_rater.md for the findings.
"""
import os

import pandas as pd

from mmtrack.bursts import load_hand_labels


# A second labeller's file: same columns plus `confidence` / `comment`, which are ignored.
# Covers araB, lacZ and chpS only. `frame` is the viewer frame, like the first labeller's.
# A row with track_id `N/A` is a whole-trench note, not a label: "Skipped" means the trench was
# not judged (drop it from both labellers' scope); anything else means reviewed, no burst.
SECOND_RATER = 'data/subgen_kymograph_heena.csv'
SECOND_RATER_GENES = ('araB', 'lacZ', 'chpS')

def load_second_rater(cells, path=SECOND_RATER, data_root='', verbose=True):
    """Returns (labels, rejected, reviewed_empty, skipped); the last two are KYMO_KEY tuple sets."""
    d = pd.read_csv(os.path.join(data_root, path), dtype=str, keep_default_na=False)
    canon = {g.lower(): g for g in SECOND_RATER_GENES}
    d['gene'] = d['gene'].str.strip().map(lambda g: canon.get(g.lower(), g))
    note = d['track_id'].str.strip().str.upper().eq('N/A')
    key = lambda r: (r.experiment_name, r.FOV.strip().zfill(3), r.trench_id.strip())
    is_skip = d['confidence'].str.strip().str.lower().eq('skipped')
    skipped = {key(r) for r in d[note & is_skip].itertuples()}
    reviewed_empty = {key(r) for r in d[note & ~is_skip].itertuples()}
    d = d.loc[~note, ['track_id', 'frame', 'experiment_name', 'FOV', 'trench_id', 'gene']]
    lab, rej = load_hand_labels(d, cells=cells, genes=SECOND_RATER_GENES, verbose=verbose)
    return lab, rej, reviewed_empty, skipped
