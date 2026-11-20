"""Stage 6: the per-experiment trackastra pickles become the tables the analysis reads.

Six outputs, built in this order:

    all_data_<gen>.csv                one cell at one frame, + gene / exposure / time
    track_pairs_<gen>.csv             one cell pair, within_track or division, + is_jump
    fluor_filtered_<gen>.csv          per-cell mean on median-filtered images
    fiji_fluor_<gen>.csv              per-cell values on the viewer's FIJI transform
    all_data_annotated_<gen>.csv      + fluor_filtered, is_degenerate_mask, per-kymograph scale
    track_pairs_annotated_<gen>.csv   + both endpoint ids and every endpoint's value
    divisions_<gen>.csv               one observed division, + doubling_min and has_jump

The two `_annotated_` tables and `divisions_` carry all experiments; the failed
acquisitions are tagged `is_failed_experiment`, never dropped.

MERGE KEY. On a division row `track_id` is the DAUGHTER while `frame_from` is the PARENT's
last frame, and `time_frame` is NaN. Join a pair back onto the cell table per endpoint, on
`(experiment_name, FOV, trench_id, from_track_id, frame_from)` and the `to_` equivalent.
Merging on `time_frame`, or keying both endpoints on `track_id`, drops every division pair
without raising.

DTYPES. Read every one of these back with `experiment_name`, `FOV`, `trench_id`, `gene` and
`predicted_lineage` forced to str, or `'003'` becomes 3 and every join matches nothing.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from mmtrack.experiment_metadata import (
    EXPOSURE_MS,
    FAILED_EXPERIMENTS,
    MINUTES_INTERVAL,
    STRAIN_EXP_DICT,
)

TRENCH_KEYS = ['experiment_name', 'FOV', 'trench_id']
LINEAGE_KEYS = TRENCH_KEYS + ['track_id']
CELL_KEY = TRENCH_KEYS + ['track_id', 'time_frame']

# One kymograph. Same partition as TRENCH_KEYS -- gene is constant within a trench -- but
# carrying gene makes the group key self-describing in the exported file.
KYMO_GROUP = ['experiment_name', 'gene', 'FOV', 'trench_id']

# NaN on within_track rows.
DIVISION_COLS = ['parent_track_id', 'sibling_track_id', 'n_siblings']

RAW_FLUOR_COL = 'intensity_mean_fluor'
EXPOSURE_COL = 'fluor_exposure_ms'

# The four fluorescence units the annotated tables carry a per-kymograph min/max for.
FLUOR_BASES = ['intensity_mean_fluor',   # raw a.u.
               'fluor_per_ms',           # raw / fluor_exposure_ms
               'fluor_filtered',         # re-measured on median-filtered images, a.u.
               'fluor_filtered_per_ms']  # filtered / fluor_exposure_ms

DTYPES = {'experiment_name': str, 'FOV': str, 'trench_id': str,
          'gene': str, 'predicted_lineage': str}

# A mask below this fraction of its experiment's own median area is degenerate. Relative,
# because one acquisition sits at about a third the linear pixel scale of every other.
MIN_AREA_FRAC = 0.05

PKL_PREFIX = 'trackastra_cell_data_'
DEFAULT_BASE_DIR = '/Volumes/mcovert/Instruments/Covert-lab-scope1/subgen_processed_data'
DEFAULT_TRACKASTRA_DIR = '/Volumes/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output'


# ---------------------------------------------------------------------------
# 6a. Combine the per-experiment pickles
# ---------------------------------------------------------------------------

def load_pickles(base_path, time_dict_path):
    """Concatenate the per-experiment pickles that belong to the current generation.

    Pickles for experiments absent from the time_dict are skipped rather than folded in:
    reprocessing a trench can renumber its peak ids, so a leftover pickle would contribute
    rows keyed to peak ids that no longer exist.
    """
    known = set(json.load(open(time_dict_path)))
    paths = sorted(glob.glob(os.path.join(base_path, f'{PKL_PREFIX}*.pkl')))
    if not paths:
        raise SystemExit(f'No {PKL_PREFIX}*.pkl found in {base_path}. Run stage 5 first.')

    keep, skipped = [], []
    for path in paths:
        exp = os.path.basename(path)[len(PKL_PREFIX):-len('.pkl')]
        (keep if exp in known else skipped).append((exp, path))

    print(f'Found {len(paths)} pickle(s); using {len(keep)}.')
    if skipped:
        print(f'SKIPPING {len(skipped)} pickle(s) not present in {time_dict_path}:')
        for exp, _ in skipped:
            print(f'    {exp}')
    if not keep:
        raise SystemExit('Every pickle was skipped; check --time-dict.')

    return pd.concat((pd.read_pickle(p) for _, p in keep), ignore_index=True)


def annotate_metadata(df):
    """Add gene, exposure and time columns in place, and return the DataFrame."""
    # FOV is a zero-padded string in the pickles but a CSV round-trip turns '005' into 5.
    # Normalize both sides so the match holds whichever source the frame came from.
    fov = df['FOV'].astype(str).str.zfill(3)

    df['gene'] = None
    for gene, experiments in STRAIN_EXP_DICT.items():
        for exp, fovs in experiments.items():
            padded = [str(f).zfill(3) for f in fovs]
            df.loc[(df['experiment_name'] == exp) & fov.isin(padded), 'gene'] = gene

    exposures = df['experiment_name'].map(EXPOSURE_MS)
    df['phase_exposure_ms'] = [e[0] if isinstance(e, (list, tuple)) else None for e in exposures]
    df['fluor_exposure_ms'] = [e[1] if isinstance(e, (list, tuple)) else None for e in exposures]

    df['minutes_per_frame'] = df['experiment_name'].map(MINUTES_INTERVAL)
    df['time_min'] = df['time_frame'] * df['minutes_per_frame']
    return df


def report_gaps(df):
    """Print any rows missing an annotation. Returns True if all rows are annotated."""
    clean = True
    for col in ('gene', 'phase_exposure_ms', 'minutes_per_frame'):
        missing = df[df[col].isna()]
        if missing.empty:
            continue
        clean = False
        print(f'WARNING: {len(missing):,} rows missing {col}; '
              f'add the entry to mmtrack/experiment_metadata.py:')
        print(missing[['experiment_name', 'FOV']].drop_duplicates().to_string(index=False))
    return clean


def build_cells(base_path, time_dict_path):
    """The combined, metadata-annotated cell table."""
    df = annotate_metadata(load_pickles(base_path, time_dict_path))
    report_gaps(df)
    return df


def load_cell_data(csv_path, gene=None, usecols=None):
    """Read a cell CSV with the key dtypes forced and FOV re-padded."""
    if usecols is None:
        df = pd.read_csv(csv_path, usecols=lambda c: c != 'coords', dtype=DTYPES)
    else:
        df = pd.read_csv(csv_path, usecols=lambda c: c in usecols, dtype=DTYPES)
    df['FOV'] = df['FOV'].str.zfill(3)
    if gene is not None:
        df = df[df['gene'] == gene].copy()
    return df


# ---------------------------------------------------------------------------
# 6b. Pairs: consecutive frames of one track, and parent -> each daughter
# ---------------------------------------------------------------------------

def compute_consecutive_pairs(df, y_col='centroid_y', fluor_col=RAW_FLUOR_COL,
                              time_col='time_frame', length_col='axis_major_length'):
    """One row per consecutive-frame transition within a track."""
    frames = []
    for keys, g in df.groupby(LINEAGE_KEYS, sort=False):
        g = g.sort_values(time_col)
        exp, fov, trench, track = keys

        cur = g.iloc[:-1]              # every row except the last -> "from" member
        nxt = g.iloc[1:]               # every row except the first -> "to" member
        if len(cur) == 0:
            continue

        pair = pd.DataFrame({
            'experiment_name': exp,
            'FOV': fov,
            'trench_id': trench,
            'track_id': track,
            # Both endpoints are the same cell here; on a division pair they differ, which
            # is why these columns exist. See the module docstring's MERGE KEY note.
            'from_track_id': track,
            'to_track_id': track,
            'gene': g['gene'].iloc[0] if 'gene' in g.columns else None,
            'predicted_lineage': cur['predicted_lineage'].to_numpy(),
            'frame_from': cur[time_col].to_numpy(),
            'frame_to': nxt[time_col].to_numpy(),
            'node_id_from': cur['node_id'].to_numpy(),
            'node_id_to': nxt['node_id'].to_numpy(),
            'y_from': cur[y_col].to_numpy(),
            'y_to': nxt[y_col].to_numpy(),
            'fluor_from': cur[fluor_col].to_numpy(),
            'fluor_to': nxt[fluor_col].to_numpy(),
        })
        # The interval varies (5 vs 10 min across experiments), so carry it and the
        # wall-clock times rather than assuming frames are comparable.
        for col, src in (('minutes_per_frame', 'minutes_per_frame'),
                         ('time_min_from', 'time_min')):
            if src in g.columns:
                pair[col] = cur[src].to_numpy()
        if 'time_min' in g.columns:
            pair['time_min_to'] = nxt['time_min'].to_numpy()

        # The mean length is the natural scale for "did this track land on a neighbour":
        # it approximates the centroid spacing of two adjacent stacked cells, and averaging
        # both endpoints keeps it symmetric in time.
        if length_col in g.columns:
            pair['length_from'] = cur[length_col].to_numpy()
            pair['length_to'] = nxt[length_col].to_numpy()
            pair['cell_length'] = (pair['length_from'] + pair['length_to']) / 2

        pair['frame_gap'] = pair['frame_to'] - pair['frame_from']
        pair['delta_y'] = pair['y_to'] - pair['y_from']
        pair['delta_fluor'] = pair['fluor_to'] - pair['fluor_from']

        # Column-aligned with compute_division_pairs so the two concat without introducing
        # NaN-only columns on one side.
        pair['pair_type'] = 'within_track'
        pair['is_division'] = False
        pair['bad_frame_gap'] = pair['frame_gap'].ne(1)
        for col in DIVISION_COLS:
            pair[col] = np.nan
        frames.append(pair)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_division_pairs(df, y_col='centroid_y', fluor_col=RAW_FLUOR_COL,
                           time_col='time_frame', length_col='axis_major_length'):
    """Pair a parent's LAST frame with each daughter's FIRST frame.

    A division is the one transition compute_consecutive_pairs cannot see, because it
    groups by track_id and the daughters carry new ones. Each daughter names its parent in
    `parent_track_id`, joined on the trench keys as well, since raw track_ids repeat across
    trenches.

    `cell_length` is the PARENT's own length, not the two-endpoint mean: the question is
    whether the daughter landed where a daughter of this cell should.

    Returns one row per (parent, daughter) link, column-aligned with the within-track
    builder, with `time_frame` left NaN.
    """
    if 'parent_track_id' not in df.columns:
        print("    NOTE: no 'parent_track_id' column; skipping division pairs.")
        return pd.DataFrame()

    ordered = df.sort_values(TRENCH_KEYS + ['track_id', time_col])
    by_track = ordered.groupby(TRENCH_KEYS + ['track_id'], sort=False)
    first = by_track.head(1)   # each track's first observation -> the daughter endpoint
    last = by_track.tail(1)    # each track's last observation  -> the parent endpoint

    daughters = first[first['parent_track_id'].notna()].copy()
    if daughters.empty:
        print("    NOTE: no tracks with a parent_track_id; skipping division pairs.")
        return pd.DataFrame()
    # float NaN forces the column to float64; the join key must match track_id's dtype.
    daughters['parent_track_id'] = daughters['parent_track_id'].astype(
        ordered['track_id'].dtype)

    # Key dtypes must match load_cell_data's (FOV zero-padded str, trench_id str) or every
    # lookup silently returns NaN.
    parent_idx = pd.MultiIndex.from_arrays([
        daughters['experiment_name'], daughters['FOV'], daughters['trench_id'],
        daughters['parent_track_id'],
    ])
    parent = last.set_index(TRENCH_KEYS + ['track_id'])

    def _parent(col):
        if col not in parent.columns:
            return None
        return parent[col].reindex(parent_idx).to_numpy()

    pair = pd.DataFrame({
        'experiment_name': daughters['experiment_name'].to_numpy(),
        'FOV': daughters['FOV'].to_numpy(),
        'trench_id': daughters['trench_id'].to_numpy(),
        'track_id': daughters['track_id'].to_numpy(),   # the DAUGHTER's track
        # The from-endpoint is the PARENT at its last frame, so it keys on parent_track_id.
        # Both are int64 here and neither is ever NaN, so unlike parent_track_id itself they
        # survive the concat as integers.
        'from_track_id': daughters['parent_track_id'].to_numpy(),
        'to_track_id': daughters['track_id'].to_numpy(),
        'gene': (daughters['gene'].to_numpy() if 'gene' in daughters.columns else None),
        'predicted_lineage': daughters['predicted_lineage'].to_numpy(),
        'frame_from': _parent(time_col),
        'frame_to': daughters[time_col].to_numpy(),
        'node_id_from': _parent('node_id'),
        'node_id_to': daughters['node_id'].to_numpy(),
        'y_from': _parent(y_col),
        'y_to': daughters[y_col].to_numpy(),
        'fluor_from': _parent(fluor_col),
        'fluor_to': daughters[fluor_col].to_numpy(),
    })
    if 'minutes_per_frame' in daughters.columns:
        pair['minutes_per_frame'] = daughters['minutes_per_frame'].to_numpy()
    if 'time_min' in daughters.columns:
        pair['time_min_from'] = _parent('time_min')
        pair['time_min_to'] = daughters['time_min'].to_numpy()
    if length_col in daughters.columns:
        pair['length_from'] = _parent(length_col)
        pair['length_to'] = daughters[length_col].to_numpy()
        pair['cell_length'] = pair['length_from']

    pair['frame_gap'] = pair['frame_to'] - pair['frame_from']
    pair['delta_y'] = pair['y_to'] - pair['y_from']
    pair['delta_fluor'] = pair['fluor_to'] - pair['fluor_from']

    pair['pair_type'] = 'division'
    pair['is_division'] = True
    pair['parent_track_id'] = daughters['parent_track_id'].to_numpy()
    # A normal division yields exactly 2. n_siblings != 2 means the lineage graph gave this
    # parent a different number of children, which is worth auditing on its own.
    sib = daughters.groupby(TRENCH_KEYS + ['parent_track_id'], sort=False)['track_id']
    pair['n_siblings'] = sib.transform('size').to_numpy()
    pair['sibling_track_id'] = (
        sib.transform('sum').to_numpy() - pair['track_id'].to_numpy()
    )
    # The "sum minus self" trick names a single sibling only on a clean 2-way split.
    pair.loc[pair['n_siblings'] != 2, 'sibling_track_id'] = np.nan

    orphans = int(pair['frame_from'].isna().sum())
    if orphans:
        print(f"    NOTE: {orphans} daughter track(s) name a parent_track_id absent from "
              f"their trench; those pairs carry NaN delta_y and drop out downstream.")
    odd = pair.loc[pair['frame_gap'].notna() & (pair['frame_gap'] != 1), 'frame_gap']
    if len(odd):
        counts = odd.value_counts().sort_index().to_dict()
        print(f"    WARNING: {len(odd)} division link(s) with frame_gap != 1 {counts}. A "
              f"gap <= 0 means the 'parent' is alive in the same frame as its 'daughter', "
              f"which is a lineage-graph defect, not motion. See `bad_frame_gap`.")
    pair['bad_frame_gap'] = pair['frame_gap'].ne(1)

    return pair.reset_index(drop=True)


def flag_jumps(pairs, length_k=1.0, division_length_k=1.5):
    """Flag pairs whose delta_y is an outlier, i.e. likely a tracking mistake.

    A pair is flagged when |delta_y| > threshold * cell_length. A tracking error means the
    track landed on a different cell, which in a trench of stacked cells takes roughly one
    cell length of apparent motion. Normalising by cell length makes the criterion
    independent of magnification, cell size and frame interval.

    The threshold is per-row: `division_length_k` on division rows, `length_k` elsewhere,
    because a division moves the centroid further than one frame of growth. The applied
    value is recorded in `dy_threshold_cell_lengths`.

    Adds `delta_y_cell_lengths` and `is_jump`. Returns (pairs, lower, upper); the bounds
    are the WITHIN-TRACK ones.
    """
    pairs = pairs.copy()

    if 'cell_length' not in pairs.columns:
        raise ValueError(
            "flag_jumps needs a 'cell_length' column; the pair builders only add it when "
            "the input carries axis_major_length."
        )
    # A zero-area mask is a segmentation failure, not evidence of a hop, and would divide
    # toward infinity. Substitute the dataset median length.
    length = pairs['cell_length']
    degenerate = ~(length > 1)
    if degenerate.any():
        fallback = float(length[~degenerate].median())
        print(f"    NOTE: {int(degenerate.sum())} pair(s) have a degenerate "
              f"cell_length (<=1 px); using the median {fallback:.1f} px for those.")
        length = length.mask(degenerate, fallback)

    pairs['delta_y_cell_lengths'] = pairs['delta_y'] / length
    is_div = (pairs['is_division'].eq(True) if 'is_division' in pairs.columns
              else pd.Series(False, index=pairs.index))
    pairs['dy_threshold_cell_lengths'] = np.where(
        is_div, float(division_length_k), float(length_k))
    pairs['is_jump'] = (pairs['delta_y_cell_lengths'].abs()
                        > pairs['dy_threshold_cell_lengths'])
    return pairs, -float(length_k), float(length_k)


def build_pairs(cells, length_k=1.0, division_length_k=1.5):
    """Both pair types for every gene, concatenated and jump-flagged."""
    within = compute_consecutive_pairs(cells)
    divisions = compute_division_pairs(cells)
    parts = [p for p in (within, divisions) if len(p)]
    if not parts:
        raise SystemExit('no pairs could be built from this cell table')
    pairs = pd.concat(parts, ignore_index=True)
    pairs, _lo, _hi = flag_jumps(pairs, length_k=length_k,
                                 division_length_k=division_length_k)
    n_div = int(pairs['is_division'].sum())
    print(f'{len(pairs):,} pairs | {len(pairs) - n_div:,} within_track | {n_div:,} division '
          f'| {int(pairs["is_jump"].sum()):,} flagged as jumps')
    return pairs


# ---------------------------------------------------------------------------
# 6c. Fluorescence re-measured on processed images (needs the raw TIFFs)
# ---------------------------------------------------------------------------

def _iter_trenches(experiments, trackastra_dir, base_dir, time_dict, fluor_c):
    """Yield (exp, fov, peak, masks, raw) per trench, both already trimmed to n frames."""
    import tifffile

    if not os.path.isdir(trackastra_dir):
        raise FileNotFoundError(
            f'trackastra_dir does not exist: {trackastra_dir}\n'
            f'If it is on a network volume, check that the volume is mounted.')

    for exp in experiments:
        mask_paths = sorted(glob.glob(os.path.join(trackastra_dir, f'{exp}_ctc_masks_*_*.npy')))
        if not mask_paths:
            print(f'    WARNING: no ctc_masks found for {exp} in {trackastra_dir}.')
        for mask_path in mask_paths:
            name = os.path.basename(mask_path)[:-4]
            fov_str, peak_str = name.split('_ctc_masks_', 1)[1].rsplit('_', 1)

            fluor_path = os.path.join(
                base_dir, exp, 'hyperstacked', 'drift_corrected', 'rotated',
                'mm_channels', 'subtracted',
                f'subtracted_FOV_{fov_str}_region_{peak_str}_c_{fluor_c}.tif',
            )
            if not os.path.exists(fluor_path):
                print(f'    WARNING: missing fluor TIFF for {exp}/{fov_str}/{peak_str}; skipping.')
                continue

            masks = np.load(mask_path)
            # The tracker trimmed the stack to [start:end], so the fluor TIFF has to be cut
            # the same way or every value lands on the wrong frame.
            start = time_dict.get(exp, {}).get(fov_str, {}).get(peak_str, {}).get('start', 0)
            T = masks.shape[0]
            raw = tifffile.imread(fluor_path)[start:start + T]
            n = min(T, raw.shape[0])
            if n < T:
                print(f'    WARNING: {exp}/{fov_str}/{peak_str}: fluor gives {n} of {T} frames.')
            if n == 0:
                continue
            yield exp, fov_str.zfill(3), peak_str, masks[:n], raw[:n]


def compute_filtered_fluor(experiments, base_dir=DEFAULT_BASE_DIR,
                           trackastra_dir=DEFAULT_TRACKASTRA_DIR,
                           time_dict_path='data/time_dict_082026.json',
                           method='median', size=3, sigma=1.0, nlm_h=1.15,
                           otsu_scale=1.0, fluor_c='1'):
    """Per-cell mean fluorescence re-measured on spatially filtered images.

    Median filtering dims cells by a few percent, and by more on a noise-dominated run, so
    `fluor_filtered` is not on the raw absolute scale. Compare it only against itself.
    """
    from scipy import ndimage
    from mmtrack.fluor_filters import denoise_fluor

    with open(time_dict_path) as fh:
        time_dict = json.load(fh)

    records = []
    for exp, fov, peak, masks, raw in _iter_trenches(
            experiments, trackastra_dir, base_dir, time_dict, fluor_c):
        fl = denoise_fluor(raw, method=method, size=size, sigma=sigma,
                           nlm_h=nlm_h, otsu_scale=otsu_scale)
        for t in range(len(fl)):
            labs = np.unique(masks[t])
            labs = labs[labs != 0]
            if labs.size == 0:
                continue
            means = np.atleast_1d(ndimage.mean(fl[t], labels=masks[t], index=labs))
            for lab, m in zip(labs, means):
                records.append((exp, fov, peak, int(lab), t, float(m)))

    return pd.DataFrame(records, columns=CELL_KEY + ['fluor_filtered'])


def fiji_display_floor(raw, low_frac=0.25, high_pct=100.0):
    """(vmin, vmax) for a raw trench stack -- the numbers the viewer displays with.

    Must stay in step with `generate_kymograph_viewer.fluor_display_range`: vmax is the
    `high_pct` percentile (100 = the true max) and vmin is `low_frac` x vmax rounded
    HALF-UP, which is what FIJI's Set Display Range receives. Python's round() is banker's
    rounding and gives 8 where a person types 9.
    """
    vmax = float(np.percentile(raw, high_pct))
    return float(int(low_frac * vmax + 0.5)), vmax


def compute_fiji_fluor(experiments, base_dir=DEFAULT_BASE_DIR,
                       trackastra_dir=DEFAULT_TRACKASTRA_DIR,
                       time_dict_path='data/time_dict_082026.json',
                       low_frac=0.25, high_pct=100.0, radius=2.0, min_size=0, fluor_c='1'):
    """Per-cell values measured on the FIJI-transformed images.

    The viewer's filtered layer as numbers: clip at the trench's own display floor
    (0.25 x max), then an ImageJ median of `radius` on the ASSEMBLED kymograph. Values come
    back as counts above the floor, so they stay linear in a.u.

    Scope matters. The median runs on the assembled kymograph, not per frame, so its kernel
    crosses frame seams the way FIJI's does. `denoise_fluor(method='fiji_median')` is the
    per-frame variant and is not interchangeable: the two differ on 8.3% of pixels,
    confined to columns 0, 1, 18 and 19 of every 20-px frame.

    Three statistics per cell-frame:
        fiji_mean  mean over the cell's pixels
        fiji_max   max over the cell's pixels
        fiji_frac  fraction of the cell's pixels above the floor

    THE FLOOR IS PER-TRENCH, so values are NOT comparable across trenches, and about half
    of all cell-frames measure exactly 0. Use them as a within-trench rank. Never feed them
    to a median+MAD threshold: the (gene, experiment) baseline collapses to median 0 on 11
    of 27 groups, putting the bar where every nonzero blip clears it.

    `min_size` > 0 turns on small-object removal, which is beyond the FIJI chain; 0
    reproduces FIJI exactly.
    """
    from scipy import ndimage
    from mmtrack.fluor_filters import (
        from_kymograph, imagej_median, remove_speckle, to_kymograph,
    )

    with open(time_dict_path) as fh:
        time_dict = json.load(fh)

    records = []
    for exp, fov, peak, masks, raw in _iter_trenches(
            experiments, trackastra_dir, base_dir, time_dict, fluor_c):
        n = len(raw)
        vmin, vmax = fiji_display_floor(raw, low_frac, high_pct)
        kymo = imagej_median(to_kymograph(raw), radius)
        kymo = remove_speckle(kymo, min_size, thr=vmin)
        tr = from_kymograph(np.clip(kymo.astype(np.float32) - vmin, 0, None), n)

        for t in range(n):
            labs = np.unique(masks[t])
            labs = labs[labs != 0]
            if labs.size == 0:
                continue
            frame, lab_img = tr[t], masks[t]
            means = np.atleast_1d(ndimage.mean(frame, labels=lab_img, index=labs))
            maxes = np.atleast_1d(ndimage.maximum(frame, labels=lab_img, index=labs))
            # fraction above the floor == mean of the boolean mask over the cell
            fracs = np.atleast_1d(ndimage.mean((frame > 0).astype(np.float32),
                                               labels=lab_img, index=labs))
            for lab, mu, mx, fr in zip(labs, means, maxes, fracs):
                records.append((exp, fov, peak, int(lab), t,
                                float(mu), float(mx), float(fr), vmin, vmax))

    return pd.DataFrame(records, columns=CELL_KEY + [
        'fiji_mean', 'fiji_max', 'fiji_frac', 'fiji_vmin', 'fiji_vmax'])


# ---------------------------------------------------------------------------
# 6d. Annotate: derived units, the degenerate-mask flag, the per-kymograph scale
# ---------------------------------------------------------------------------

def annotate_cells(cells, filtered):
    """The annotated cell table: all experiments, the failed ones tagged not dropped.

    Adds `is_failed_experiment`, `fluor_filtered` (merged in), the two per-ms units,
    `is_degenerate_mask`, and a `_kymo_min` / `_kymo_max` / `_kymo_minmax` triple for each
    of `FLUOR_BASES`.

    The scale comes from NON-degenerate masks only. On a 1-pixel mask the mean is a single
    raw pixel with no averaging, and one such mask can set a whole kymograph's maximum.
    Degenerate rows stay in the table and still get a normalised value against the clean
    scale, so they can land outside [0, 1]. Filter on the flag; do not clip.
    """
    out = cells.copy()
    out['FOV'] = out['FOV'].astype(str).str.zfill(3)
    out['is_failed_experiment'] = out['experiment_name'].isin(FAILED_EXPERIMENTS)

    # fluor_per_ms is only a per-experiment constant rescale if exposure is single-valued,
    # and every cross-gene comparison rests on that.
    per_exp = out.groupby('experiment_name')[EXPOSURE_COL].nunique()
    assert (per_exp == 1).all(), \
        f'{EXPOSURE_COL} varies within: {sorted(per_exp[per_exp > 1].index)}'
    assert out[EXPOSURE_COL].notna().all() and (out[EXPOSURE_COL] > 0).all(), \
        f'{EXPOSURE_COL} has a null or non-positive value'
    # A null in the group key becomes its own min/max group, or vanishes. Neither belongs
    # in an archive.
    assert out[KYMO_GROUP].notna().all().all(), 'a null in the kymograph group key'
    # A null area compares False against the floor and so reads as "not degenerate".
    assert out['area'].notna().all(), 'a null area would read as "not degenerate"'
    assert not out.duplicated(CELL_KEY).any(), 'the cell key is not unique'

    n = len(out)
    filtered = filtered.copy()
    filtered['FOV'] = filtered['FOV'].astype(str).str.zfill(3)
    assert not filtered.duplicated(CELL_KEY).any(), 'duplicate key in the filtered table'
    out = out.merge(filtered, on=CELL_KEY, how='left', validate='one_to_one')
    assert len(out) == n, 'the filtered-fluor merge changed the row count'
    assert out['fluor_filtered'].notna().all(), 'a cell-frame has no filtered fluorescence'

    out['fluor_per_ms'] = out[RAW_FLUOR_COL] / out[EXPOSURE_COL]
    out['fluor_filtered_per_ms'] = out['fluor_filtered'] / out[EXPOSURE_COL]

    floor = MIN_AREA_FRAC * out['experiment_name'].map(
        out.groupby('experiment_name')['area'].median())
    assert floor.notna().all(), 'an experiment has no median area'
    out['is_degenerate_mask'] = out['area'] < floor

    gkeys = [out[c] for c in KYMO_GROUP]
    for base in FLUOR_BASES:
        clean = out[base].where(~out['is_degenerate_mask'])
        g = clean.groupby(gkeys, dropna=False, sort=False)
        lo, hi = g.transform('min'), g.transform('max')
        span = (hi - lo).where(lambda s: s > 0)   # 0 or NaN -> NaN, never a divide-by-zero
        out[f'{base}_kymo_min'] = lo
        out[f'{base}_kymo_max'] = hi
        out[f'{base}_kymo_minmax'] = (out[base] - lo) / span

    print(f'{len(out):,} cell-frames | {out.experiment_name.nunique()} experiments | '
          f'{out.gene.nunique()} genes | {len(out[KYMO_GROUP].drop_duplicates()):,} kymographs '
          f'| {int(out.is_degenerate_mask.sum()):,} degenerate masks | '
          f'{int(out.is_failed_experiment.sum()):,} rows from failed runs (tagged, kept)')
    return out


def check_filtered_alignment(cells):
    """Assert the filtered column is aligned in time with the raw one, per experiment.

    A wrong time_dict `start` makes compute_filtered_fluor read a SHIFTED window and return
    a full set of plausible numbers at 100% coverage. An absolute correlation floor does not
    catch that -- it measures brightness, and on a dim run the aligned correlation is
    legitimately low. What catches it is whether lag 0 beats a deliberate one-frame shift,
    which is self-calibrating per experiment.
    """
    al = cells.sort_values(CELL_KEY[:-1] + ['time_frame'])
    shifted = al.groupby(CELL_KEY[:-1], sort=False)['fluor_filtered']
    al = al.assign(_lag_p1=shifted.shift(1), _lag_m1=shifted.shift(-1))
    lag = al.groupby('experiment_name').apply(
        lambda g: pd.Series({'lag0': g[RAW_FLUOR_COL].corr(g['fluor_filtered']),
                             'shift+1': g[RAW_FLUOR_COL].corr(g['_lag_p1']),
                             'shift-1': g[RAW_FLUOR_COL].corr(g['_lag_m1'])}),
        include_groups=False)
    lag['margin'] = lag['lag0'] - lag[['shift+1', 'shift-1']].max(axis=1)
    bad = lag[lag['margin'] <= 0]
    assert bad.empty, f'a one-frame shift fits better -- start offset misaligned:\n{bad}'
    print(f'alignment: lag 0 beats a +/-1 frame shift for all {len(lag)} experiments '
          f'(smallest margin {lag.margin.min():.4f}, {lag.margin.idxmin()})')
    return lag


def annotate_pairs(pairs, cells):
    """The annotated pair table: both endpoint ids, and every endpoint's cell values.

    Columns are ADDED, never overwritten, so the raw a.u. columns stay exactly as the pair
    builders wrote them and each derived unit gets its own name.
    """
    out = pairs.copy()
    out['FOV'] = out['FOV'].astype(str).str.zfill(3)
    out['is_failed_experiment'] = out['experiment_name'].isin(FAILED_EXPERIMENTS)

    if 'from_track_id' not in out.columns:
        frm = np.where(out['is_division'].astype(bool),
                       out['parent_track_id'], out['track_id'])
        assert not np.isnan(np.asarray(frm, dtype=float)).any(), \
            'a division row has no parent_track_id'
        out['from_track_id'] = frm.astype('int64')
        out['to_track_id'] = out['track_id'].astype('int64')
    # parent_track_id is float64 on a concatenated table, and a float level in the
    # MultiIndex below can reindex to all-NaN without raising.
    out['from_track_id'] = out['from_track_id'].to_numpy().astype('int64')
    out['to_track_id'] = out['to_track_id'].to_numpy().astype('int64')

    lut_src = cells.set_index(CELL_KEY)

    def endpoint(col, tid_col, frame_col):
        idx = pd.MultiIndex.from_arrays([
            out['experiment_name'], out['FOV'], out['trench_id'],
            out[tid_col], out[frame_col].to_numpy().astype('int64'),
        ])
        return lut_src[col].reindex(idx).to_numpy()

    # Proof the keying is right before anything is derived from it: the raw column pulled
    # back this way must reproduce what the pair builder already wrote, on BOTH pair types.
    # This is the only thing standing between a wrong join and a plausible-looking CSV.
    for lbl, tid, fr, want in (('from', 'from_track_id', 'frame_from', 'fluor_from'),
                               ('to', 'to_track_id', 'frame_to', 'fluor_to')):
        got = endpoint(RAW_FLUOR_COL, tid, fr)
        assert np.isfinite(got).all(), \
            f'{lbl}-endpoint lookup missed {int((~np.isfinite(got)).sum()):,} rows'
        dev = float(np.nanmax(np.abs(got - out[want])))
        assert dev < 1e-9, f'{lbl}-endpoint disagrees with the pair table by {dev:g}'
        print(f'{lbl:>4}-endpoint verified: {len(out):,} rows, max deviation {dev:g}')

    for base in FLUOR_BASES:
        out[f'{base}_from'] = endpoint(base, 'from_track_id', 'frame_from')
        out[f'{base}_to'] = endpoint(base, 'to_track_id', 'frame_to')
        out[f'delta_{base}'] = out[f'{base}_to'] - out[f'{base}_from']
        assert out[f'delta_{base}'].notna().all(), f'delta_{base} has nulls'

    out['area_from'] = endpoint('area', 'from_track_id', 'frame_from')
    out['area_to'] = endpoint('area', 'to_track_id', 'frame_to')
    out[EXPOSURE_COL] = endpoint(EXPOSURE_COL, 'to_track_id', 'frame_to')
    for lbl, tid, fr in (('from', 'from_track_id', 'frame_from'),
                         ('to', 'to_track_id', 'frame_to')):
        flag = endpoint('is_degenerate_mask', tid, fr)
        assert pd.notna(flag).all(), f'is_degenerate_mask_{lbl} has nulls'
        out[f'is_degenerate_mask_{lbl}'] = flag.astype(bool)

    # The kymograph scale, so a delta can be normalised straight out of this file. Merged
    # on the three trench keys, not KYMO_GROUP: gene is carried independently in both
    # tables, and joining on it would turn any disagreement into a silent NaN.
    scale_cols = [f'{b}_kymo_{s}' for b in FLUOR_BASES for s in ('min', 'max')]
    assert not (set(scale_cols) & set(out.columns)), 'a scale column already exists on pairs'
    scale = cells[TRENCH_KEYS + scale_cols].drop_duplicates(TRENCH_KEYS)
    n = len(out)
    out = out.merge(scale, on=TRENCH_KEYS, how='left', validate='many_to_one')
    assert len(out) == n, 'the scale merge changed the row count'
    assert out[scale_cols].notna().all().all(), 'a pair belongs to no kymograph in the cells'

    # Both tables must agree on a trench's gene, or the same trench can be grouped two ways.
    gene_cells = cells.drop_duplicates(TRENCH_KEYS).set_index(TRENCH_KEYS)['gene']
    gene_pairs = out.drop_duplicates(TRENCH_KEYS).set_index(TRENCH_KEYS)['gene']
    assert gene_cells.reindex(gene_pairs.index).equals(gene_pairs), \
        "the cell and pair tables disagree about a trench's gene"

    print(f'{len(out):,} pairs x {out.shape[1]} cols | gene agrees for all '
          f'{len(gene_pairs):,} trenches')
    return out


# ---------------------------------------------------------------------------
# 6e. Divisions
# ---------------------------------------------------------------------------

def track_lifetimes(cells):
    """First and last observed frame per unique cell, one row each."""
    return (cells.groupby(LINEAGE_KEYS, observed=True)
                 .agg(first_frame=('time_frame', 'min'),
                      last_frame=('time_frame', 'max'),
                      n_frames=('time_frame', 'nunique'),
                      minutes_per_frame=('minutes_per_frame', 'first'),
                      parent_track_id=('parent_track_id', 'first'))
                 .reset_index())


def division_events(pairs, cells):
    """One row per observed division, keyed to the cell that divided.

    `doubling_min` spans the parent's own first to last observed frame, so it is one
    interval short of a generation: a parent on [f, l] divides at l and its daughter starts
    at l + 1. Pass the UNSCOPED cell table, or parents fall out of the inner join.
    """
    div = pairs[pairs['is_division'].astype(bool)]
    if div.empty:
        raise ValueError('no division pairs in this scope; nothing to average')

    # The lifetime definition below has no meaning if a track divides twice.
    per_parent = (div.groupby(TRENCH_KEYS + ['parent_track_id'], observed=True)['frame_from']
                     .nunique())
    assert (per_parent == 1).all(), 'a parent track divides at more than one frame'

    ev = (div.drop_duplicates(subset=TRENCH_KEYS + ['parent_track_id'])
             .loc[:, TRENCH_KEYS + ['gene', 'parent_track_id', 'frame_from', 'n_siblings']]
             .rename(columns={'parent_track_id': 'track_id',
                              'frame_from': 'division_frame',
                              'n_siblings': 'n_daughters'}))
    ev['track_id'] = ev['track_id'].astype(int)

    n_before = len(ev)
    ev = ev.merge(track_lifetimes(cells), on=LINEAGE_KEYS, how='inner')
    assert len(ev) == n_before, (
        f'{n_before - len(ev)} division events lost their parent rows in the cell table; '
        f'pass the unscoped cell table as `cells`')
    assert (ev['division_frame'] == ev['last_frame']).all(), (
        'a parent divides before its last observed frame -- the pair table and the cell '
        'table disagree about where the track ends')

    ev['doubling_min'] = (ev['last_frame'] - ev['first_frame']) * ev['minutes_per_frame']
    ev['birth_observed'] = ev['parent_track_id'].notna()

    # Jump granularity is the DIVIDING cell's own consecutive-frame pairs. A flagged
    # division pair is the birth of a daughter and does not disqualify the parent's
    # lifetime: daughters are judged against a wider threshold precisely because a division
    # displaces a centroid.
    wt = pairs[~pairs['is_division'].astype(bool)]
    flagged = set(wt.loc[wt['is_jump'].astype(bool), LINEAGE_KEYS]
                    .drop_duplicates().itertuples(index=False, name=None))
    ev['has_jump'] = [k in flagged for k in ev[LINEAGE_KEYS].itertuples(index=False, name=None)]
    print(f'{len(ev):,} divisions | {int(ev.birth_observed.sum()):,} with an observed birth '
          f'| {int(ev.has_jump.sum()):,} with a flagged jump')
    return ev
