# Script made with the intention to figure out delta Y pixel positions of cell centroids
# as well as delta fluorescence between cell pairs 
# used as a QC for bad tracking if delta-Y is abnormal. Output is two type of pairs 
# those in the same cell track, and a connection of pairs connection mother cells with both 
# daughter cells 
# output df is one cell and adds frame_from and frame_to 

import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Full key that uniquely identifies one track (raw track_id is reused across trenches).
LINEAGE_KEYS = ['experiment_name', 'FOV', 'trench_id', 'track_id']

# One trench. track_id and parent_track_id are only comparable inside one of these.
TRENCH_KEYS = ['experiment_name', 'FOV', 'trench_id']

# Columns that identify which parent a division pair came from. NaN on within_track rows.
DIVISION_COLS = ['parent_track_id', 'sibling_track_id', 'n_siblings']

# lacZ experiments on disk (used to locate trackastra outputs / raw fluor TIFFs).
LACZ_EXPERIMENTS = [
    'DUMM_giTG059_068_SC_093025',
    'DUMM_giTG059_noKan_Glucose_031125',
    'DUMM_giTG059_060_061125',
]

DEFAULT_BASE_DIR = '/Volumes/mcovert/Instruments/Covert-lab-scope1/subgen_processed_data'
DEFAULT_TRACKASTRA_DIR = '/Volumes/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output'


def load_cell_data(csv_path, gene='lacZ'):
    usecols = [
        'gene', 'experiment_name', 'FOV', 'trench_id', 'track_id',
        'predicted_lineage', 'parent_track_id', 'time_frame', 'time_min',
        'minutes_per_frame', 'centroid_y', 'centroid_x', 'axis_major_length',
        'intensity_mean_fluor', 'node_id',
    ]
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in usecols,
        dtype={'FOV': str, 'trench_id': str, 'experiment_name': str, 'gene': str},
    )
    df['FOV'] = df['FOV'].str.zfill(3)
    if gene is not None:
        df = df[df['gene'] == gene].copy()
    return df


def compute_filtered_fluor(
    experiments,
    base_dir=DEFAULT_BASE_DIR,
    trackastra_dir=DEFAULT_TRACKASTRA_DIR,
    time_dict_path='data/time_dict_082026.json',
    method='median',
    size=3,
    sigma=1.0,
    nlm_h=1.15,
    otsu_scale=1.0,
    phase_c='0',
    fluor_c='1',
):
    from scipy import ndimage
    import tifffile
    from mmtrack.fluor_filters import denoise_fluor

    with open(time_dict_path) as fh:
        time_dict = json.load(fh)

    # An unreachable trackastra_dir is the single most likely reason this returns nothing --
    if not os.path.isdir(trackastra_dir):
        raise FileNotFoundError(
            f'trackastra_dir does not exist: {trackastra_dir}\n'
            f'If it is on a network volume, check that the volume is mounted.')

    records = []
    for exp in experiments:
        pattern = os.path.join(trackastra_dir, f'{exp}_ctc_masks_*_*.npy')
        mask_paths = sorted(glob.glob(pattern))
        if not mask_paths:
            print(f'    WARNING: no ctc_masks found for {exp} in {trackastra_dir}; '
                  f'it will have no filtered fluorescence.')
        for mask_path in mask_paths:
            name = os.path.basename(mask_path)[:-4]
            fov_str, peak_str = name.split('_ctc_masks_', 1)[1].rsplit('_', 1)

            fluor_path = os.path.join(
                base_dir, exp, 'hyperstacked', 'drift_corrected', 'rotated',
                'mm_channels', 'subtracted',
                f'subtracted_FOV_{fov_str}_region_{peak_str}_c_{fluor_c}.tif',
            )
            if not os.path.exists(fluor_path):
                print(f"    WARNING: missing fluor TIFF for {exp}/{fov_str}/{peak_str}; skipping.")
                continue

            masks = np.load(mask_path)
            start = time_dict.get(exp, {}).get(fov_str, {}).get(peak_str, {}).get('start', 0)
            T = masks.shape[0]
            raw = tifffile.imread(fluor_path)
            fl = raw[start:start + T]
            n = min(T, fl.shape[0])
            if n < T:
                print(f'    WARNING: {exp}/{fov_str}/{peak_str}: fluor stack gives {n} of {T} '
                      f'masked frames (start={start}, tif has {raw.shape[0]}); '
                      f'the last {T - n} frame(s) get no filtered value.')
            masks, fl = masks[:n], fl[:n]

            fl = denoise_fluor(fl, method=method, size=size, sigma=sigma,
                               nlm_h=nlm_h, otsu_scale=otsu_scale)

            fov_i, peak_i = fov_str.zfill(3), peak_str
            for t in range(n):
                labs = np.unique(masks[t])
                labs = labs[labs != 0]
                if labs.size == 0:
                    continue
                means = ndimage.mean(fl[t], labels=masks[t], index=labs)
                for lab, m in zip(labs, np.atleast_1d(means)):
                    records.append((exp, fov_i, peak_i, int(lab), t, float(m)))

    return pd.DataFrame(
        records,
        columns=['experiment_name', 'FOV', 'trench_id', 'track_id',
                 'time_frame', 'fluor_filtered'],
    )


def fiji_display_floor(raw, low_frac=0.25, high_pct=100.0):
    """(vmin, vmax) for a raw trench stack -- the same numbers the viewer displays with.

    Must stay in step with `generate_kymograph_viewer.fluor_display_range`: vmax is the
    `high_pct` percentile (100 = the true max) and vmin is `low_frac` x vmax rounded
    HALF-UP to an integer, which is what FIJI's Set Display Range receives. Python's
    round() is banker's rounding and would give 8 where a person types 9.
    """
    vmax = float(np.percentile(raw, high_pct))
    return float(int(low_frac * vmax + 0.5)), vmax


def compute_fiji_fluor(
    experiments,
    base_dir=DEFAULT_BASE_DIR,
    trackastra_dir=DEFAULT_TRACKASTRA_DIR,
    time_dict_path='data/time_dict_082026.json',
    low_frac=0.25,
    high_pct=100.0,
    radius=2.0,
    min_size=0,
    fluor_c='1',
):
    """Per-cell values measured on the FIJI-transformed images.

    Reproduces the kymograph viewer's filtered layer as NUMBERS: clip at the trench's own
    display floor (0.25 x max), then an ImageJ median of `radius` on the ASSEMBLED kymograph.
    That two-step chain is the whole transform; `min_size` defaults to 0, i.e. small-object
    removal is OFF and opt-in. It was on (10 px) up to 090826 and was dropped because it had
    become a DETECTION parameter rather than a denoiser -- trenches read as fully dark only
    because sub-10px components were deleted, not because nothing cleared the floor. Turning
    it off raises the lit fraction from 40.3% to 48.9% of cell-frames, which is why
    a superseded FIJI-values burst rule had its lit_gate re-swept from 0.60 to 0.90. Order and scope matter --

      * the median runs on the assembled kymograph, not per frame, so its kernel crosses
        frame seams the way FIJI's does. `denoise_fluor(method='fiji_median')` is the
        per-frame variant and is NOT interchangeable: they differ on 8.3% of pixels,
        confined to columns 0,1,18,19 of every 20-px frame.
      * values are returned as counts ABOVE the floor (`clip(kymo - vmin, 0, None)`), so
        the display map is not applied and the numbers stay linear in a.u.

    Three statistics per cell-frame, all on that transformed image:
        fiji_mean  mean over the cell's pixels
        fiji_max   max  over the cell's pixels
        fiji_frac  fraction of the cell's pixels that are above the floor

    They are SPARSE by construction -- ~98.9% of pixels sit at zero after the clip, so most
    cells measure exactly 0. That is the intended behaviour (a cell only registers when it
    clears 25% of its own trench's brightest pixel), but it breaks the (gene, experiment)
    median+MAD baseline the v1-v7 routes rely on: measured, three of four gene-experiment
    groups collapse to median 0 / mad_scale 1e-6, giving height ~ 4e-6 where every nonzero
    blip clears the bar. Do not feed these columns to a MAD-based definition.

    PER-TRENCH FLOOR, so values are NOT comparable across trenches: each trench's floor is
    set by its own brightest pixel, which in a bursting trench is the burst itself. Within
    a trench they rank bursts well (measured: 21/21 v6 events nonzero, median percentile
    rank 0.993).

    Returns a DataFrame keyed by (experiment_name, FOV, trench_id, track_id, time_frame),
    with the same key dtypes as `compute_filtered_fluor` -- FOV zero-padded to 3, trench_id
    the raw peak id, track_id/time_frame ints, time_frame the 0-based kymograph column.
    """
    import tifffile
    from scipy import ndimage

    from mmtrack.fluor_filters import (
        from_kymograph, imagej_median, remove_speckle, to_kymograph,
    )

    with open(time_dict_path) as fh:
        time_dict = json.load(fh)

    if not os.path.isdir(trackastra_dir):
        raise FileNotFoundError(
            f'trackastra_dir does not exist: {trackastra_dir}\n'
            f'If it is on a network volume, check that the volume is mounted.')

    records = []
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
            start = time_dict.get(exp, {}).get(fov_str, {}).get(peak_str, {}).get('start', 0)
            T = masks.shape[0]
            raw = tifffile.imread(fluor_path)[start:start + T]
            n = min(T, raw.shape[0])
            if n < T:
                print(f'    WARNING: {exp}/{fov_str}/{peak_str}: fluor gives {n} of {T} frames.')
            masks, raw = masks[:n], raw[:n]
            if n == 0:
                continue

            vmin, vmax = fiji_display_floor(raw, low_frac, high_pct)
            kymo = imagej_median(to_kymograph(raw), radius)
            kymo = remove_speckle(kymo, min_size, thr=vmin)
            tr = from_kymograph(np.clip(kymo.astype(np.float32) - vmin, 0, None), n)

            fov_i = fov_str.zfill(3)
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
                    records.append((exp, fov_i, peak_str, int(lab), t,
                                    float(mu), float(mx), float(fr), vmin, vmax))

    return pd.DataFrame(
        records,
        columns=['experiment_name', 'FOV', 'trench_id', 'track_id', 'time_frame',
                 'fiji_mean', 'fiji_max', 'fiji_frac', 'fiji_vmin', 'fiji_vmax'],
    )


def endpoint_track_ids(pairs):
    """(from_track_id, to_track_id) as int64 arrays, one entry per pair row.

    Reads the columns the pair builders now emit. Falls back to deriving them for a pair
    table written before those columns existed (`data/track_pairs_082026.csv` and the
    `notebooks/figures/lacZ_*_track_pairs.csv` files), so an old table still loads.

    The int64 cast is not cosmetic: `parent_track_id` is float64 on a concatenated pair
    table because within_track rows fill it with NaN, and a float level in a MultiIndex can
    reindex to all-NaN without raising.
    """
    if 'from_track_id' in pairs.columns and 'to_track_id' in pairs.columns:
        return (pairs['from_track_id'].to_numpy().astype('int64'),
                pairs['to_track_id'].to_numpy().astype('int64'))

    is_div = (pairs['is_division'].eq(True).to_numpy() if 'is_division' in pairs.columns
              else np.zeros(len(pairs), dtype=bool))
    frm = pairs['track_id'].to_numpy()
    if 'parent_track_id' in pairs.columns:
        frm = np.where(is_div, pairs['parent_track_id'].to_numpy(), frm)
    if np.isnan(np.asarray(frm, dtype=float)).any():
        raise ValueError('a division pair has no parent_track_id; cannot key its from-endpoint')
    return frm.astype('int64'), pairs['track_id'].to_numpy().astype('int64')


def apply_filtered_fluor(pairs, filtered_df):
    """Overwrite each pair's fluorescence with filtered values and recompute delta_fluor.

    Maps `filtered_df` (keyed by exp/FOV/trench/track/time) onto both endpoints of every
    pair. Pairs whose endpoints lack a filtered value get NaN (dropped downstream).
    delta_y is untouched. Returns a new pairs DataFrame.

    Each endpoint keys on its OWN track: on a division row `frame_from` belongs to the parent
    while `track_id` is the daughter, so keying both endpoints on `track_id` -- as this did
    until the from-endpoint was split out -- left every division pair with a NaN `fluor_from`
    and therefore a NaN `delta_fluor`, silently.
    """
    lut = filtered_df.set_index(
        ['experiment_name', 'FOV', 'trench_id', 'track_id', 'time_frame']
    )['fluor_filtered']

    def _lookup(track_vals, frame_col):
        idx = pd.MultiIndex.from_arrays([
            pairs['experiment_name'], pairs['FOV'], pairs['trench_id'],
            track_vals, pairs[frame_col].to_numpy().astype('int64'),
        ])
        return lut.reindex(idx).to_numpy()

    from_tid, to_tid = endpoint_track_ids(pairs)
    out = pairs.copy()
    out['fluor_from'] = _lookup(from_tid, 'frame_from')
    out['fluor_to'] = _lookup(to_tid, 'frame_to')
    out['delta_fluor'] = out['fluor_to'] - out['fluor_from']
    return out

# does bulk of creating cell pairs using centroid_y and trench_ids in combo with function below 

def compute_consecutive_pairs(
    df,
    y_col='centroid_y',
    fluor_col='intensity_mean_fluor',
    time_col='time_frame',
    length_col='axis_major_length',
):
    frames = []
    for keys, g in df.groupby(LINEAGE_KEYS, sort=False):
        g = g.sort_values(time_col)
        exp, fov, trench, track = keys

        cur = g.iloc[:-1]              # every row except the last -> "from" member
        nxt = g.iloc[1:]              # every row except the first -> "to" member
        if len(cur) == 0:
            continue

        pair = pd.DataFrame({
            'experiment_name': exp,
            'FOV': fov,
            'trench_id': trench,
            'track_id': track,
            # Endpoint keys. Both endpoints are the same cell here; on a division pair they
            # differ, which is the whole reason these columns exist. Carrying them makes the
            # merge back onto the cell table unambiguous per endpoint -- see the MERGE KEY
            # note in the module docstring.
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
        # Acquisition interval varies (5 vs 10 min across experiments), so carry it and
        # the wall-clock times through rather than assuming frames are comparable.
        for col, src in (('minutes_per_frame', 'minutes_per_frame'),
                         ('time_min_from', 'time_min')):
            if src in g.columns:
                pair[col] = cur[src].to_numpy()
        if 'time_min' in g.columns:
            pair['time_min_to'] = nxt['time_min'].to_numpy()

        # Cell length at both endpoints. The mean is the natural scale for "did this
        # track land on a neighbour": it approximates the centroid spacing between two
        # adjacent stacked cells, and averaging both endpoints keeps it symmetric in time.
        if length_col in g.columns:
            pair['length_from'] = cur[length_col].to_numpy()
            pair['length_to'] = nxt[length_col].to_numpy()
            pair['cell_length'] = (pair['length_from'] + pair['length_to']) / 2

        pair['frame_gap'] = pair['frame_to'] - pair['frame_from']
        pair['delta_y'] = pair['y_to'] - pair['y_from']
        pair['delta_fluor'] = pair['fluor_to'] - pair['fluor_from']

        # Kept column-aligned with compute_division_pairs so the two concat without
        # introducing NaN-only columns on one side.
        pair['pair_type'] = 'within_track'
        pair['is_division'] = False
        pair['bad_frame_gap'] = pair['frame_gap'].ne(1)
        for col in DIVISION_COLS:
            pair[col] = np.nan
        frames.append(pair)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_division_pairs(
    df,
    y_col='centroid_y',
    fluor_col='intensity_mean_fluor',
    time_col='time_frame',
    length_col='axis_major_length',
):
    """Pair parent cell A's LAST frame with each daughter's FIRST frame.

    A division is the one transition compute_consecutive_pairs cannot see, because it
    groups by track_id and the daughters carry new track_ids. Each daughter row in `df`
    names its parent in `parent_track_id` (NaN for root tracks), so the parent's last
    observation is joined on (experiment_name, FOV, trench_id, parent_track_id) -- raw
    track_ids repeat across trenches, so the trench keys are part of the join.

    `cell_length` is cell A's own length, NOT the two-endpoint mean used for within_track
    pairs: the question here is "did the daughter land where a daughter of THIS cell
    should", and A's length is what sets that scale. flag_jumps applies the wider
    division threshold to it.

    Returns one row per (parent, daughter) link -- two per normal division -- column-aligned
    with compute_consecutive_pairs, with `pair_type='division'`, `is_division=True`, and
    `time_frame` left NaN (see the module docstring's MERGE KEY note). Returns an empty
    DataFrame if `df` carries no parent_track_id column or no non-root tracks.
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

    # Same reindex-by-MultiIndex lookup as apply_filtered_fluor. Key dtypes must match
    # load_cell_data's (FOV zero-padded str, trench_id str) or every lookup silently
    # returns NaN.
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
        # Both are int64 here (parent_track_id was cast to track_id's dtype above) and neither
        # is ever NaN, so unlike parent_track_id itself they survive the concat as integers.
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
        # Cell A's own length -- see the docstring.
        pair['cell_length'] = pair['length_from']

    pair['frame_gap'] = pair['frame_to'] - pair['frame_from']
    pair['delta_y'] = pair['y_to'] - pair['y_from']
    pair['delta_fluor'] = pair['fluor_to'] - pair['fluor_from']

    pair['pair_type'] = 'division'
    pair['is_division'] = True
    pair['parent_track_id'] = daughters['parent_track_id'].to_numpy()
    # Sibling bookkeeping: a normal division yields exactly 2. n_siblings != 2 means the
    # lineage graph gave this parent a different number of children, which is worth
    # auditing on its own.
    sib = daughters.groupby(TRENCH_KEYS + ['parent_track_id'], sort=False)['track_id']
    pair['n_siblings'] = sib.transform('size').to_numpy()
    pair['sibling_track_id'] = (
        sib.transform('sum').to_numpy() - pair['track_id'].to_numpy()
    )
    # Only meaningful for a clean 2-way split; otherwise the "sum minus self" trick has
    # no single sibling to name.
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
    """Flag pairs whose delta_y is an outlier (likely a tracking mistake).

    A pair is flagged when |delta_y| > threshold * cell_length. A tracking error means the
    track landed on a *different* cell, which in a trench of stacked cells takes roughly
    one cell length of apparent motion. Normalising by cell length is what makes the
    criterion independent of magnification, cell size and frame interval, which a fixed
    pixel window is not. Adds `delta_y_cell_lengths`, so the bounds come back in
    cell-length units too.

    The threshold is per-row: `division_length_k` (default 1.5) on rows with is_division
    True, `length_k` elsewhere. A division moves the centroid further than one frame of
    growth, so holding it to the within-track cutoff would flag correct divisions. The
    applied value is recorded in `dy_threshold_cell_lengths`. Inputs without an
    `is_division` column are treated as all-within-track.

    Adds an `is_jump` boolean column. Returns (pairs, lower, upper); the bounds are the
    WITHIN-TRACK ones, so read the division bound off `dy_threshold_cell_lengths`
    (callers plot the two subsets separately).
    """
    pairs = pairs.copy()

    if 'cell_length' not in pairs.columns:
        raise ValueError(
            "flag_jumps needs a 'cell_length' column; compute_consecutive_pairs only "
            "adds it when the input carries axis_major_length."
        )
    # A zero-area mask is a segmentation failure, not evidence of a hop, and would
    # otherwise divide to infinity. Substitute the dataset median length.
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


# ---------------------------------------------------------------------------
# Plotting  (transparent SVG + PNG, top/right spines hidden)
# ---------------------------------------------------------------------------

# matplotlib tab10, so these panels share one palette with the gene-keyed figures in
# notebooks/analysis_notebooks/trackastra_figures_080926.ipynb. These panels pool across
# genes and carry no gene legend, so C_ALL reusing lacZ's gene colour is not a clash --
# the two never appear in the same axes.
C_MAIN = '#1f77b4'   # tab10 blue   - primary histogram
C_GOOD = '#2ca02c'   # tab10 green  - retained pairs
C_JUMP = '#d62728'   # tab10 red    - flagged pairs and cutoff lines
C_ALL  = '#7f7f7f'   # tab10 gray   - all-pairs backdrop


def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save(fig, out_dir, name, close=True):
    """Write transparent SVG + 200 dpi PNG and return the SVG path.

    close=False leaves the figure open so a notebook's inline backend renders it;
    the CLI keeps the default so long runs do not accumulate open figures.
    """
    os.makedirs(out_dir, exist_ok=True)
    svg = os.path.join(out_dir, name + '.svg')
    png = os.path.join(out_dir, name + '.png')
    fig.savefig(svg, transparent=True, bbox_inches='tight')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    if close:
        plt.close(fig)
    return svg


# Column, axis label and default plotting window for every Δy panel. The cell-length
# ratio has a very long tail (out to ~43 cell lengths on this dataset), so without a
# window the +/-1 cutoff collapses into the central bin and the plot is useless.
DY_VALUE_COL = 'delta_y_cell_lengths'
DY_XLABEL = 'Δy / cell length'
DY_DEFAULT_RANGE = (-4.0, 4.0)


def plot_dy_histogram(pairs, bounds, gene, out_dir, bins=60, close=True,
                      xrange='default'):
    lower, upper = bounds
    if xrange == 'default':
        xrange = DY_DEFAULT_RANGE
    n_jump = int(pairs['is_jump'].sum())
    n_total = len(pairs)

    values = pairs[DY_VALUE_COL].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, range=xrange, color=C_MAIN, alpha=0.85)
    ax.axvline(lower, color=C_JUMP, ls='--', lw=1.5,
               label=f'cutoff [{lower:.1f}, {upper:.1f}]')
    ax.axvline(upper, color=C_JUMP, ls='--', lw=1.5)
    if xrange is not None:
        n_off = int(((values < xrange[0]) | (values > xrange[1])).sum())
        if n_off:
            # Say so rather than letting a clipped axis imply the tail does not exist.
            ax.plot([], [], ' ',
                    label=f'{n_off} pair(s) beyond ±{xrange[1]:g}, off-axis')
    ax.set_yscale('log')
    ax.set_xlabel(DY_XLABEL)
    ax.set_ylabel('pair count (log)')
    ax.set_title(f'{gene}: frame-to-frame y-displacement\n'
                 f'{n_jump}/{n_total} pairs flagged as jumps '
                 f'({100*n_jump/max(n_total,1):.1f}%)')
    ax.legend(frameon=False)
    _style(ax)
    return _save(fig, out_dir, f'{gene}_delta_y_hist', close=close)


def plot_dfluor_histogram(all_pairs, good_pairs, gene, out_dir, bins=60, close=True):
    # Only positive fluorescence increases (drops are not plotted).
    all_pos = all_pairs.loc[all_pairs['delta_fluor'] > 0, 'delta_fluor'].dropna()
    good_pos = good_pairs.loc[good_pairs['delta_fluor'] > 0, 'delta_fluor'].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    hi = np.nanpercentile(all_pos, 99.5) if len(all_pos) else 1.0
    rng = (0, hi)
    ax.hist(all_pos, bins=bins, range=rng,
            color=C_ALL, alpha=0.6, label=f'all pairs (n={len(all_pos)})')
    ax.hist(good_pos, bins=bins, range=rng,
            color=C_GOOD, alpha=0.7,
            label=f'good pairs (n={len(good_pos)})')
    ax.set_yscale('log')
    ax.set_xlabel('Δfluor increase = intensity_mean_fluor(t+1) − (t),  > 0 only')
    ax.set_ylabel('pair count (log)')
    ax.set_title(f'{gene}: frame-to-frame fluorescence change')
    ax.legend(frameon=False)
    _style(ax)
    return _save(fig, out_dir, f'{gene}_delta_fluor_hist', close=close)


def plot_dy_vs_dfluor(pairs, bounds, gene, out_dir, close=True, xrange='default'):
    # Only positive fluorescence increases (drops are not plotted).
    if xrange == 'default':
        xrange = DY_DEFAULT_RANGE
    pos = pairs[pairs['delta_fluor'] > 0]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    good = pos[~pos['is_jump']]
    bad = pos[pos['is_jump']]
    ax.scatter(good[DY_VALUE_COL], good['delta_fluor'], s=8, alpha=0.4,
               color=C_GOOD, label='good')
    ax.scatter(bad[DY_VALUE_COL], bad['delta_fluor'], s=12, alpha=0.6,
               color=C_JUMP, label='jump')
    for b in bounds:
        ax.axvline(b, color=C_JUMP, ls='--', lw=1)
    if xrange is not None:
        ax.set_xlim(*xrange)
    ax.set_xlabel(DY_XLABEL)
    ax.set_ylabel('Δfluor increase  (> 0 only)')
    ax.set_title(f'{gene}: Δy vs Δfluor increase per pair')
    ax.legend(frameon=False)
    _style(ax)
    return _save(fig, out_dir, f'{gene}_dy_vs_dfluor', close=close)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_track_displacement_analysis(
    csv_path,
    gene='lacZ',
    out_dir='notebooks/figures',
    dy_length_k=1.0,
    division_length_k=1.5,
    include_divisions=True,
    fluor_col='intensity_mean_fluor',
    bins=60,
    fluor_method='none',
    base_dir=DEFAULT_BASE_DIR,
    trackastra_dir=DEFAULT_TRACKASTRA_DIR,
    time_dict_path='data/time_dict_082026.json',
    filter_size=3,
    filter_sigma=1.0,
    nlm_h=1.15,
    otsu_scale=1.0,
    experiments=None,
    pairs_csv=None,
    close_figs=True,
):
    """Full analysis: load -> pairs -> flag jumps -> fluor change -> plots + pair CSV.

    If `fluor_method` is a real filter (median/mean/gaussian/nlm/otsu), per-cell
    fluorescence is recomputed on the filtered images and swapped in before the
    fluorescence-change step. delta_y is unaffected.

    Two pair populations are analysed together (see the module docstring): within-track
    pairs at `dy_length_k` x cell length, and parent->daughter division pairs at
    `division_length_k` x the parent's own length. `include_divisions=False` restores the
    within-track-only behaviour.

    pairs_csv overrides where the pair table is written (default
    `{out_dir}/{label}_track_pairs.csv`). close_figs=False leaves the figures open so
    a notebook renders them inline.
    """
    filtering = fluor_method not in (None, 'none', 'raw')
    label = f'{gene or "all"}_{fluor_method}' if filtering else (gene or 'all')
    print(f"Loading {csv_path} (gene={gene}) ...")
    df = load_cell_data(csv_path, gene=gene)
    print(f"  {len(df)} cell×frame rows; "
          f"{df.groupby(LINEAGE_KEYS, sort=False).ngroups} tracks")

    pairs = compute_consecutive_pairs(df, fluor_col=fluor_col)
    if pairs.empty:
        print("No consecutive pairs found. Exiting.")
        return None
    print(f"  {len(pairs)} consecutive within-track pairs "
          f"(frame gaps: {sorted(pairs['frame_gap'].unique().tolist())})")

    if include_divisions:
        div = compute_division_pairs(df, fluor_col=fluor_col)
        if not div.empty:
            n_parents = div.groupby(TRENCH_KEYS + ['parent_track_id'], sort=False).ngroups
            print(f"  {len(div)} parent->daughter division pairs from {n_parents} "
                  f"division(s)")
            pairs = pd.concat([pairs, div], ignore_index=True)

    # Cell-length normalisation is the only mode: a fixed pixel window's meaning shifts
    # with the frame interval, so there is nothing sensible to fall back to.
    if not dy_length_k or dy_length_k <= 0:
        raise ValueError(f"dy_length_k must be positive (got {dy_length_k!r}).")
    pairs, lower, upper = flag_jumps(
        pairs, length_k=dy_length_k, division_length_k=division_length_k)
    n_jump = int(pairs['is_jump'].sum())
    print(f"  Δy cutoff = [{lower:.2f}, {upper:.2f}] cell_lengths "
          f"(|Δy| > {dy_length_k}× cell length); {n_jump} pairs flagged as jumps "
          f"({100*n_jump/len(pairs):.1f}%)")
    # Per-type breakdown: the two populations have different cutoffs and very different
    # jump rates, so a single pooled percentage hides which one is misbehaving.
    for kind, sub in pairs.groupby('pair_type', sort=False):
        n_k = int(sub['is_jump'].sum())
        cut = float(sub['dy_threshold_cell_lengths'].iloc[0])
        print(f"    {kind:<12} {len(sub):>6} pairs, cutoff ±{cut:.2f} cell lengths, "
              f"{n_k} jump(s) ({100*n_k/max(len(sub),1):.1f}%)")

    if filtering:
        exps = experiments if experiments is not None else sorted(df['experiment_name'].unique())
        print(f"  Recomputing fluorescence on '{fluor_method}'-filtered images "
              f"for {len(exps)} experiment(s) ...")
        filt = compute_filtered_fluor(
            exps, base_dir=base_dir, trackastra_dir=trackastra_dir,
            time_dict_path=time_dict_path, method=fluor_method,
            size=filter_size, sigma=filter_sigma, nlm_h=nlm_h, otsu_scale=otsu_scale,
        )
        pairs = apply_filtered_fluor(pairs, filt)
        # Broken out per pair_type on purpose. A pooled percentage is what let the
        # from-endpoint keying bug hide: division pairs were 100% NaN, which showed up as a
        # plausible-looking ~86% pooled coverage and read like a missing-TIFF problem.
        cov = int(pairs['delta_fluor'].notna().sum())
        print(f"  Filtered fluor mapped onto {cov}/{len(pairs)} pairs "
              f"({100*cov/len(pairs):.1f}% coverage)")
        for kind, sub_p in pairs.groupby('pair_type', sort=False):
            n_ok = int(sub_p['delta_fluor'].notna().sum())
            print(f"    {kind:<12} {n_ok:>6}/{len(sub_p):<6} "
                  f"({100*n_ok/max(len(sub_p),1):.1f}%)")
            if n_ok == 0:
                raise RuntimeError(
                    f"every '{kind}' pair has a NaN delta_fluor. Either the filtered table is "
                    f"empty (is {trackastra_dir!r} mounted?) or the endpoint keys do not match "
                    f"it -- check endpoint_track_ids and the key dtypes.")

    good = pairs[~pairs['is_jump']].copy()
    print(f"  {len(good)} good pairs retained")

    # Join key back onto the cell table: a within-track pair is attributed to the frame it
    # starts from, so this left-merges 1:1 onto all_data on
    # (experiment_name, FOV, trench_id, track_id, time_frame). Left NaN on division rows,
    # where frame_from is the parent's frame but track_id is the daughter's -- see the
    # module docstring's MERGE KEY note.
    pairs.insert(len(pairs.columns), 'time_frame',
                 pairs['frame_from'].where(~pairs['is_division'].eq(True)))

    # Figures. The three original panels stay on within-track pairs only, so their
    # content is unchanged by the addition of division pairs; the division population
    # gets its own copies of two of them (same functions, wider cutoff, own filenames).
    within = pairs[pairs['pair_type'] == 'within_track']
    good_within = good[good['pair_type'] == 'within_track']
    figs = [
        plot_dy_histogram(within, (lower, upper), label, out_dir, bins=bins,
                          close=close_figs),
        plot_dfluor_histogram(within, good_within, label, out_dir, bins=bins,
                              close=close_figs),
        plot_dy_vs_dfluor(within, (lower, upper), label, out_dir, close=close_figs),
    ]
    div_pairs = pairs[pairs['pair_type'] == 'division']
    if not div_pairs.empty:
        div_bounds = (-float(division_length_k), float(division_length_k))
        figs += [
            plot_dy_histogram(div_pairs, div_bounds, f'{label}_division', out_dir,
                              bins=bins, close=close_figs),
            plot_dy_vs_dfluor(div_pairs, div_bounds, f'{label}_division', out_dir,
                              close=close_figs),
        ]
    print('  Figures: ' + '\n           '.join(figs))

    # Tables
    os.makedirs(out_dir, exist_ok=True)
    if pairs_csv is None:
        pairs_csv = os.path.join(out_dir, f'{label}_track_pairs.csv')
    os.makedirs(os.path.dirname(os.path.abspath(pairs_csv)), exist_ok=True)
    pairs.to_csv(pairs_csv, index=False)
    print(f"  Tables:  {pairs_csv}")

    return {
        'pairs': pairs,
        'good': good,
        'dy_bounds': (lower, upper),
        'division_dy_bounds': (-float(division_length_k), float(division_length_k)),
        'n_jump': n_jump,
        'n_by_type': pairs['pair_type'].value_counts().to_dict(),
    }
