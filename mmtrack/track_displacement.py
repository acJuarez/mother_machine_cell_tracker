"""
Track y-displacement QC + fluorescence-change analysis.

In Mother Machine trenches cells only move along the trench's y-axis between
consecutive frames. Large frame-to-frame y-jumps within a track are almost always
tracking mistakes (the track hopped to a different cell). This module:

  1. Breaks each track into consecutive-frame pairs (connected by track_id). A cell
     that has both a previous and a next frame appears in two pairs -- once as the
     first member and once as the second.
  2. Computes the y-displacement (delta_y) for every pair and uses its distribution
     to flag likely-mis-tracked pairs (big jumps).
  3. On the retained "good" pairs, computes the change in average fluorescence
     (delta_fluor) and flags pairs with significant deltas.

Pairs are formed strictly WITHIN a track_id (never across divisions): a daughter's
first frame vs. the parent's last frame is a real positional jump, not an error.

Coordinate note: `centroid_y` is the true along-trench axis. `centroid_x` in the
tracked data is kymograph-space (offset by time * frame_width) and must NOT be used
for displacement (see mmtrack/trackastra_feature_extraction.py).
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Full key that uniquely identifies one track (raw track_id is reused across trenches).
LINEAGE_KEYS = ['experiment_name', 'FOV', 'trench_id', 'track_id']

_MAD_TO_STD = 1.4826  # scale factor: robust std estimate = 1.4826 * MAD

# lacZ experiments on disk (used to locate trackastra outputs / raw fluor TIFFs).
LACZ_EXPERIMENTS = [
    'DUMM_giTG059_068_SC_093025',
    'DUMM_giTG059_noKan_Glucose_031125',
    'DUMM_giTG059_060_061125',
]

DEFAULT_BASE_DIR = '/Volumes/mcovert/Instruments/Covert-lab-scope1/subgen_processed_data'
DEFAULT_TRACKASTRA_DIR = '/Volumes/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output'


def load_cell_data(csv_path, gene='lacZ'):
    """Load the aggregated cell×frame CSV and filter to one gene.

    Only the columns needed for this analysis are read (the `coords` column is huge
    and skipped). Pass gene=None to keep all genes.
    """
    usecols = [
        'gene', 'experiment_name', 'FOV', 'trench_id', 'track_id',
        'predicted_lineage', 'parent_track_id', 'time_frame',
        'centroid_y', 'centroid_x', 'intensity_mean_fluor', 'node_id',
    ]
    df = pd.read_csv(csv_path, usecols=usecols)
    if gene is not None:
        df = df[df['gene'] == gene].copy()
    return df


def compute_filtered_fluor(
    experiments,
    base_dir=DEFAULT_BASE_DIR,
    trackastra_dir=DEFAULT_TRACKASTRA_DIR,
    time_dict_path='data/time_dict_070826.json',
    method='median',
    size=3,
    sigma=1.0,
    nlm_h=1.15,
    otsu_scale=1.0,
    phase_c='0',
    fluor_c='1',
):
    """Recompute per-cell mean fluorescence on FILTERED images.

    For each trackastra trench (ctc_masks .npy) of the given experiments, loads the
    raw fluor TIFF, trims it to the tracked window (via time_dict start offset), applies
    `denoise_fluor(method, ...)`, then computes per-frame per-label mean fluorescence
    (label == track_id). This reproduces the pipeline's `intensity_mean_fluor` exactly
    when method='none', and gives the filtered equivalent otherwise.

    Returns a DataFrame keyed by (experiment_name, FOV, trench_id, track_id, time_frame)
    with column `fluor_filtered`. FOV/trench_id/track_id/time_frame are ints (to match
    the CSV loaded by load_cell_data).
    """
    from scipy import ndimage
    import tifffile
    from mmtrack.fluor_filters import denoise_fluor

    with open(time_dict_path) as fh:
        time_dict = json.load(fh)

    records = []
    for exp in experiments:
        pattern = os.path.join(trackastra_dir, f'{exp}_ctc_masks_*_*.npy')
        for mask_path in sorted(glob.glob(pattern)):
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
            masks, fl = masks[:n], fl[:n]

            fl = denoise_fluor(fl, method=method, size=size, sigma=sigma,
                               nlm_h=nlm_h, otsu_scale=otsu_scale)

            fov_i, peak_i = int(fov_str), int(peak_str)
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


def apply_filtered_fluor(pairs, filtered_df):
    """Overwrite each pair's fluorescence with filtered values and recompute delta_fluor.

    Maps `filtered_df` (keyed by exp/FOV/trench/track/time) onto both endpoints of every
    pair. Pairs whose endpoints lack a filtered value get NaN (dropped downstream).
    delta_y is untouched. Returns a new pairs DataFrame.
    """
    lut = filtered_df.set_index(
        ['experiment_name', 'FOV', 'trench_id', 'track_id', 'time_frame']
    )['fluor_filtered']

    def _lookup(frame_col):
        idx = pd.MultiIndex.from_arrays([
            pairs['experiment_name'], pairs['FOV'], pairs['trench_id'],
            pairs['track_id'], pairs[frame_col],
        ])
        return lut.reindex(idx).to_numpy()

    out = pairs.copy()
    out['fluor_from'] = _lookup('frame_from')
    out['fluor_to'] = _lookup('frame_to')
    out['delta_fluor'] = out['fluor_to'] - out['fluor_from']
    return out


def compute_consecutive_pairs(
    df,
    y_col='centroid_y',
    fluor_col='intensity_mean_fluor',
    time_col='time_frame',
):
    """Form consecutive-frame pairs within each track.

    Groups by LINEAGE_KEYS, sorts by time, and joins each observation to the next
    one in the same track via shift(-1). The last observation of each track has no
    successor and is dropped. Returns one row per pair.
    """
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
        pair['frame_gap'] = pair['frame_to'] - pair['frame_from']
        pair['delta_y'] = pair['y_to'] - pair['y_from']
        pair['delta_fluor'] = pair['fluor_to'] - pair['fluor_from']
        frames.append(pair)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _robust_center_scale(values):
    """Return (median, 1.4826*MAD) for a 1-D array, ignoring NaNs."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0.0, 0.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, _MAD_TO_STD * mad


def flag_jumps(pairs, k=4.0, abs_cutoff=None):
    """Flag pairs whose delta_y is an outlier (likely a tracking mistake).

    Default: robust bounds centered on the median delta_y (which is non-zero because
    of the systematic growth drift): keep |delta_y - median| <= k * (1.4826*MAD).
    If `abs_cutoff` is given, use a simple symmetric cutoff |delta_y| > abs_cutoff.

    Adds an `is_jump` boolean column. Returns (pairs, lower_bound, upper_bound).
    """
    pairs = pairs.copy()
    if abs_cutoff is not None:
        lower, upper = -float(abs_cutoff), float(abs_cutoff)
    else:
        med, scale = _robust_center_scale(pairs['delta_y'])
        lower, upper = med - k * scale, med + k * scale

    pairs['is_jump'] = (pairs['delta_y'] < lower) | (pairs['delta_y'] > upper)
    return pairs, lower, upper


def flag_significant_fluor(good_pairs, k=4.0):
    """Flag good pairs with a significant fluorescence INCREASE via a robust z-score.

    z = (delta_fluor - median) / (1.4826*MAD). Only positive increases are counted
    significant: z >= k AND delta_fluor > 0. Adds `fluor_zscore`, `is_significant`.
    Operates on a copy.
    """
    good = good_pairs.copy()
    med, scale = _robust_center_scale(good['delta_fluor'])
    if scale == 0:
        good['fluor_zscore'] = 0.0
        good['is_significant'] = False
    else:
        good['fluor_zscore'] = (good['delta_fluor'] - med) / scale
        good['is_significant'] = (good['fluor_zscore'] >= k) & (good['delta_fluor'] > 0)
    return good


# ---------------------------------------------------------------------------
# Plotting  (transparent SVG + PNG, top/right spines hidden)
# ---------------------------------------------------------------------------

def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    svg = os.path.join(out_dir, name + '.svg')
    png = os.path.join(out_dir, name + '.png')
    fig.savefig(svg, transparent=True, bbox_inches='tight')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return svg


def plot_dy_histogram(pairs, bounds, gene, out_dir, bins=60):
    lower, upper = bounds
    n_jump = int(pairs['is_jump'].sum())
    n_total = len(pairs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pairs['delta_y'].dropna(), bins=bins, color='#4C72B0', alpha=0.85)
    ax.axvline(lower, color='#C44E52', ls='--', lw=1.5,
               label=f'cutoff [{lower:.1f}, {upper:.1f}]')
    ax.axvline(upper, color='#C44E52', ls='--', lw=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Δy = centroid_y(t+1) − centroid_y(t)  [px]')
    ax.set_ylabel('pair count (log)')
    ax.set_title(f'{gene}: frame-to-frame y-displacement\n'
                 f'{n_jump}/{n_total} pairs flagged as jumps '
                 f'({100*n_jump/max(n_total,1):.1f}%)')
    ax.legend(frameon=False)
    _style(ax)
    return _save(fig, out_dir, f'{gene}_delta_y_hist')


def plot_dfluor_histogram(all_pairs, good_pairs, gene, out_dir, bins=60):
    # Only positive fluorescence increases (drops are not plotted).
    all_pos = all_pairs.loc[all_pairs['delta_fluor'] > 0, 'delta_fluor'].dropna()
    good_pos = good_pairs.loc[good_pairs['delta_fluor'] > 0, 'delta_fluor'].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    hi = np.nanpercentile(all_pos, 99.5) if len(all_pos) else 1.0
    rng = (0, hi)
    ax.hist(all_pos, bins=bins, range=rng,
            color='#999999', alpha=0.6, label=f'all pairs (n={len(all_pos)})')
    ax.hist(good_pos, bins=bins, range=rng,
            color='#55A868', alpha=0.7,
            label=f'good pairs (n={len(good_pos)})')
    ax.set_yscale('log')
    ax.set_xlabel('Δfluor increase = intensity_mean_fluor(t+1) − (t),  > 0 only')
    ax.set_ylabel('pair count (log)')
    ax.set_title(f'{gene}: frame-to-frame fluorescence change')
    ax.legend(frameon=False)
    _style(ax)
    return _save(fig, out_dir, f'{gene}_delta_fluor_hist')


def plot_dy_vs_dfluor(pairs, bounds, gene, out_dir):
    # Only positive fluorescence increases (drops are not plotted).
    pos = pairs[pairs['delta_fluor'] > 0]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    good = pos[~pos['is_jump']]
    bad = pos[pos['is_jump']]
    ax.scatter(good['delta_y'], good['delta_fluor'], s=8, alpha=0.4,
               color='#55A868', label='good')
    ax.scatter(bad['delta_y'], bad['delta_fluor'], s=12, alpha=0.6,
               color='#C44E52', label='jump')
    for b in bounds:
        ax.axvline(b, color='#C44E52', ls='--', lw=1)
    ax.set_xlabel('Δy  [px]')
    ax.set_ylabel('Δfluor increase  (> 0 only)')
    ax.set_title(f'{gene}: Δy vs Δfluor increase per pair')
    ax.legend(frameon=False)
    _style(ax)
    return _save(fig, out_dir, f'{gene}_dy_vs_dfluor')


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_track_displacement_analysis(
    csv_path,
    gene='lacZ',
    out_dir='notebooks/figures',
    dy_k=4.0,
    dy_abs_cutoff=None,
    fluor_k=4.0,
    fluor_col='intensity_mean_fluor',
    bins=60,
    fluor_method='none',
    base_dir=DEFAULT_BASE_DIR,
    trackastra_dir=DEFAULT_TRACKASTRA_DIR,
    time_dict_path='data/time_dict_070826.json',
    filter_size=3,
    filter_sigma=1.0,
    nlm_h=1.15,
    otsu_scale=1.0,
    experiments=None,
):
    """Full analysis: load -> pairs -> flag jumps -> fluor change -> plots + CSVs.

    If `fluor_method` is a real filter (median/mean/gaussian/nlm/otsu), per-cell
    fluorescence is recomputed on the filtered images and swapped in before the
    fluorescence-change step. delta_y is unaffected.
    """
    filtering = fluor_method not in (None, 'none', 'raw')
    label = f'{gene}_{fluor_method}' if filtering else (gene or 'all')
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

    pairs, lower, upper = flag_jumps(pairs, k=dy_k, abs_cutoff=dy_abs_cutoff)
    n_jump = int(pairs['is_jump'].sum())
    print(f"  Δy cutoff = [{lower:.2f}, {upper:.2f}] px "
          f"({'abs' if dy_abs_cutoff is not None else f'median±{dy_k}·MAD'}); "
          f"{n_jump} pairs flagged as jumps ({100*n_jump/len(pairs):.1f}%)")

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
        cov = int(pairs['delta_fluor'].notna().sum())
        print(f"  Filtered fluor mapped onto {cov}/{len(pairs)} pairs "
              f"({100*cov/len(pairs):.1f}% coverage)")

    good = pairs[~pairs['is_jump']].copy()
    good = flag_significant_fluor(good, k=fluor_k)
    n_sig = int(good['is_significant'].sum())
    print(f"  {len(good)} good pairs; {n_sig} with significant Δfluor "
          f"(|z| >= {fluor_k})")

    # Figures
    f1 = plot_dy_histogram(pairs, (lower, upper), label, out_dir, bins=bins)
    f2 = plot_dfluor_histogram(pairs, good, label, out_dir, bins=bins)
    f3 = plot_dy_vs_dfluor(pairs, (lower, upper), label, out_dir)
    print(f"  Figures: {f1}\n           {f2}\n           {f3}")

    # Tables
    os.makedirs(out_dir, exist_ok=True)
    pairs_csv = os.path.join(out_dir, f'{label}_track_pairs.csv')
    pairs.to_csv(pairs_csv, index=False)
    sig = good[good['is_significant']].sort_values(
        'fluor_zscore', key=lambda s: s.abs(), ascending=False
    )
    sig_csv = os.path.join(out_dir, f'{label}_significant_fluor_pairs.csv')
    sig.to_csv(sig_csv, index=False)
    print(f"  Tables:  {pairs_csv}\n           {sig_csv}")

    return {
        'pairs': pairs,
        'good': good,
        'dy_bounds': (lower, upper),
        'n_jump': n_jump,
        'n_significant': n_sig,
    }
