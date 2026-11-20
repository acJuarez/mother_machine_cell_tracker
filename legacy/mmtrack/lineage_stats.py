# script use in and for functions that help determine the doubling time of cells in image data 
# uses time_elapsed_DICT to add minutes values, doubling time calculated based on how long it takes a cell
# to divide. Primarily accounts for cells in data that we see come from parent cells and consequently divide again 
# aims to use a lower bound when cells are not seen born in the image (i.e initial cells )

import numpy as np
import pandas as pd

# One trench, and one unique cell. Raw track_id repeats across trenches, so the trench keys
# are part of a cell's identity.
KYMO_KEY = ['experiment_name', 'FOV', 'trench_id']
CELL_KEY = KYMO_KEY + ['track_id']

# first and last observed frame per unique cell, becomes one row. 
def track_lifetimes(cells):

    return (cells.groupby(CELL_KEY, observed=True)
                 .agg(first_frame=('time_frame', 'min'),
                      last_frame=('time_frame', 'max'),
                      n_frames=('time_frame', 'nunique'),
                      minutes_per_frame=('minutes_per_frame', 'first'),
                      parent_track_id=('parent_track_id', 'first'))
                 .reset_index())

# creates a df where each row is one observed division, and tracks important information regarding that division
# such as gene, experiment, time elapsed, etc. Includes an 'is jump' flag from delta y analysis 
def division_events(pairs, cells):
    div = pairs[pairs['is_division'].astype(bool)]
    if div.empty:
        raise ValueError('no division pairs in this scope; nothing to average')

    # The lifetime definition below has no meaning if a track divides twice.
    per_parent = (div.groupby(KYMO_KEY + ['parent_track_id'], observed=True)['frame_from']
                     .nunique())
    assert (per_parent == 1).all(), 'a parent track divides at more than one frame'

    ev = (div.drop_duplicates(subset=KYMO_KEY + ['parent_track_id'])
             .loc[:, KYMO_KEY + ['gene', 'parent_track_id', 'frame_from', 'n_siblings']]
             .rename(columns={'parent_track_id': 'track_id',
                              'frame_from': 'division_frame',
                              'n_siblings': 'n_daughters'}))
    ev['track_id'] = ev['track_id'].astype(int)

    n_before = len(ev)
    ev = ev.merge(track_lifetimes(cells), on=CELL_KEY, how='inner')
    assert len(ev) == n_before, (
        f'{n_before - len(ev)} division events lost their parent rows in the cell table; '
        f'pass the unscoped cell table as `cells`')
    assert (ev['division_frame'] == ev['last_frame']).all(), (
        'a parent divides before its last observed frame -- the pair table and the cell '
        'table disagree about where the track ends')

    ev['doubling_min'] = (ev['last_frame'] - ev['first_frame']) * ev['minutes_per_frame']
    ev['birth_observed'] = ev['parent_track_id'].notna()

    # Jump granularity: the DIVIDING cell's own consecutive-frame pairs. A flagged division
    # pair (the birth of a daughter) does not disqualify the parent's lifetime -- daughters
    # are judged against a wider threshold precisely because a division displaces a centroid.
    wt = pairs[~pairs['is_division'].astype(bool)]
    flagged = set(wt.loc[wt['is_jump'].astype(bool), CELL_KEY]
                    .drop_duplicates().itertuples(index=False, name=None))
    ev['has_jump'] = [k in flagged for k in ev[CELL_KEY].itertuples(index=False, name=None)]
    return ev


def doubling_by(events, keys):
    """Doubling-time stats grouped by `keys`. n is DIVISIONS, one row per event."""
    g = events.groupby(keys, observed=True)['doubling_min']
    bo = events[events['birth_observed']].groupby(keys, observed=True)['doubling_min']
    out = pd.DataFrame({
        'n_div':               g.size(),
        'mean_dt_min':         g.mean(),
        'median_dt_min':       g.median(),
        'sd_dt_min':           g.std(),
        'sem_dt_min':          g.sem(),
        'n_div_birth_obs':     bo.size(),
        'mean_dt_birth_obs':   bo.mean(),
        'median_dt_birth_obs': bo.median(),
        'n_zero_dt':           events[events['doubling_min'] == 0]
                                   .groupby(keys, observed=True).size(),
    })
    for c in ('n_div_birth_obs', 'n_zero_dt'):
        out[c] = out[c].fillna(0).astype(int)
    out['pct_zero_dt'] = (100 * out['n_zero_dt'] / out['n_div']).round(1)
    return out


def _dt_total(events, label='TOTAL'):
    """The TOTAL row, recomputed over the whole frame rather than summed -- 10 of the 23
    experiments hold two genes in different FOVs, so summing a per-gene table would
    double-count them."""
    total = doubling_by(events.assign(_all=label), ['_all'])
    total.index = [label]
    return total


def doubling_by_experiment(events):
    """Table: one row per experiment, plus a recomputed TOTAL."""
    keys = ['experiment_name']
    out = doubling_by(events, keys)
    # '+'-joined because 10 of 23 experiments hold two genes in different FOVs.
    out.insert(0, 'gene', events.groupby(keys, observed=True)['gene']
                                .apply(lambda s: '+'.join(sorted(s.dropna().unique()))))
    out.insert(1, 'min_per_frame',
               events.groupby(keys, observed=True)['minutes_per_frame'].first())
    out = out.sort_values('mean_dt_min', ascending=False)

    total = _dt_total(events)
    total['gene'] = f"{events['gene'].nunique()} genes"
    total['min_per_frame'] = np.nan
    return pd.concat([out, total[out.columns]])


def doubling_by_gene(events):
    """Table: one row per gene, plus a recomputed TOTAL. Also returns the per-experiment
    means the `mean_of_exp_means` column and the figure's dots are built from."""
    out = doubling_by(events, ['gene'])
    per_exp = (events.groupby(['gene', 'experiment_name'], observed=True)
                     .agg(mean=('doubling_min', 'mean'),
                          n_div=('doubling_min', 'size'),
                          min_per_frame=('minutes_per_frame', 'first'))
                     .reset_index())

    out['n_experiments']     = per_exp.groupby('gene').size()
    out['mean_of_exp_means'] = per_exp.groupby('gene')['mean'].mean()
    out['min_exp_mean']      = per_exp.groupby('gene')['mean'].min()
    out['max_exp_mean']      = per_exp.groupby('gene')['mean'].max()

    cols = ['n_experiments', 'n_div', 'mean_dt_min', 'median_dt_min', 'sd_dt_min',
            'sem_dt_min', 'mean_of_exp_means', 'min_exp_mean', 'max_exp_mean',
            'n_div_birth_obs', 'mean_dt_birth_obs', 'median_dt_birth_obs',
            'n_zero_dt', 'pct_zero_dt']
    out = out[cols].sort_values('mean_dt_min', ascending=False)

    total = _dt_total(events)
    total['n_experiments']     = events['experiment_name'].nunique()
    total['mean_of_exp_means'] = per_exp['mean'].mean()
    total['min_exp_mean']      = per_exp['mean'].min()
    total['max_exp_mean']      = per_exp['mean'].max()
    return pd.concat([out, total[cols]]), per_exp


# bursts per cell cycle has a division count as part of the denominator here if the tracker 
# followed the cell through its initization and its division. kymographs with poor tracking 
# struggles to count divisions accurately 
# `doubling_min == (n_frames - 1) * minutes_per_frame` | n_frames == last_frame - first_frame + 1
# So `n_frames` counts # SAMPLES and `doubling_min` counts INTERVALS, and they differ by one frame per cycle.

#
# The cycle length wanted here is `mean(n_frames)`, not `mean(doubling_min) / mpf`:
#
#     100-frame trench, one cell slot, perfect tracking, division every 10 frames.
#     Cells occupy [0..9], [10..19], ... [90..99] -- ten complete cycles.
#     total cell-frames = 100, n_frames = 10, doubling_min / mpf = 9.
#         100 / mean(n_frames)     = 10.0   <- right
#         100 / mean(n_frames - 1) = 11.1   <- overcounts by 12%
#
# A parent on frames [f, l] divides AT l and its daughter's first frame is l + 1, so
# birth-to-birth is `n_frames` intervals. `n_frames` is therefore both the sample-consistent
# divisor for a cell-frame numerator AND the unbiased generation time -- `doubling_min`, the
# shipped T_d, is one interval short of a generation. `test_effective_cycles_identity` pins
# this; it fails at 11.1 under the other reading.
#
# THE NUMERATOR IS CELL-FRAMES, not trench time-frames. Measured, the two are not
# interchangeable: trenches hold 1.4-9.8 cells per frame (CV 0.38, p95/p05 = 3.97), so
# swapping them re-weights rather than rescales. `n_time_frames / frames_per_cycle` lands at
# 0.36x the raw division count -- it undercounts the observation ~2.8x because it counts the
# trench once instead of each cell in it. It is kept as a column, because "how many
# generations deep did the movie go" is a real question, but it is not the denominator.
#
# `minutes_per_frame` cancels between numerator and divisor, so the 5-min/10-min acquisition
# mix needs no normalising: measured, eff/n_div is 1.663 at 5 min and 1.726 at 10 min.

FULL_CYCLE_MIN_FRAMES = 2          # n_frames == 1 means born and divided in one frame

# just a function to clean up anf find all cell cycles where birth and division was seen
def full_cycles(divisions, require_no_jump=True):
    keep = divisions['birth_observed'].astype(bool)
    if require_no_jump:
        keep &= ~divisions['has_jump'].astype(bool)
    keep &= divisions['n_frames'] >= FULL_CYCLE_MIN_FRAMES
    return divisions[keep]


# returns a df of one row per trench with cycle length, observation effort?, and effective cell cycle
# a collection of cell cycles that is used downstream 
def cell_cycles_per_trench(divisions, cells, min_full_cycles=3, require_no_jump=True,
                           verbose=True):
    assert divisions['minutes_per_frame'].notna().all(), 'divisions lack minutes_per_frame'

    # Observation effort. nunique, never max-min+1: two trenches have interior missing frames
    # (008/1488 and 015/1508), so a span would overstate them.
    eff = (cells.groupby(KYMO_KEY, observed=True)
                .agg(gene=('gene', 'first'),
                     minutes_per_frame=('minutes_per_frame', 'first'),
                     n_cell_frames=('time_frame', 'size'),
                     n_time_frames=('time_frame', 'nunique'),
                     n_tracks=('track_id', 'nunique')))
    mpf_spread = cells.groupby(KYMO_KEY, observed=True)['minutes_per_frame'].nunique()
    assert (mpf_spread == 1).all(), (
        f'{int((mpf_spread != 1).sum())} trenches carry more than one minutes_per_frame; '
        'the frames<->minutes conversion below is per trench')

    extra = set(map(tuple, divisions[KYMO_KEY].drop_duplicates().values.tolist())) - set(eff.index)
    assert not extra, (
        f'{len(extra)} trenches have divisions but no rows in `cells` -- the two are scoped '
        f'differently. Pass load_cells(drop_failed=True). First: {sorted(extra)[:3]}')

    full = full_cycles(divisions, require_no_jump=require_no_jump)
    per_trench = full.groupby(KYMO_KEY, observed=True)['n_frames'].agg(
        n_full_cycles='size', frames_per_cycle='mean')

    out = eff.join(per_trench, how='left')
    out['n_divisions'] = (divisions.groupby(KYMO_KEY, observed=True).size()
                                   .reindex(out.index).fillna(0).astype(int))
    out['n_zero_dt'] = (divisions[divisions['n_frames'] < FULL_CYCLE_MIN_FRAMES]
                        .groupby(KYMO_KEY, observed=True).size()
                        .reindex(out.index).fillna(0).astype(int))
    out['n_full_cycles'] = out['n_full_cycles'].fillna(0).astype(int)

    # The fallback. A gene's median is taken over the trenches that DID clear the bar, so a
    # sparse trench borrows a measurement rather than its own noise.
    measured = out['n_full_cycles'] >= min_full_cycles
    gene_med = out.loc[measured].groupby('gene')['frames_per_cycle'].median()
    pooled = float(out.loc[measured, 'frames_per_cycle'].median())

    out['cycle_len_source'] = np.where(measured, 'trench', 'gene')
    borrowed = out['gene'].map(gene_med)
    out.loc[~measured, 'frames_per_cycle'] = borrowed[~measured]
    # A gene with no qualifying trench at all falls through to the pooled median; flagged
    # separately because it is a weaker claim than borrowing within a gene.
    still_nan = out['frames_per_cycle'].isna()
    out.loc[still_nan, 'frames_per_cycle'] = pooled
    out.loc[still_nan, 'cycle_len_source'] = 'pooled'

    out['min_per_cycle'] = out['frames_per_cycle'] * out['minutes_per_frame']
    out['cycles_per_frame'] = 1.0 / out['frames_per_cycle']
    out['mean_cells_per_frame'] = out['n_cell_frames'] / out['n_time_frames']
    out['eff_cycles'] = out['n_cell_frames'] * out['cycles_per_frame']
    out['eff_cycles_timeframes'] = out['n_time_frames'] * out['cycles_per_frame']
    out['eff_per_division'] = out['eff_cycles'] / out['n_divisions'].where(out['n_divisions'] > 0)

    cols = ['gene', 'minutes_per_frame', 'n_divisions', 'n_full_cycles', 'n_zero_dt',
            'frames_per_cycle', 'min_per_cycle', 'cycles_per_frame', 'cycle_len_source',
            'n_tracks', 'n_cell_frames', 'n_time_frames', 'mean_cells_per_frame',
            'eff_cycles', 'eff_cycles_timeframes', 'eff_per_division']
    out = out[cols]

    if verbose:
        src = out['cycle_len_source'].value_counts()
        print(f'{len(out)} trenches | cycle length from own trench: '
              f'{int(src.get("trench", 0))}, borrowed from gene: {int(src.get("gene", 0))}, '
              f'pooled: {int(src.get("pooled", 0))} '
              f'(threshold {min_full_cycles} full cycles)')
        print(f'  full cycles: {len(full):,} of {len(divisions):,} divisions '
              f'(birth_observed{" & ~has_jump" if require_no_jump else ""} & n_frames >= '
              f'{FULL_CYCLE_MIN_FRAMES}) | {int(out.n_zero_dt.sum())} single-frame parents '
              f'in the dataset, none of which can be a full cycle')
        print(f'  frames per cycle: median {out.frames_per_cycle.median():.2f} | '
              f'minutes per cycle: median {out.min_per_cycle.median():.1f}')
        print(f'  effective cycles {out.eff_cycles.sum():,.0f} vs '
              f'{int(out.n_divisions.sum()):,} observed divisions | '
              f'per-trench ratio median {out.eff_per_division.median():.3f}')
    return out

# eff_cycles is calculated here as 
def effective_cycles_by_gene(per_trench):
    """Per gene: the two denominators side by side, plus a recomputed TOTAL.

    `eff_cycles` is what `figures.bursts_per_cell_cycle(denom=...)` consumes. TOTAL is summed
    here rather than recomputed from a ratio, because both columns are extensive counts over
    disjoint trenches -- unlike the rate tables, where a ratio of sums is not a sum of ratios.
    """
    g = per_trench.groupby('gene', observed=True)
    out = pd.DataFrame({
        'n_trenches':      g.size(),
        'n_divisions':     g['n_divisions'].sum(),
        'eff_cycles':      g['eff_cycles'].sum(),
        'n_cell_frames':   g['n_cell_frames'].sum(),
        'median_min_per_cycle': g['min_per_cycle'].median(),
        'n_borrowed':      g['cycle_len_source'].apply(lambda s: int((s != 'trench').sum())),
    })
    out['eff_per_division'] = out['eff_cycles'] / out['n_divisions'].where(out['n_divisions'] > 0)
    out.loc['TOTAL'] = {
        'n_trenches':      len(per_trench),
        'n_divisions':     per_trench['n_divisions'].sum(),
        'eff_cycles':      per_trench['eff_cycles'].sum(),
        'n_cell_frames':   per_trench['n_cell_frames'].sum(),
        'median_min_per_cycle': per_trench['min_per_cycle'].median(),
        'n_borrowed':      int((per_trench['cycle_len_source'] != 'trench').sum()),
    }
    out.loc['TOTAL', 'eff_per_division'] = (out.loc['TOTAL', 'eff_cycles']
                                            / out.loc['TOTAL', 'n_divisions'])
    body = out.drop(index='TOTAL').sort_values('eff_cycles', ascending=False)
    return pd.concat([body, out.loc[['TOTAL']]])
