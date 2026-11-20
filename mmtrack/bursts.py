"""Burst detection: the three definitions (v1, v2, v3), the hand labels, and the build.

V1 = hand built rule with four different routes, incomplete and both missed and overclassified bursts that were not bursts 
V2 = a gradient boosted classifier trained from 2 subgen and 1 constitutive gene of data 
V3 = a monotone gradient-boosted classifier from annotated data of every kymograph availble excluding mazF and hupA
manual = manually annotated bursts used for training and validation 

v1's height bar is a (gene, experiment) median + MAD, so it is NOT local: after an exclusion v1 must be re-run, never filtered.

Every statistic in v2 and v3 is computed inside one kymograph, so excluding a trench changes
no other trench's result. 

"""

import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from mmtrack.experiment_metadata import FAILED_KYMOGRAPHS, STRAIN_EXP_DICT

# one kymograph
KYMO_KEY = ['experiment_name', 'FOV', 'trench_id']       

#one cell frame 
FRAME_KEY = KYMO_KEY + ['track_id', 'time_frame']        
# columns for image signal 
FLUOR_COL = 'fluor_per_ms'                               
F_RAW = 'intensity_mean_fluor'                          

# predicted lineage read in format of 1 , 1.1, 3.1.1 where each decimal is a daughter of cell the prior 
# FOV written in format of 00X and trenc_ids are just a numeric value 

READ_DTYPES = {'experiment_name': str, 'FOV': str, 'trench_id': str,
               'gene': str, 'predicted_lineage': str}

# file inputs
# all_data_annotated* = one row is a single cell within a single frame for all 23 experiments
# fiji_fluor_nospeckle = same as above but with FIHI clip + median filter, used by V2 and V3 
# annotated_bursts = all manually annotated bursts, frame number uses the viewer index
# time_dict = description of start and end for each kymograph 
#
INPUTS = {
    'cells': 'data/all_data_annotated_082026.csv',
    'fiji': 'data/fiji_fluor_nospeckle_090926.csv',
    'labels': 'data/annotated_bursts_100226.csv',
    'time_dict': 'data/time_dict_082026.json',
    'v2_model': 'models/burst_classifier_091026.joblib',
}
CELLS_CSV = INPUTS['cells']
TIME_DICT_JSON = INPUTS['time_dict']

OUTPUTS = {
    'v1': 'bursts_v1_100226.csv',
    'v2': 'bursts_v2_100226.csv',
    'v3': 'bursts_v3_100226.csv',
    'manual': 'bursts_manual_100226.csv',
    'unresolved_labels': 'burst_labels_recheck_100226.csv',
    'splits': 'burst_splits_100226.csv',
    'union': 'bursts_viewer_v1v2v3manual_100226.csv',
    'v3_model': 'burst_v3_100226.joblib',
}
BURSTS_CSV = os.path.join('data', OUTPUTS['union'])

EXPECTED = {'cells': 78931, 'trenches': 243, 'v1': 158, 'v2': 1041, 'v3': 1065,
            'manual': 837, 'unresolved_labels': 19, 'union': 1732,
            'v1_routes': {'absolute': 97, 'fold_left': 36, 'fold': 18, 'tukey': 7}}

REVIEWED_GENES = ('alkA', 'araB', 'baeS', 'chpS', 'gfcE', 'gldA', 'lacZ')
UNREVIEWED_GENES = ('hupA', 'mazF')


# loads data and drops kymographs that failed/wanted to exclude 
def drop_excluded_kymographs(df, name='table', excluded=FAILED_KYMOGRAPHS, verbose=True,
                             require_match=True):
    
    if not excluded or not len(df):
        return df
    key = list(zip(df['experiment_name'], df['FOV'], df['trench_id']))
    hit = pd.Series([k in set(excluded) for k in key], index=df.index)
    present = {k for k in key}
    if require_match:
        unmatched = [t for t in excluded if tuple(t) not in present]
        assert not unmatched, (
            f'{name}: these FAILED_KYMOGRAPHS entries match no trench ')
    if verbose and hit.any():
        print(f'  {name}: dropped {int(hit.sum())} rows in {len(excluded)} excluded '
              f'kymograph(s) (FAILED_KYMOGRAPHS)')
    return df[~hit].copy()


def _read_annotated(csv_path, name, drop_failed, data_root='', drop_excluded=True):

    df = pd.read_csv(os.path.join(data_root, csv_path), dtype=READ_DTYPES)
    assert df['FOV'].str.len().eq(3).all(), f'{name}: FOV lost its zero padding'
    assert 'is_failed_experiment' in df.columns, (
        f'{name}: no is_failed_experiment column -- this is not an annotated export')
    if drop_failed:
        df = df[~df['is_failed_experiment']].copy()
    if drop_excluded:
        df = drop_excluded_kymographs(df, name)
    return keep_tracked_genes(df, name)

# the genes of interest are exactly the keys of STRAIN_EXP_DICT; rows of any other gene
# (or of a FOV with no gene) are dropped at load, so the frozen CSVs need not be rewritten
GENES = tuple(STRAIN_EXP_DICT)


def keep_tracked_genes(df, name='table', verbose=True):
    keep = df['gene'].isin(GENES)
    if verbose and (~keep).any():
        print(f'  {name}: dropped {int((~keep).sum())} rows outside STRAIN_EXP_DICT')
    return df[keep].copy()


def load_divisions(path='data/divisions_082026.csv', data_root='', verbose=True):
    d = pd.read_csv(os.path.join(data_root, path), dtype=READ_DTYPES)
    d['FOV'] = d['FOV'].str.zfill(3)
    return keep_tracked_genes(d, 'divisions', verbose)


def load_cells(cells_csv=CELLS_CSV, drop_failed=True, data_root='', drop_excluded=True):
    return _read_annotated(cells_csv, 'cells', drop_failed, data_root, drop_excluded)

def clean_cells(data_root=''):
    c = load_cells(data_root=data_root)
    return c[~c['is_degenerate_mask']].reset_index(drop=True)

def load_fiji(path=INPUTS['fiji'], data_root=''):
    f = pd.read_csv(os.path.join(data_root, path), dtype=READ_DTYPES)
    f['FOV'] = f['FOV'].str.zfill(3)
    return f

#read a burst table written by build_all 
def load_bursts(bursts_csv=BURSTS_CSV, version=None, data_root='', expect_n=None):
    b = pd.read_csv(os.path.join(data_root, bursts_csv), dtype=READ_DTYPES)
    b['FOV'] = b['FOV'].str.zfill(3)
    b = keep_tracked_genes(b, bursts_csv)
    if version is not None:
        col = f'burst_{version}'
        if col not in b.columns:
            have = sorted(c for c in b.columns if c.startswith('burst_'))
            raise ValueError(f'{bursts_csv} has no {col!r} column; it carries {have}')
        b = b[b[col].astype(bool)].reset_index(drop=True)
    if expect_n is not None and len(b) != expect_n:
        raise AssertionError(f'expected {expect_n} {version!r} rows in {bursts_csv}, '
                             f'got {len(b)}')
    return b

def trench_index(cells):
    g = cells.groupby(KYMO_KEY, sort=True)
    out = pd.DataFrame({
        'gene': g['gene'].first(),
        'n_cell_frames': g.size(),
        'n_tracks': g['track_id'].nunique(),
        'n_time_frames': g['time_frame'].nunique(),
        'first_frame': g['time_frame'].min(),
        'last_frame': g['time_frame'].max(),
    }).reset_index()
    out['reviewed'] = out['gene'].isin(REVIEWED_GENES)
    return out

# for V1 calculates baseline and MAD scale thats then used for v1 burst detection. 
# baseline = median fluorescence 
# scale = 1.4826 * MAD (median absolute deviation) 
# all grouped by (gene, experiment), returns dataframe with median and mad_scale 
def compute_baselines(df, fluor_col=FLUOR_COL):
    rows = []
    for (gene, exp), g in df.groupby(['gene', 'experiment_name']):
        x = g[fluor_col].to_numpy()
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        rows.append({'gene': gene, 'experiment_name': exp,
                     'median': med, 'mad_scale': 1.4826 * mad})
    base = pd.DataFrame(rows).set_index(['gene', 'experiment_name'])
    base['mad_scale'] = base['mad_scale'].clip(lower=1e-6)
    return base


# brightest cell at each frame tracked across time. since a expression dilutes over time we should be able to see a peak
# and follow that decay. This identifies lineages by following the brightest cell at each frame 
def trench_max_envelope(df_trench, fluor_col=FLUOR_COL):
    idx = df_trench.groupby('time_frame')[fluor_col].idxmax()
    return df_trench.loc[idx].sort_values('time_frame').reset_index(drop=True)

# this time grouping by lineage and not just brightest cell assumption 
def lineage_family_roots(df_trench):
    par = df_trench.groupby('track_id')['parent_track_id'].first().to_dict()
    out = {}
    for t in par:
        cur, seen = int(t), set()
        while True:
            p = par.get(cur)
            if pd.isna(p) or int(p) not in par or int(p) in seen:
                break
            seen.add(cur)
            cur = int(p)
        out[int(t)] = cur
    return out

# input is parent cell, produces all the parent and daughter cells within that kymograph for given cell 
def _relatives(par):
    kids = {}
    for t, p in par.items():
        if p is not None and p in par:
            kids.setdefault(p, []).append(t)

    def walk(t, step, seen):
        out = set()
        stack = [t]
        while stack:
            cur = stack.pop()
            for nxt in step(cur):
                if nxt not in out and nxt != t:
                    out.add(nxt)
                    stack.append(nxt)
        return out

    up = lambda t: [par[t]] if par.get(t) in par else []
    down = lambda t: kids.get(t, [])
    return {t: walk(t, up, None) | walk(t, down, None) for t in par}

def _lineage_maps(g):
    idx = {(int(t), int(f)): i for i, (t, f) in
           enumerate(zip(g['track_id'].to_numpy(), g['time_frame'].to_numpy()))}
    par = (g.groupby('track_id')['parent_track_id'].first()
           .apply(lambda p: int(p) if pd.notna(p) else None).to_dict())
    kid = {}
    for t, p in par.items():
        if p is not None:
            kid.setdefault(p, []).append(int(t))
    return idx, par, kid

# a way to track daughter cells to their parent cell after a division 
def _lineage_lag(g, col, lag):
    idx, par, _ = _lineage_maps(g)
    v = g[col].to_numpy(dtype=float)
    out = np.full(len(g), np.nan)
    tr = g['track_id'].to_numpy().astype(int)
    fr = g['time_frame'].to_numpy().astype(int)
    for i in range(len(g)):
        t, f = int(tr[i]), int(fr[i])
        for _ in range(lag):
            f -= 1
            if (t, f) in idx:
                continue
            p = par.get(t)
            if p is None or (p, f) not in idx:
                t = None
                break
            t = p
        if t is not None and (t, f) in idx:
            out[i] = v[idx[(t, f)]]
    return out

def _lineage_window(g, col, k, how='max', direction='back'):
    """Aggregate `col` over the k frames before (or after) each cell-frame, along the lineage.

    Backward is the causal one and the interesting one: `x(t) / median(x(t-k..t-1))` is the
    per-cell-frame analogue of v1's `fold_left` route. Forward uses the MAX over daughters, matching the max-envelope
    convention the rest of the codebase uses for inherited signal.
    """
    idx, par, kid = _lineage_maps(g)
    v = g[col].to_numpy(dtype=float)
    tr = g['track_id'].to_numpy().astype(int)
    fr = g['time_frame'].to_numpy().astype(int)
    agg = {'max': np.nanmax, 'median': np.nanmedian, 'mean': np.nanmean,
           'min': np.nanmin}[how]
    out = np.full(len(g), np.nan)
    for i in range(len(g)):
        vals = []
        if direction == 'back':
            t, f = int(tr[i]), int(fr[i])
            for _ in range(k):
                f -= 1
                if (t, f) in idx:
                    vals.append(v[idx[(t, f)]]); continue
                p = par.get(t)
                if p is None or (p, f) not in idx:
                    break
                t = p
                vals.append(v[idx[(t, f)]])
        else:
            front = [(int(tr[i]), int(fr[i]))]
            for _ in range(k):
                nxt = []
                for t, f in front:
                    if (t, f + 1) in idx:
                        nxt.append((t, f + 1))
                    else:
                        nxt += [(c, f + 1) for c in kid.get(t, []) if (c, f + 1) in idx]
                if not nxt:
                    break
                vals += [v[idx[c]] for c in nxt]
                front = nxt
        if vals:
            with np.errstate(all='ignore'):
                out[i] = agg(vals)
    return out

# rule used to add time and spatial drift into bursts. Sometimes cell burst will drift over time to a different 
# y pixel value 
def _link_ty_capped(frames, ys, slack, drift, link_dt):
    """Connected components in (t, y) with a tolerance of `slack + drift * dt`.

    The superseded rules linked with a tolerance of `1.5 * cell_length * dt`
    -- a PRODUCT, so at dt=3 it reaches 4.5 cell lengths, most of a trench. That is fine for
    a sparse rule but not for a classifier, which lights up whole neighbourhoods: measured on
    `DUMM_giTG059_060_061125` 008/1488, it linked two cells 109.6 px apart (three cell
    lengths, opposite ends of the trench) because the tolerance at dt=2 was 110.2 px -- a
    margin of 0.6 px -- and silently absorbed two hand-labelled bursts into their
    neighbours' events. The trench went from 5 of 5 labels reported to 3 of 5.

    The product is also the wrong model of the physics. Cells advect about 4.7 px/frame
    (median; 17 at p10), so the y-displacement to tolerate over dt frames is a DRIFT term
    plus a fixed spatial slack, not a multiplication. Measured on the shipped model this is
    better on every axis -- 5/5 labels in that trench and event F1 0.731 against 0.716 --
    and out of fold the two are tied within noise (0.661 vs 0.667).
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
        while j < n and frames[j] - frames[i] <= link_dt:
            gap = frames[j] - frames[i]
            if gap != 0 and abs(ys[j] - ys[i]) <= slack + drift * gap:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
            j += 1
    return [find(i) for i in range(n)]

# loads cell-frame columns 
def make_row_key(exp, fov, trench, track_id, time_frame):
    """One text field identifying a cell-frame: exp|FOV|trench|track|time_frame.

    FOV is zero-filled to 3 here so a key built from a de-padded spreadsheet cell still
    matches one built from the tables.
    """
    return '|'.join([str(exp), str(fov).zfill(3), str(trench),
                         str(int(track_id)), str(int(time_frame))])

def _load_time_dict(time_dict=None, data_root=''):
    if time_dict is not None:
        return time_dict
    with open(os.path.join(data_root, TIME_DICT_JSON)) as fh:
        return json.load(fh)

# determines beginning and end time frame and returns None for end frames to go to the end of the frame 
def trench_window(exp, fov, trench, time_dict=None, data_root=''):
    td = _load_time_dict(time_dict, data_root)
    entry = td.get(exp, {}).get(str(fov).zfill(3), {}).get(str(trench))
    if entry is None:
        return None
    end = entry.get('end')
    return int(entry.get('start', 0) or 0), (None if end in (None, '', 0) else int(end))

# Labels dropped as probable mislabels, with the measurement that condemned each. All three
# sit below the 0.50 within-trench percentile and carry no signal on any available axis:
# fold-above-own-past medians 1.15 against a negative population whose p90 is 1.30, frame
# rank 0.22-0.50 (not even the brightest cell present), and 2 of 3 have fiji_mean exactly 0.
# The middle one is FALLING relative to its own recent past (fold 0.834), i.e. a decay tail
# or the wrong cell. Confirmed with the labeller before removal -- do not silently re-add.
KNOWN_MISLABELS = [
    ('DUMM_giTG062_064_121325', '015', '1235', 21, 19),
    ('DUMM_giTG064_Glucose_022625', '001', '1596', 19, 66),
    ('DUMM_giTG064_Glucose_022625', '015', '1584', 1, 17),
]

# fixes excel fill handle, can probably get rid of this 
def _repair_experiment_names(d, known, verbose=True):
    """Fix experiment names mangled by Excel's fill handle.

    Dragging a cell down in Excel increments a trailing number, so
    `DUMM_giTG059_060_061125` became `...061126`, `...061127`, `...061128`, `...061129` on
    consecutive rows. Rather than hard-coding that, this finds unknown names whose digits-only
    suffix differs from exactly one KNOWN experiment sharing the same non-digit prefix, and
    which also has the row's (FOV, trench) -- so a repair has to be corroborated by the data,
    not just by string similarity.
    """
    out = d.copy()
    fixes = {}
    for name in sorted(set(out['experiment_name']) - set(known)):
        stem = re.sub(r'\d+$', '', name)
        cand = [k for k in known if re.sub(r'\d+$', '', k) == stem]
        if len(cand) == 1:
            fixes[name] = cand[0]
    if fixes:
        out['experiment_name'] = out['experiment_name'].replace(fixes)
        if verbose:
            print(f'  repaired {len(fixes)} Excel fill-handle experiment name(s):')
            for k, v in fixes.items():
                print(f'    {k}  ->  {v}   ({int((d.experiment_name == k).sum())} row(s))')
    return out, fixes

# loads manually labeled bursts from csv 
def load_hand_labels(path=INPUTS['labels'], cells=None, data_root='', genes=None,
                     verbose=True, extra_cols=()):
    if isinstance(path, pd.DataFrame):
        full, d = 'labels DataFrame', path.astype(str).copy()
    else:
        full = os.path.join(data_root, path)
        d = pd.read_csv(full, dtype=str, keep_default_na=False)
    n_in = len(d)
    known = set(cells['experiment_name'].unique())
    d, _ = _repair_experiment_names(d, known, verbose=verbose)

    d['FOV'] = d['FOV'].str.strip().str.zfill(3)
    d['trench_id'] = d['trench_id'].str.strip()
    d['track_id'] = d['track_id'].astype(float).astype(int)
    d['viewer_frame'] = d['frame'].astype(float).astype(int)
    d['gene'] = d['gene'].str.strip()

    _n_before = len(d)
    d = drop_excluded_kymographs(d, 'hand labels', verbose=False, require_match=False)
    if verbose and len(d) < _n_before:
        print(f'  dropped {_n_before - len(d)} label(s) in an excluded kymograph '
              f'(experiment_metadata.FAILED_KYMOGRAPHS) -- not a labelling error')

    if genes is not None:
        keep_gene = d['gene'].isin(tuple(genes))
        if verbose and not keep_gene.all():
            dropped = d.loc[~keep_gene, 'gene'].value_counts().to_dict()
            print(f'  label CSV holds {n_in} rows; keeping genes {tuple(genes)}, '
                  f'setting aside {dropped}')
        d = d[keep_gene].reset_index(drop=True)
        n_in = len(d)

    starts, why = [], []
    for r in d.itertuples():
        w = trench_window(r.experiment_name, r.FOV, r.trench_id, data_root=data_root)
        starts.append(np.nan if w is None else w[0])
        why.append('' if w is not None else 'trench not in the time dict')
    d['start_frame'] = starts
    d['why'] = why
    d['time_frame'] = (d['viewer_frame'] - d['start_frame']).astype('Int64')

    c = cells[FRAME_KEY + ['gene']].copy()
    c['track_id'] = c['track_id'].astype(int)
    c['time_frame'] = c['time_frame'].astype(int)
    m = d.merge(c.rename(columns={'gene': '_gene_cells'}),
                left_on=KYMO_KEY + ['track_id', 'time_frame'],
                right_on=FRAME_KEY, how='left')

    ok = m[m['_gene_cells'].notna()].copy()
    bad = m[m['_gene_cells'].isna()].copy()

    # Diagnose each reject against what actually exists there, so the re-check is targeted.
    diag = []
    for r in bad.itertuples():
        if r.why:
            diag.append(r.why)
            continue
        sub = cells[(cells.experiment_name == r.experiment_name) & (cells.FOV == r.FOV)
                    & (cells.trench_id == r.trench_id)]
        if sub.empty:
            diag.append('no such trench in the cell table')
            continue
        at_frame = sorted(sub[sub.time_frame.astype(int) == r.time_frame].track_id.astype(int))
        of_track = sorted(sub[sub.track_id.astype(int) == r.track_id].time_frame.astype(int))
        if r.time_frame is pd.NA or r.time_frame < 0:
            diag.append(f'viewer frame {r.viewer_frame} is before the trench starts '
                        f'(start={int(r.start_frame)})')
        elif not of_track:
            diag.append(f'track {r.track_id} does not exist in this trench')
        elif not at_frame:
            diag.append(f'frame {r.time_frame} has no cells (trench ends earlier)')
        else:
            diag.append(f'track {r.track_id} exists only at frames {of_track[0]}-{of_track[-1]}, '
                        f'not {r.time_frame}; tracks present at that frame: {at_frame[:10]}')
    bad['diagnosis'] = diag

    # Drop the confirmed mislabels before anything downstream can train on them.
    _bad = set(KNOWN_MISLABELS)
    _is_bad = [(r.experiment_name, r.FOV, r.trench_id, int(r.track_id), int(r.time_frame))
               in _bad for r in ok.itertuples()]
    if any(_is_bad):
        if verbose:
            print(f'  dropped {sum(_is_bad)} confirmed mislabel(s) (see KNOWN_MISLABELS)')
        ok = ok[~pd.Series(_is_bad, index=ok.index)]

    ok['row_key'] = [make_row_key(r.experiment_name, r.FOV, r.trench_id, r.track_id,
                                  r.time_frame) for r in ok.itertuples()]
    # `extra_cols` carries further label-file columns through unchanged.
    keep = KYMO_KEY + ['track_id', 'time_frame', 'viewer_frame', 'gene', 'row_key'] + list(extra_cols)
    rej = bad[KYMO_KEY + ['track_id', 'viewer_frame', 'gene', 'diagnosis']].copy()
    rej.insert(0, 'time_frame_tried', bad['time_frame'].values)

    if verbose:
        print(f'{full}: {n_in} rows -> {len(ok)} resolved, {len(bad)} rejected')
        print(f'  genes: {ok.gene.value_counts().to_dict()}')
        print(f'  trenches with labels: {ok.groupby(KYMO_KEY).ngroups}')
        if len(bad):
            print(f'  REJECTED (re-check these in the viewer):')
            for r in rej.itertuples():
                print(f'    {r.gene} {r.experiment_name} {r.FOV}/{r.trench_id} '
                      f'track {r.track_id} viewer {r.viewer_frame}: {r.diagnosis}')
    return ok[keep].reset_index(drop=True), rej.reset_index(drop=True)


def build_manual(clean, labels):
    """The manual layer: the resolved labels plus geometry and the along-trench position."""
    _geo = [c for c in GEO_COLS if c in clean.columns]
    lab = labels.astype({'track_id': int, 'time_frame': int}).merge(
        clean.astype({'track_id': int, 'time_frame': int})[FRAME_KEY + _geo],
        on=FRAME_KEY, how='left', suffixes=('', '_cell'))
    assert lab['centroid_y'].notna().all(), 'a resolved label has no geometry'
    return add_relative_y(lab, clean).assign(burst_manual=True)



# V1 bursts: 
# A frame of a per-lineage-family x spatial-cluster max envelope is a burst when it is a local
# maximum (+/- `window` frames) admitted by ANY of four routes, tried in this order:
#
#   absolute   clears m + k*s AND has topographic prominence >= k_prom*s, where m and s are
#              the (gene, experiment) median and 1.4826*MAD of `fluor_col`
#   fold       peak / max(left_base, right_base) >= fold_prom
#   fold_left  peak / max(left_base, fold_base_floor_frac*m) >= fold_left_prom
#   tukey      peak > its own trench's Q3 + tukey_iqr_k*IQR
#
# Both ratio routes also require peak >= m + fold_height_floor_s*s; the tukey route does not
# (that is the whole point of it). The four are strictly nested, 98 -> 116 -> 153 -> 161, so
# each only ever ADDS events. Then an episode merge collapses a slow decay to one event and a
# concurrent merge collapses one burst seen in two adjacent lineage families.
# `bursts.md` section 3 has every constant's measured band.

FAMILY_SEP = '#'          #

BURST_DEF_V1 = {
    'version': 'v1',
    'fluor_col': FLUOR_COL,       # a.u./ms; detection is invariant to the exposure rescale
    'baseline_keys': ('gene', 'experiment_name'),   # the scope of m and s
    'window': 5,                  # +/- frames for the local-max check and peak separation
    'k': 4.0,                     # absolute route: height bar m + k*s
    'k_prom': 3.7,                # absolute route: prominence >= k_prom*s; plateau [3.6, 3.8]
    'fold_prom': 3.3,             # fold route; band (3.10, 3.60]
    'fold_height_floor_s': 1.0,   # both ratio routes: peak >= m + this*s
    'fold_left_prom': 3.6,        # fold_left route; band [3.4, 3.79]
    'fold_base_floor_frac': 0.5,  # fold_left denominator floored at this*m -- NOT optional
    'tukey_iqr_k': 4.0,           # tukey route multiplier; control-clean only at [4.4, 4.8]
    'merge_window': 5,            # concurrent merge: peaks this many frames apart ...
    'merge_gap_cell_lengths': 1.5,    # ... and this close in y are one burst
    'merge_keep': 'earliest',     # report the ONSET of a merged burst
    'split_gap_cell_lengths': 1.5,    # cut a lineage family where its cells are this far apart
    'constitutive_genes': ('mazF', 'hupA', 'chpS'),   # only tags `is_constitutive`
}

# Route precedence, strongest first. Picks an episode's representative peak and the survivor
# of a concurrent merge, so a ratio candidate on a rising shoulder can never absorb the
# confirmed bright peak behind it. 'tukey' ranks LAST, which is what lets that route ADD
# events without relabelling any the other three found.
ROUTE_RANK = {'absolute': 0, 'fold': 1, 'fold_left': 2, 'tukey': 3}


# main burst detection using V1 rules. For each experiment, FOV, trench pairing the brightest cell at each frame forms 
# one signal over time, then those candidate peaks are quantified as bursts based on the intensity of their signal and prominence 
# Then candidates are merged in a rolling window so signal decay is captured as one event instead of many. 
# returns DataFrame with one row per burst event: experiment_name, FOV, trench_id, track_id,
# predicted_lineage, node_id, time_frame, fluor, gene, threshold, route, is_constitutive.
# (track_id / node_id / predicted_lineage identify the brightest cell at the peak.
def detect_bursts(df, k=4.0, window=5, fluor_col=FLUOR_COL, constitutive_genes=None,
                  k_prom=None, fold_prom=None, fold_height_floor_s=1.0,
                  fold_left_prom=None, fold_base_floor_frac=0.5, fence_col=None):
    """
    Args:
        df: per-cell-per-frame DataFrame (needs gene, experiment_name, FOV, trench_id,
            track_id, time_frame, node_id, predicted_lineage, fluor_col).
        k: robust z-score multiplier for the absolute height threshold.
        window: +/- frames for the local-maximum check and minimum peak separation.
        fluor_col: fluorescence column (default a.u./ms).
        constitutive_genes: optional set; only tags an 'is_constitutive' column.
        k_prom: multiplier for the required prominence. None -> `k`.
        fold_prom: enables the FOLD route as a second way in -- a peak whose
            ``peak / max(left_base, right_base)`` reaches this ratio qualifies even when it
            never clears `height`. Unit-free, so one bar serves a 283x spread of gene
            brightness. None leaves the route off.
        fold_height_floor_s: on either ratio route, the peak must also sit this many
            MAD-scales above the gene median. Guards the ratio's denominator.
        fold_left_prom: enables the FOLD-LEFT route -- the same ratio taken against the
            LEFT base alone. ``max(left, right)`` binds on the right whenever a burst's decay
            has not returned to its pre-burst floor, which is the normal case here (the decay
            is dilution by growth, not protein loss), so the max-base rule systematically
            penalises long-decay events. None leaves this route off.
        fold_base_floor_frac: floors the fold-left denominator at this fraction of the
            baseline median. NOT optional: unfloored, a near-zero left base manufactures an
            arbitrary ratio (chpS 010/46 scores 30x on a left_base of 0.006). Bracketed on
            both sides at 0.5 -- 0.25 leaves the controls loose, 0.75 rises above the
            motivating case's own left base and loses it.
        fence_col: a column carrying each row's own TRENCH's Tukey fence (see
            `attach_kymograph_fence`); enables the TUKEY route. Read as a column because
            by now `trench_id` is a family x cluster subgroup, so a fence computed here
            would be per-family, not per-trench.

    The routes are OR'd, and every row carries `route` ('absolute', 'fold', 'fold_left' or
    'tukey') recording which admitted it. Neither ratio route substitutes for the absolute rule:
    measured, fold alone finds 67 events and fails all three regression trenches, because it
    misses the bright canonical bursts. They only ever add.
    """
    baselines = compute_baselines(df, fluor_col=fluor_col)
    constitutive_genes = constitutive_genes or set()
    kp = k if k_prom is None else k_prom
    records = []

    for (exp, fov, trench), df_trench in df.groupby(['experiment_name', 'FOV', 'trench_id']):
        gene = df_trench['gene'].iloc[0]
        if (gene, exp) not in baselines.index:
            continue
        med = baselines.loc[(gene, exp), 'median']
        scale = baselines.loc[(gene, exp), 'mad_scale']
        height = med + k * scale
        prom = kp * scale

        fence = None
        if fence_col is not None:
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
        cand = {int(pi): 'absolute' for pi in peaks
                if sig[pi] >= sig[max(0, pi - window):min(len(sig), pi + window + 1)].max()}

        if fold_prom is not None or fold_left_prom is not None or fence is not None:
            fpk, _ = find_peaks(sig, distance=window)
            if len(fpk):
                _, lb, rb = peak_prominences(sig, fpk)
                floor = med + fold_height_floor_s * scale
                
                den_floor = fold_base_floor_frac * med
                for i, pi in enumerate(fpk):
                    if sig[pi] < sig[max(0, pi - window):min(len(sig), pi + window + 1)].max():
                        continue
                    _tukey_hit = fence is not None and sig[pi] > fence
                    if sig[pi] < floor:
                        if _tukey_hit:
                            cand.setdefault(int(pi), 'tukey')
                        continue

                    if fold_prom is not None:
                        base = max(sig[lb[i]], sig[rb[i]])
                        if base > 0 and sig[pi] / base >= fold_prom:
                            cand.setdefault(int(pi), 'fold')
                    if fold_left_prom is not None:
                        base_l = max(sig[lb[i]], den_floor)
                        if base_l > 0 and sig[pi] / base_l >= fold_left_prom:
                            cand.setdefault(int(pi), 'fold_left')
                    if _tukey_hit:
                        cand.setdefault(int(pi), 'tukey')

        episodes, cur_peak, last_i, reset = [], -np.inf, None, height
        for pi in sorted(cand):
            route = cand[pi]
            if (last_i is None                              
                    or sig[last_i:pi + 1].min() < reset     
                    or sig[pi] > cur_peak + prom):          
                episodes.append([])
                cur_peak = sig[pi]
                if route == 'tukey':
                    reset = fence
                else:
                    _div = {'fold': fold_prom, 'fold_left': fold_left_prom}.get(route)
                    reset = height if _div is None else sig[pi] / _div
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

# creates lineage_family column to a cell table 
def add_lineage_family(df):
    """Add a `lineage_family` column (root track_id within the trench) to a cell table."""
    df = df.sort_values(KYMO_KEY + ['track_id', 'time_frame']).reset_index(drop=True)
    fam = np.empty(len(df), dtype=object)
    for _, g in df.groupby(KYMO_KEY, observed=True, sort=False):
        roots = lineage_family_roots(g)
        fam[g.index] = [roots[int(t)] for t in g['track_id']]
    df['lineage_family'] = fam
    return df

# merges two bursts if tracking is wrong and two different lineages overcount the same burst 
# determines this if track_ids are not too far apart (~1.5 cell length apart)
# determines earliest onset of burst as burst 

def merge_concurrent_bursts(bursts, bdef=None):
    bdef = BURST_DEF_V1 if bdef is None else bdef
    W = bdef['merge_window']
    GAP = bdef['merge_gap_cell_lengths']
    if bdef.get('merge_keep', 'earliest') != 'earliest':
        raise ValueError("only merge_keep='earliest' is implemented")

    drop, audit = set(), []
    for _, g in bursts.groupby(KYMO_KEY, observed=True):
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
    bdef = BURST_DEF_V1 if bdef is None else bdef
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

# runs the Tukey fence for detect_bursts 
# This is a quartile fence rather than a location + scale bar 
# Returns a copy of `df` with `fence_col` added.

def attach_kymograph_fence(df, fluor_col=FLUOR_COL, iqr_k=4.0, fence_col='_tukey_fence'):
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


def run_v1(clean, bdef=BURST_DEF_V1, verbose=True):
    """v1's events on `clean` (see `clean_cells`). Returns `(events, concurrent_merge_audit)`.

    Attaches each trench's Tukey fence, partitions every trench into lineage family x spatial
    cluster, detects on each partition's max envelope by encoding the partition into
    `trench_id` (which is what `detect_bursts` groups by), restores `trench_id`, runs the
    concurrent merge, and adds the along-trench position.

    RE-RUN this after any change to the cell table, never filter its output: the height bar
    is a (gene, experiment) median + MAD, so dropping one trench moves the bar for every
    surviving trench of that gene in that experiment (excluding gfcE 001/2501 cost one event
    in ANOTHER trench).
    """
    assert tuple(bdef['baseline_keys']) == ('gene', 'experiment_name'), (
        'compute_baselines groups by (gene, experiment_name); change the function, not the dict')
    fenced = attach_kymograph_fence(clean, fluor_col=bdef['fluor_col'],
                                    iqr_k=bdef['tukey_iqr_k'], fence_col='_tukey_fence')
    cells = add_spatial_subgroup(add_lineage_family(fenced), bdef)
    keyed = cells.copy()
    keyed['trench_id'] = (keyed['trench_id'].astype(str) + FAMILY_SEP
                          + keyed['spatial_group'].astype(str))
    b = detect_bursts(keyed, k=bdef['k'], k_prom=bdef['k_prom'], window=bdef['window'],
                      fluor_col=bdef['fluor_col'],
                      constitutive_genes=set(bdef['constitutive_genes']),
                      fold_prom=bdef['fold_prom'],
                      fold_height_floor_s=bdef['fold_height_floor_s'],
                      fold_left_prom=bdef['fold_left_prom'],
                      fold_base_floor_frac=bdef['fold_base_floor_frac'],
                      fence_col='_tukey_fence')
    split = b['trench_id'].str.rsplit(FAMILY_SEP, n=1, expand=True)
    b['trench_id'] = split[0]
    b['partition'] = split[1]
    b['lineage_family'] = b['partition'].astype(str).str.split('.').str[0].astype(int)

    # Position and size at the peak, for the merge. centroid_y is the true along-trench axis;
    # centroid_x is kymograph-space and must never be used for distance.
    b = b.merge(cells[FRAME_KEY + ['centroid_y', 'axis_major_length']], on=FRAME_KEY,
                how='left')
    assert b['centroid_y'].notna().all(), 'a burst peak failed to match a cell row'
    b, audit = merge_concurrent_bursts(b, bdef)
    b['def_version'] = bdef['version']
    b['baseline_scope'] = 'gene_experiment'
    b = add_relative_y(b.reset_index(drop=True), clean)
    if verbose:
        print(f'v1: {len(b)} events | routes {b["route"].value_counts().to_dict()}')
    return b.assign(burst_v1=True), audit



# V2: normalizes just within one kymograph and uses manually labled bursts to train a 
# gradient hist classifier to predict other bursts, also using fiji modified values 

FLUOR_COLS = ('fiji_mean', 'intensity_mean_fluor')
FOLD_FLOOR_FRAC = 0.01

def _safe_ratio(a, b, floor):
    return np.asarray(a, dtype=float) / np.maximum(np.asarray(b, dtype=float), floor)

# creates features from a dataframe of one row per cell-frame
# returns a copy with the feature columns added 
def featurize(df, verbose=True):
    d = df.copy()
    d['track_id'] = d['track_id'].astype(int)
    d['time_frame'] = d['time_frame'].astype(int)
    d = d.sort_values(KYMO_KEY + ['track_id', 'time_frame']).reset_index(drop=True)
    g = d.groupby(KYMO_KEY, sort=False)
    feats = []

    #within-trench normalisation: the block that already scores AUC ~0.95 
    for c in FLUOR_COLS:
        d[f'{c}__pct'] = g[c].rank(pct=True)
        tmax = g[c].transform('max')
        d[f'{c}__over_tmax'] = _safe_ratio(d[c], tmax, 1e-9)
        lit = d[c].where(d['fiji_mean'] > 0)
        lmed = lit.groupby([d[k] for k in KYMO_KEY]).transform('median')
        lmad = (lit - lmed).abs().groupby([d[k] for k in KYMO_KEY]).transform('median')
        d[f'{c}__over_litmed'] = _safe_ratio(d[c], lmed, 1e-6)
        d[f'{c}__z_lit'] = _safe_ratio(d[c] - lmed, 1.4826 * lmad, 1e-6)
        feats += [f'{c}__pct', f'{c}__over_tmax', f'{c}__over_litmed', f'{c}__z_lit']
    # percentile among LIT values only
    d['fiji_mean__lit_pct'] = (d['fiji_mean'].where(d['fiji_mean'] > 0)
                               .groupby([d[k] for k in KYMO_KEY]).rank(pct=True)).fillna(0.0)
    feats.append('fiji_mean__lit_pct')

    #B. raw values and shape
    d['fiji_max_over_mean'] = _safe_ratio(d['fiji_max'], d['fiji_mean'], 1e-6)
    for c in ('fiji_mean', 'fiji_frac', 'fluor_per_ms', 'area', 'axis_major_length',
              'fiji_max_over_mean'):
        if c in d.columns:
            feats.append(c)

    #C. temporal, along the lineage
    parts = []
    for (_, _, _), gg in d.groupby(KYMO_KEY, sort=False):
        sub = {}
        for c in FLUOR_COLS:
            for lag in (1, 2, 3):
                sub[f'{c}__lag{lag}'] = _lineage_lag(gg, c, lag)
            sub[f'{c}__bmax3'] = _lineage_window(gg, c, 3, 'max', 'back')
            sub[f'{c}__bmed3'] = _lineage_window(gg, c, 3, 'median', 'back')
            sub[f'{c}__bmed5'] = _lineage_window(gg, c, 5, 'median', 'back')
            sub[f'{c}__fmax3'] = _lineage_window(gg, c, 3, 'max', 'fwd')
        parts.append(pd.DataFrame(sub, index=gg.index))
    T = pd.concat(parts).sort_index()
    for c in FLUOR_COLS:
        scale_c = g[c].transform('max').to_numpy()
        # rise: how far above its own recent past, as a fraction of the trench's range
        d[f'{c}__rise1'] = (d[c].to_numpy() - T[f'{c}__lag1'].to_numpy()) / np.maximum(scale_c, 1e-9)
        d[f'{c}__rise3'] = (d[c].to_numpy() - T[f'{c}__bmax3'].to_numpy()) / np.maximum(scale_c, 1e-9)
        # Unit-free fold against the trailing median -- the per-cell-frame `fold_left`.
        # The denominator is floored at 5% of the TRENCH'S OWN scale, not at an epsilon.
        # This is v1's `fold_base_floor_frac` lesson, and it is not optional: an eps floor
        # lets a near-zero trailing median manufacture arbitrary fold (a depth-3 tree fitted
        # on the eps version split at fold_back3 <= 3179, which is noise, not signal --
        # exactly how chpS 010/46 scored 30x on a base of 0.006 before it was floored).
        floor = FOLD_FLOOR_FRAC * np.maximum(scale_c, 1e-9)
        d[f'{c}__fold_back3'] = np.asarray(d[c], dtype=float) / np.maximum(
            T[f'{c}__bmed3'].to_numpy(), floor)
        d[f'{c}__fold_back5'] = np.asarray(d[c], dtype=float) / np.maximum(
            T[f'{c}__bmed5'].to_numpy(), floor)
        # asymmetry: a burst rises fast and decays slowly, constitutive signal does neither
        d[f'{c}__asym'] = ((d[c].to_numpy() - T[f'{c}__bmax3'].to_numpy())
                           - (T[f'{c}__fmax3'].to_numpy() - d[c].to_numpy())) / np.maximum(scale_c, 1e-9)
        feats += [f'{c}__rise1', f'{c}__rise3', f'{c}__fold_back3', f'{c}__fold_back5',
                  f'{c}__asym']

    # within-frame context
    fg = d.groupby(KYMO_KEY + ['time_frame'], sort=False)
    for c in FLUOR_COLS:
        d[f'{c}__frame_pct'] = fg[c].rank(pct=True)
        d[f'{c}__over_frame_max'] = _safe_ratio(d[c], fg[c].transform('max'), 1e-9)
        feats += [f'{c}__frame_pct', f'{c}__over_frame_max']
    d['n_cells_in_frame'] = fg['track_id'].transform('size')
    d['cell_index_from_open_end'] = fg['centroid_y'].rank(method='first') - 1
    feats += ['n_cells_in_frame', 'cell_index_from_open_end']

    # E. trench context, and ABSOLUTE magnitude
    # Every feature above is RELATIVE, 
    d['trench_lit_frac'] = g['fiji_mean'].transform(lambda x: (x > 0).mean())
    d['trench_n_frames'] = g['time_frame'].transform('nunique')
    d['trench_fiji_max'] = g['fiji_mean'].transform('max')
    d['trench_fiji_litn'] = g['fiji_mean'].transform(lambda x: (x > 0).sum())
    d['trench_raw_max'] = g['intensity_mean_fluor'].transform('max')
    d['trench_raw_med'] = g['intensity_mean_fluor'].transform('median')
    feats += ['trench_lit_frac', 'trench_n_frames', 'trench_fiji_max', 'trench_fiji_litn',
              'trench_raw_max', 'trench_raw_med']
    for c in ('intensity_mean_fluor', 'intensity_max_fluor', 'fiji_max'):
        if c in d.columns and c not in feats:
            feats.append(c)

    #F. cell-cycle position
    tg = d.groupby(KYMO_KEY + ['track_id'], sort=False)
    d['frames_since_birth'] = d['time_frame'] - tg['time_frame'].transform('min')
    d['frames_to_track_end'] = tg['time_frame'].transform('max') - d['time_frame']
    d['track_n_frames'] = tg['time_frame'].transform('nunique')
    feats += ['frames_since_birth', 'frames_to_track_end', 'track_n_frames']

    d.attrs['feature_cols'] = feats
    if verbose:
        print(f'featurize: {len(d):,} cell-frames x {len(feats)} features')
    return d


# Event linking: one burst and its decay tail become one event, reported at its highest-scoring
# frame.
LINK_DT = 3                     # frames
LINK_SLACK_CELL_LENGTHS = 1.5   # fixed y slack, in cell lengths
LINK_DRIFT_PX = 10.0            # ... plus this much per frame of gap


def score_to_events(df, threshold, link_dt=LINK_DT, slack=LINK_SLACK_CELL_LENGTHS,
                    drift=LINK_DRIFT_PX):
    """Collapse scored cell-frames into events, reported at the highest-scoring frame.

    A burst and its decay tail become one event; two bursts at opposite ends of a trench stay
    two. See `_link_ty_capped` for the tolerance.

    `df` needs `score`, `centroid_y`, `axis_major_length` and the cell key.
    """
    out = []
    hot = df[df['score'] >= threshold]
    for (e, f, t), g in hot.groupby(KYMO_KEY, sort=False):
        L = float(df[(df.experiment_name == e) & (df.FOV == f)
                     & (df.trench_id == t)]['axis_major_length'].median())
        L = L if np.isfinite(L) and L > 0 else 1.0
        gg = g.sort_values('time_frame').reset_index(drop=True)
        gg['_ev'] = _link_ty_capped(gg['time_frame'].to_numpy(),
                                    gg['centroid_y'].to_numpy(), slack * L, drift, link_dt)
        for _, ev in gg.groupby('_ev', sort=False):
            pk = ev.loc[ev['score'].idxmax()]
            out.append({'experiment_name': e, 'FOV': f, 'trench_id': t,
                        'track_id': int(pk['track_id']), 'time_frame': int(pk['time_frame']),
                        'score': float(pk['score']), 'gene': pk.get('gene'),
                        'n_cell_frames': int(len(ev)),
                        'n_frames': int(ev['time_frame'].nunique()),
                        'first_frame': int(ev['time_frame'].min()),
                        'last_frame': int(ev['time_frame'].max())})
    cols = ['experiment_name', 'FOV', 'trench_id', 'track_id', 'time_frame', 'score', 'gene',
            'n_cell_frames', 'n_frames', 'first_frame', 'last_frame']
    return pd.DataFrame(out, columns=cols)


def run_v2(clean, fiji, model_path=INPUTS['v2_model'], verbose=True):
    """v2's events: featurize every cell-frame, score with the SHIPPED model, link at 0.75.

    The model is LOADED, never refitted. It was fitted on 203 lacZ/gfcE hand labels from
    `data/bursts_lacZ_gfcE_091026.csv` (60 exhaustively reviewed trenches, decay tail masked
    at +-8 frames, `StratifiedGroupKFold` by trench, threshold 0.75 = max out-of-fold F1)
    BEFORE three of those labels were found to be mislabels (`KNOWN_MISLABELS`) and before
    two trenches were excluded (`FAILED_KYMOGRAPHS`). A refit today trains on different rows
    and would not reproduce these 1,041 events, so the joblib is a frozen input.

    Scored on `clean`, which already lacks the excluded trenches; because every feature is
    within-trench that gives exactly the events of scoring everything and dropping them
    afterwards.
    """
    import joblib
    M = joblib.load(model_path)
    X = featurize(clean.merge(fiji, on=FRAME_KEY, how='left'), verbose=False)
    X['score'] = M['model'].predict_proba(X[M['feature_cols']].to_numpy(dtype=float))[:, 1]
    ev = score_to_events(X, M['threshold'])
    ev.insert(5, 'burst_v2', True)
    if verbose:
        print(f'v2: {len(ev)} events at threshold {M["threshold"]} '
              f'({len(M["feature_cols"])} features, {model_path})')
    return ev, X


# V3: similar to V2 is a trained classifier but uses more training data this time around 
# No trench is scored by a model that saw it, an attempt was made to make features of model 
# something that could be explained by data/physical phenomena. Also uses fiji_mean values 
# instead of raw values 

F_FIJI = 't_fiji'             # v3's name for `fiji_mean` (it was read from an identical copy)
V3_MASK_WINDOW = 3            # +-frames of decay tail dropped from the loss; nearly inert
V3_THRESHOLD = 0.90           # best out-of-fold event F1 on dev
V3_DRIFT = 10.0               # px/frame in the event linker (median drift 6.56, p90 18.55)
V3_SEED = 0
SPLIT_SEED = 20261002
SPLIT_TEST_FRAC = 0.25
DECAY_K = 5                   # forward horizon, in frames, for the DECAY block

 
def make_splits(tindex, labels, test_frac=0.25, seed=20261002, n_tries=4000):
    """Hold out whole trenches, stratified by gene, ~`test_frac` of each gene's LABELS.

    Returns `tindex` plus `split` ('dev'/'test') and `n_labels`. Only reviewed-gene trenches
    are split; mazF/hupA get `split = 'unreviewable'` and are in neither.

    WHOLE TRENCHES, because cell-frames in a trench share a baseline, a clip floor and usually
    a lineage -- a random cell-frame split trains and tests on the same burst. The GNN's
    time-based split does exactly that and was deliberately not copied.

    Stratified on LABELS rather than trenches, because trenches carry 1-13 labels each and a
    trench-count split would put wildly different numbers of positives on the two sides. The
    search is a randomised one: `n_tries` shuffled greedy fills per gene, keeping the fill
    whose test label share lands closest to `test_frac`. A single shuffle overshoots badly on
    the small genes -- lacZ's target is 17.5 labels against trenches holding up to 13.

    Certified-empty trenches (reviewed gene, zero labels) are split in the same proportion so
    the holdout can measure false positives on confirmed-empty ground.
    """
    out = tindex.copy()
    n_lab = labels.groupby(KYMO_KEY).size()
    out['n_labels'] = (out.set_index(KYMO_KEY).index.map(n_lab)
                       .to_series(index=out.index).fillna(0).astype(int))
    out['split'] = 'unreviewable'

    rng = np.random.default_rng(seed)
    for gene in REVIEWED_GENES:
        sub = out[(out['gene'] == gene) & out['reviewed']]
        lab_tr = sub[sub['n_labels'] > 0]
        emp_tr = sub[sub['n_labels'] == 0]

        target = test_frac * lab_tr['n_labels'].sum()
        best, best_err = None, np.inf
        order_pool = lab_tr.index.to_numpy()
        counts = lab_tr['n_labels'].to_numpy()
        pos = {ix: i for i, ix in enumerate(order_pool)}
        for _ in range(n_tries):
            order = rng.permutation(order_pool)
            tot, pick = 0, []
            for ix in order:
                if tot >= target:
                    break
                pick.append(ix)
                tot += counts[pos[ix]]
            err = abs(tot - target)
            if err < best_err:
                best, best_err = list(pick), err
        out.loc[best, 'split'] = 'test'
        out.loc[[i for i in lab_tr.index if i not in set(best)], 'split'] = 'dev'

        n_emp_test = int(round(test_frac * len(emp_tr)))
        if len(emp_tr):
            etest = rng.choice(emp_tr.index.to_numpy(), size=n_emp_test, replace=False)
            out.loc[emp_tr.index, 'split'] = 'dev'
            out.loc[etest, 'split'] = 'test'

    return out

def _lineage_forward_trace(g, fcol, acol, k):
    """Walk FORWARD along the lineage and return, per row, two traces of length `k`.

    `fmax[i, j]` is the maximum of `fcol` over every descendant alive `j+1` frames later;
    `asum[i, j]` is the SUM of `acol` over those same cells. NaN once the frontier dies.

    The two aggregators are different ON PURPOSE and it is the whole physical content of the
    DECAY block:

      * fluorescence is a CONCENTRATION, so the right summary across daughters is the MAXIMUM
        -- if either daughter is still bright, the signal has not gone away. A mean would be
        diluted by however many cells happen to be present, which is the same argument v1 uses
        for its max envelope (`bursts.md` section 3.1).
      * area is an EXTENSIVE quantity, so the right summary is the SUM. A dividing cell's two
        daughters together occupy about what the parent did, so the sum is the volume the
        original protein is now spread through.

    Together they let the fall be separated from the dilution that growth alone would cause:
    if C = N/V with N fixed, then d(ln C)/dt = -d(ln V)/dt exactly. A burst that falls only at
    that rate is one synthesis event being spread out; one that falls faster is losing protein.
    Nothing in v1 or v2 measures this.
    """
    idx, par, kid = _lineage_maps(g)
    fv = g[fcol].to_numpy(dtype=float)
    av = g[acol].to_numpy(dtype=float)
    tr = g['track_id'].to_numpy().astype(int)
    fr = g['time_frame'].to_numpy().astype(int)
    n = len(g)
    fmax = np.full((n, k), np.nan)
    asum = np.full((n, k), np.nan)
    for i in range(n):
        front = [(int(tr[i]), int(fr[i]))]
        for j in range(k):
            nxt = []
            for t, f in front:
                if (t, f + 1) in idx:
                    nxt.append((t, f + 1))
                else:
                    nxt += [(c, f + 1) for c in kid.get(t, []) if (c, f + 1) in idx]
            if not nxt:
                break
            cells = [idx[c] for c in nxt]
            fmax[i, j] = np.nanmax(fv[cells])
            asum[i, j] = np.nansum(av[cells])
            front = nxt
    return fmax, asum

def _loglin_slope(y):
    """Least-squares slope of ln(y) against t, over the finite, strictly positive entries.

    `y` is (n, m) with t = 0, 1, ... along axis 1. Returns (n,) of slopes, NaN where fewer
    than 3 usable points remain. Working in logs is what makes the two slopes comparable:
    an exponential decay and an exponential growth become two numbers on the same scale, and
    their sum is the excess that is not explained by dilution.
    """
    n, m = y.shape
    t = np.arange(m, dtype=float)
    out = np.full(n, np.nan)
    with np.errstate(all='ignore'):
        ly = np.log(np.where(y > 0, y, np.nan))
    for i in range(n):
        ok = np.isfinite(ly[i])
        if ok.sum() < 3:
            continue
        tt, yy = t[ok], ly[i][ok]
        tm, ym = tt.mean(), yy.mean()
        den = float(((tt - tm) ** 2).sum())
        if den <= 0:
            continue
        out[i] = float(((tt - tm) * (yy - ym)).sum() / den)
    return out

# featurization for V3, tried to be grouped into physically-named blocks 
# normalization should be within trench instead of globally 
# this can include how bright a cell is relatyive to its trench, Rise, decay, neighbor, shape, and cycle 
def build_features_v3(d, k_decay=DECAY_K, verbose=True):
    
    d = d.sort_values(KYMO_KEY + ['track_id', 'time_frame']).reset_index(drop=True).copy()
    g = d.groupby(KYMO_KEY, sort=False)
    blocks = {b: [] for b in ('LEVEL', 'RISE', 'DECAY', 'NEIGHBOUR', 'SHAPE', 'CYCLE')}

    def add(block, name, values):
        d[name] = values
        blocks[block].append(name)

    # per-trench anchors 
    tmax_F = g[F_RAW].transform('max')
    tmed_F = g[F_RAW].transform('median')
    tmad_F = g[F_RAW].transform(lambda s: np.median(np.abs(s - np.median(s))))
    scale_F = np.maximum(1.4826 * tmad_F, 1e-9)
    tmax_J = g[F_FIJI].transform('max')
    floor_F = FOLD_FLOOR_FRAC * np.maximum(tmax_F, 1e-9)
    floor_J = FOLD_FLOOR_FRAC * np.maximum(tmax_J, 1e-9)

    # LEVEL 
    add('LEVEL', 'F__pct', g[F_RAW].rank(pct=True))
    add('LEVEL', 'F__over_tmax', d[F_RAW] / np.maximum(tmax_F, 1e-9))
    add('LEVEL', 'F__over_tmed', d[F_RAW] / np.maximum(tmed_F, floor_F))
    add('LEVEL', 'F__z_trench', (d[F_RAW] - tmed_F) / scale_F)
    add('LEVEL', 'J__pct', g[F_FIJI].rank(pct=True))
    add('LEVEL', 'J__over_tmax', d[F_FIJI] / np.maximum(tmax_J, 1e-9))
    # absolute magnitude the only features carrying a scale
    add('LEVEL', 'F_abs', d[F_RAW])
    add('LEVEL', 'trench_F_max', tmax_F)
    add('LEVEL', 'trench_F_med', tmed_F)
    add('LEVEL', 'trench_J_max', tmax_J)

    # RISE and DECAY: the lineage walks 
    n = len(d)
    cols = {c: np.full(n, np.nan) for c in (
        'F_lag1', 'F_bmax3', 'F_bmed3', 'F_bmed5', 'J_bmed3',
        'F_fmax1', 'F_fmax3', 'F_fmax5', 'F_fmed3', 'slope_f', 'slope_a')}
    for key, idxs in g.indices.items():
        sub = d.iloc[idxs]
        cols['F_lag1'][idxs] = _lineage_lag(sub, F_RAW, 1)
        cols['F_bmax3'][idxs] = _lineage_window(sub, F_RAW, 3, 'max', 'back')
        cols['F_bmed3'][idxs] = _lineage_window(sub, F_RAW, 3, 'median', 'back')
        cols['F_bmed5'][idxs] = _lineage_window(sub, F_RAW, 5, 'median', 'back')
        cols['J_bmed3'][idxs] = _lineage_window(sub, F_FIJI, 3, 'median', 'back')
        fmax, asum = _lineage_forward_trace(sub, F_RAW, 'area', k_decay)
        cols['F_fmax1'][idxs] = fmax[:, 0]
        cols['F_fmax3'][idxs] = np.nanmax(fmax[:, :3], axis=1)
        cols['F_fmax5'][idxs] = np.nanmax(fmax, axis=1)
        cols['F_fmed3'][idxs] = np.nanmedian(fmax[:, :3], axis=1)
        # the two log-slopes, anchored on the cell itself at t = 0
        f_ser = np.column_stack([sub[F_RAW].to_numpy(dtype=float), fmax])
        a_ser = np.column_stack([sub['area'].to_numpy(dtype=float), asum])
        cols['slope_f'][idxs] = _loglin_slope(f_ser)
        cols['slope_a'][idxs] = _loglin_slope(a_ser)
    if verbose:
        print(f'  lineage walks done over {len(g.indices)} trenches')

    x = d[F_RAW].to_numpy(dtype=float)
    add('RISE', 'F__rise1', (x - cols['F_lag1']) / scale_F)
    add('RISE', 'F__rise3', (x - cols['F_bmax3']) / scale_F)
    add('RISE', 'F__fold_back3', x / np.maximum(cols['F_bmed3'], floor_F))
    add('RISE', 'F__fold_back5', x / np.maximum(cols['F_bmed5'], floor_F))
    add('RISE', 'J__fold_back3', d[F_FIJI] / np.maximum(cols['J_bmed3'], floor_J))

    add('DECAY', 'F__drop1', (x - cols['F_fmax1']) / scale_F)
    add('DECAY', 'F__drop3', (x - cols['F_fmax3']) / scale_F)
    add('DECAY', 'F__drop5', (x - cols['F_fmax5']) / scale_F)
    add('DECAY', 'F__fold_fwd3', x / np.maximum(cols['F_fmed3'], floor_F))
    add('DECAY', 'F__asym3', ((x - cols['F_bmax3']) - (cols['F_fmax3'] - x)) / scale_F)
    # the physics: C = N/V, so pure dilution gives d(ln C)/dt = -d(ln V)/dt exactly.
    add('DECAY', 'decay_rate', -cols['slope_f'])
    add('DECAY', 'dilution_rate', cols['slope_a'])
    add('DECAY', 'decay_excess', -cols['slope_f'] - cols['slope_a'])

    # NEIGHBOUR 
    fg = d.groupby(KYMO_KEY + ['time_frame'], sort=False)
    nif = fg[F_RAW].transform('size')
    add('NEIGHBOUR', 'F__frame_pct', fg[F_RAW].rank(pct=True))
    add('NEIGHBOUR', 'J__frame_pct', fg[F_FIJI].rank(pct=True))
    add('NEIGHBOUR', 'F__over_frame_max', x / np.maximum(fg[F_RAW].transform('max'), 1e-9))
    # leave-one-out MEDIAN, not mean: with a mean aggregator the leave-one-out baseline is an
    # exact rescale of the frame mean and not a new statistic (measured r = 0.995).
    loo_med = fg[F_RAW].transform(
        lambda s: s.median() if len(s) > 1 else np.nan)
    add('NEIGHBOUR', 'F__over_frame_med', x / np.maximum(loo_med, floor_F))
    dd = d.sort_values(KYMO_KEY + ['time_frame', 'centroid_y'])
    ag = dd.groupby(KYMO_KEY + ['time_frame'], sort=False)[F_RAW]
    adj = ((ag.shift(1) + ag.shift(-1)) / 2).combine_first(
        ag.shift(1)).combine_first(ag.shift(-1)).reindex(d.index)
    add('NEIGHBOUR', 'F__over_adjacent', x / np.maximum(adj.to_numpy(dtype=float), floor_F))
    add('NEIGHBOUR', 'n_cells_in_frame', nif)

    # SHAPE 
    y_lo = g['centroid_y'].transform('min')
    y_hi = g['centroid_y'].transform('max')
    add('SHAPE', 'area', d['area'])
    add('SHAPE', 'axis_major_length', d['axis_major_length'])
    add('SHAPE', 'axis_minor_length', d['axis_minor_length'])
    add('SHAPE', 'centroid_y_rel',
        (d['centroid_y'] - y_lo) / np.maximum(y_hi - y_lo, 1e-9))
    add('SHAPE', 'cell_index_from_open_end',
        fg['centroid_y'].rank(method='first') - 1)

    # CYCLE 
    tg = d.groupby(KYMO_KEY + ['track_id'], sort=False)['time_frame']
    first, last = tg.transform('min'), tg.transform('max')
    add('CYCLE', 'frames_since_birth', d['time_frame'] - first)
    add('CYCLE', 'frames_to_track_end', last - d['time_frame'])
    add('CYCLE', 'track_n_frames', last - first + 1)
    add('CYCLE', 'frac_through_track',
        (d['time_frame'] - first) / np.maximum(last - first, 1))

    feats = [c for b in blocks.values() for c in b]
    d.attrs['blocks'] = blocks
    d.attrs['feature_cols'] = feats
    if verbose:
        print(f'  {len(feats)} features in {len(blocks)} blocks: '
              f'{ {b: len(v) for b, v in blocks.items()} }')
    return d


# Physically-defensible sign for every feature, for the monotone model (sklearn
# `monotonic_cst`: +1 increasing, -1 decreasing, 0 unconstrained).
# +1 means "more of this makes it MORE burst-like" and every one of them is a claim about the
# physics 
MONOTONE_SIGNS = {
    'F__pct': 1, 'F__over_tmax': 1, 'F__over_tmed': 1, 'F__z_trench': 1,
    'J__pct': 1, 'J__over_tmax': 1,
    'F_abs': 1, 'trench_F_max': 1, 'trench_F_med': 0, 'trench_J_max': 1,
    'F__rise1': 1, 'F__rise3': 1, 'F__fold_back3': 1, 'F__fold_back5': 1,
    'J__fold_back3': 1,
    'F__drop1': 1, 'F__drop3': 1, 'F__drop5': 1, 'F__fold_fwd3': 1, 'F__asym3': 1,
    'decay_rate': 1, 'dilution_rate': 0, 'decay_excess': 1,
    'F__frame_pct': 1, 'J__frame_pct': 1, 'F__over_frame_max': 1,
    'F__over_frame_med': 1, 'F__over_adjacent': 1, 'n_cells_in_frame': 0,
    'area': 0, 'axis_major_length': 0, 'axis_minor_length': 0,
    'centroid_y_rel': 0, 'cell_index_from_open_end': 0,
    'frames_since_birth': 0, 'frames_to_track_end': 0, 'track_n_frames': 0,
    'frac_through_track': 0,
}

V3_MODEL_KW = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20,
                   l2_regularization=1.0, class_weight='balanced')


def make_model_v3(seed=0, monotone=None, feature_cols=None, **kw):
    """`HistGradientBoostingClassifier`, optionally with a monotone constraint per feature.

    sklearn's `monotonic_cst` takes +1/-1/0 in FEATURE ORDER, so it is built from
    `MONOTONE_SIGNS` 
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    kws = {**V3_MODEL_KW, **kw}
    if monotone:
        missing = [c for c in feature_cols if c not in MONOTONE_SIGNS]
        if missing:
            raise ValueError(f'no monotone sign declared for {missing}')
        kws['monotonic_cst'] = [MONOTONE_SIGNS[c] for c in feature_cols]
    return HistGradientBoostingClassifier(random_state=seed, **kws)


def cv_score_all(tt, feature_cols, n_splits=5, seed=0, shuffle_y=False, monotone=False,
                 use_weights=False):
    """Out-of-fold scores for EVERY row, grouped by trench. Returns `(scored, auc, ap)`.

    Two things are deliberate:

      * it TRAINS on `sample_weight > 0` rows but PREDICTS on every row of the held-out
        trenches, including the decay-tail-masked ones. Masked rows are excluded from the
        LOSS because their label is unknown, not because they do not exist -- leaving them
        unscored punches holes in the middle of exactly the episodes being detected, and the
        event linker then splits one burst into two.
      * the group check is asserted on the actual split, per fold.

    GROUPING BY TRENCH IS NOT OPTIONAL. Cell-frames in a trench share a baseline, a clip floor
    and usually a lineage, so a random split trains and tests on the same burst. The GNN's
    split is time-based and puts every trench in both halves; it was deliberately not copied.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    tt = tt.reset_index(drop=True)
    fit_mask = (tt['sample_weight'] > 0).to_numpy()
    y = tt['y'].to_numpy().astype(int)
    if shuffle_y:
        rng = np.random.default_rng(seed)
        y = y.copy()
        y[fit_mask] = rng.permutation(y[fit_mask])
    groups = tt['cv_group'].to_numpy()
    X = tt[feature_cols]

    oof = np.full(len(tt), np.nan)
    fit_idx = np.flatnonzero(fit_mask)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(fit_idx, y[fit_idx], groups[fit_idx]):
        tr_i = fit_idx[tr]
        tr_groups = set(groups[tr_i])
        te_groups = set(groups[fit_idx[te]])
        assert not (tr_groups & te_groups), 'a trench leaked across the split'
        m = make_model_v3(seed=seed, monotone=monotone, feature_cols=feature_cols)
        # `sample_weight` is used as a 0/1 mask; callers may pass graded weights instead
        sw = tt['sample_weight'].to_numpy(dtype=float)[tr_i] if use_weights else None
        m.fit(X.iloc[tr_i], y[tr_i], sample_weight=sw)
        # predict on EVERY row of the held-out trenches, masked or not
        pred_i = np.flatnonzero(np.isin(groups, list(te_groups)))
        oof[pred_i] = m.predict_proba(X.iloc[pred_i])[:, 1]

    # Score against `y`, which is the SHUFFLED vector when shuffle_y is set -- not against
    # tt['y']. Scoring a noise-trained model against the TRUE labels measures anti-correlation
    # and reports an AUC well below 0.5, which looks like a finding and is an artifact.
    tt = tt.assign(score=oof)
    keep = fit_mask & np.isfinite(oof)
    return (tt, float(roc_auc_score(y[keep], oof[keep])),
            float(average_precision_score(y[keep], oof[keep])))

def build_training_table(cells, labels, reviewed, mask_window=3, verbose=True):
    """Attach `y`, `sample_weight` and `cv_group` to every cell-frame in `reviewed` trenches.

    y == 1   a labelled burst peak
    y == 0   a confirmed negative -- reviewed trench, outside every decay mask
    weight 0 inside a decay mask: within `mask_window` frames of a positive, on that
             positive's own track OR any ancestor/descendant of it in the same trench.
             These rows are neither positive nor negative and must not enter the loss.

    `mask_window=0` disables masking and is almost certainly wrong: the labeller's own note is
    that neighbouring frames still show signal because the decay is dilution by growth. Sweep
    it; do not assume it.

    `cv_group` is the trench. Every split downstream must group by it -- cell-frames within a
    trench share a baseline, a clip floor and usually a lineage, so a random split leaks badly.
    (The GNN's existing time-based split puts the same trench in train and test; do not copy it.)
    """
    keys = cells[KYMO_KEY].apply(tuple, axis=1)
    pool = cells[keys.isin(reviewed)].copy()
    pool['track_id'] = pool['track_id'].astype(int)
    pool['time_frame'] = pool['time_frame'].astype(int)

    pos = set(map(tuple, labels[FRAME_KEY].astype({'track_id': int, 'time_frame': int}).values))
    pool['y'] = [tuple(r) in pos for r in
                 pool[FRAME_KEY].astype({'track_id': int, 'time_frame': int}).values]
    pool['y'] = pool['y'].astype(int)

    mask = np.zeros(len(pool), dtype=bool)
    if mask_window:
        idx = {k: i for i, k in enumerate(map(tuple, pool[FRAME_KEY].values))}
        for (exp, fov, tr), g in pool.groupby(KYMO_KEY, sort=False):
            lab = labels[(labels.experiment_name == exp) & (labels.FOV == fov)
                         & (labels.trench_id == tr)]
            if lab.empty:
                continue
            par = (g.groupby('track_id')['parent_track_id'].first()
                   .apply(lambda p: int(p) if pd.notna(p) else None).to_dict())
            rel = _relatives(par)
            for r in lab.itertuples():
                fam = {int(r.track_id)} | rel.get(int(r.track_id), set())
                lo, hi = r.time_frame - mask_window, r.time_frame + mask_window
                sel = g[(g.track_id.isin(fam)) & (g.time_frame.between(lo, hi))]
                for k in map(tuple, sel[FRAME_KEY].values):
                    mask[idx[k]] = True
    pool['in_decay_mask'] = mask
    pool['sample_weight'] = np.where(pool['y'] == 1, 1.0,
                                     np.where(pool['in_decay_mask'], 0.0, 1.0))
    pool['cv_group'] = pool[KYMO_KEY].apply(lambda r: '|'.join(r), axis=1)

    if verbose:
        n_use = int((pool.sample_weight > 0).sum())
        print(f'training table: {len(pool):,} cell-frames over {len(reviewed)} reviewed '
              f'trenches ({pool.gene.nunique()} genes, {pool.experiment_name.nunique()} exps)')
        print(f'  positives {int(pool.y.sum())}  ({100*pool.y.mean():.2f}%)')
        print(f'  decay-masked (weight 0, mask_window={mask_window}): '
              f'{int((pool.sample_weight == 0).sum())} extra rows')
        print(f'  usable in the loss: {n_use:,}  '
              f'({int(pool.y.sum())} pos / {n_use - int(pool.y.sum())} neg, '
              f'1:{(n_use - int(pool.y.sum())) / max(1, int(pool.y.sum())):.0f})')
    return pool

def flags_to_events(d, value_col, flag, link_dt=3, slack=1.5, drift=10.0, L_by=None,
                    report='peak'):
    """Collapse flagged cell-frames into events, one row per connected component.

    The tolerance is a SUM, `slack * cell_length + drift * dt`, via
    `_link_ty_capped` -- never the superseded product `1.5 * L * dt`, which
    reaches 4.5 cell lengths at dt=3 and was measured merging two cells 109.6 px apart.
    `L` is the trench's median `axis_major_length` over ALL its cell-frames, not just the
    flagged ones, so the tolerance does not shrink as the threshold rises.

    Equivalent to `score_to_events` but driven by a boolean rather than a
    threshold on a `score` column, and with `L` precomputed -- that function re-scans the
    whole table once per trench, which is too slow inside a threshold sweep.

    `report` selects WHICH cell-frame of a component represents the event: `'peak'` the
    highest `value_col`, `'onset'` the earliest frame. THE TWO MUST NEVER BE QUOTED AS ONE.
    The same events reported at the peak instead of the onset once moved an agreement
    number from 13/156 to 89/156. v1 reports onset (its concurrent merge keeps the EARLIER peak); v2 and
    v3 report the peak.
    """
    cols = KYMO_KEY + ['track_id', 'time_frame', 'gene', value_col]
    hot = d.loc[flag]
    if not len(hot):
        return pd.DataFrame(columns=cols)
    if L_by is None:
        L_by = d.groupby(KYMO_KEY)['axis_major_length'].median()
    out = []
    for key, g in hot.groupby(KYMO_KEY, sort=False):
        g = g.sort_values('time_frame')
        L = float(L_by.get(key, np.nan))
        if not np.isfinite(L) or L <= 0:
            L = 1.0
        comp = _link_ty_capped(g['time_frame'].astype(int).tolist(),
                               g['centroid_y'].astype(float).tolist(),
                               slack * L, drift, link_dt)
        g = g.assign(_c=comp)
        pick = (g.groupby('_c')[value_col].idxmax() if report == 'peak'
                else g.groupby('_c')['time_frame'].idxmin())
        out.append(g.loc[pick, cols])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def run_v3(clean, fiji, labels, model_out=None, verbose=True):
    """Freeze the splits, build the features, fit, score, and link v3's events.

    Returns `(events, splits, model_bundle)`. With `model_out` the bundle is also saved
    there with joblib. Deterministic: the splits are seeded (`SPLIT_SEED`), as are the CV
    folds and the boosting (`V3_SEED`), so a rebuild reproduces
    `models/burst_v3_100226.joblib` prediction-for-prediction.
    """
    import joblib

    splits = make_splits(trench_index(clean), labels, test_frac=SPLIT_TEST_FRAC,
                         seed=SPLIT_SEED)
    dev = set(map(tuple, splits.loc[splits['split'] == 'dev', KYMO_KEY].values))
    lab_dev = labels[[tuple(r) in dev for r in labels[KYMO_KEY].values]].copy()

    d = clean.merge(fiji[FRAME_KEY + ['fiji_mean']].rename(columns={'fiji_mean': F_FIJI}),
                    on=FRAME_KEY, how='inner')
    assert len(d) == len(clean), f'the FIJI join lost rows: {len(clean)} -> {len(d)}'
    feat = build_features_v3(d, verbose=verbose)
    FEATS = feat.attrs['feature_cols']

    # out-of-fold scores for every dev cell-frame, then the final model on all of dev
    tt = build_training_table(feat, lab_dev, reviewed=dev, mask_window=V3_MASK_WINDOW,
                              verbose=False)
    oof, auc, ap = cv_score_all(tt, FEATS, seed=V3_SEED, monotone=True)
    ttr = tt.reset_index(drop=True)
    fit_idx = np.flatnonzero((ttr['sample_weight'] > 0).to_numpy())
    model = make_model_v3(seed=V3_SEED, monotone=True, feature_cols=FEATS)
    model.fit(ttr.loc[fit_idx, FEATS], ttr['y'].to_numpy().astype(int)[fit_idx])
    bundle = {'model': model, 'feature_cols': list(FEATS), 'blocks': feat.attrs['blocks'],
              'threshold': V3_THRESHOLD, 'monotone': True,
              'monotone_signs': {c: MONOTONE_SIGNS[c] for c in FEATS},
              'mask_window': V3_MASK_WINDOW, 'drift_px': V3_DRIFT,
              'link_dt': LINK_DT, 'slack_cell_lengths': LINK_SLACK_CELL_LENGTHS,
              'seed': V3_SEED, 'trained_on': f'dev split of data/{OUTPUTS["splits"]}',
              'n_labels': int(len(lab_dev))}
    if model_out is not None:
        os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
        joblib.dump(bundle, model_out)

    # every cell-frame gets the final model's score; dev cell-frames are then overwritten
    # with their out-of-fold score, so no trench is scored by a model that saw it
    feat['score'] = model.predict_proba(feat[FEATS])[:, 1]
    k_oof = dict(zip(map(tuple, oof[FRAME_KEY].values), oof['score']))
    feat['score'] = [k_oof.get(k, s) for k, s in
                     zip(map(tuple, feat[FRAME_KEY].values), feat['score'])]

    L_by = feat.groupby(KYMO_KEY)['axis_major_length'].median()
    ev = flags_to_events(feat, 'score', feat['score'] >= V3_THRESHOLD, link_dt=LINK_DT,
                         slack=LINK_SLACK_CELL_LENGTHS, drift=V3_DRIFT, L_by=L_by)
    geo = [c for c in GEO_COLS if c in clean.columns and c != 'gene']
    ev = ev.astype({'track_id': int, 'time_frame': int}).merge(
        clean.astype({'track_id': int, 'time_frame': int})[FRAME_KEY + geo],
        on=FRAME_KEY, how='left')
    assert ev['centroid_y'].notna().all(), 'a v3 event has no geometry'
    ev = add_relative_y(ev, clean).assign(burst_v3=True)
    if verbose:
        print(f'v3: {len(ev)} events | dev out-of-fold AUC {auc:.4f} PR-AUC {ap:.4f} | '
              f'{len(dev)} dev trenches, {len(lab_dev)} dev labels')
    return ev, splits, bundle


# Scoring events against labels

def match_events(events, labels, tol=1, tol_col=None):
    """Maximum-cardinality one-to-one matching of events to labels, per trench, on time only.

    `tol` is +-frames; with `tol_col` each label carries its own tolerance (for example its
    `duration`). Returns `(tp, matched_label_index, matched_event_index)` with the input
    frames' index labels. One-to-one matters: "any label within tol" lets one event claim
    several labels and inflates precision and recall at once.
    """
    from scipy.optimize import linear_sum_assignment
    lb_by = dict(list(labels.groupby(KYMO_KEY, sort=False)))
    li, ei = [], []
    for key, e in events.groupby(KYMO_KEY, sort=False):
        lab = lb_by.get(key)
        if lab is None:
            continue
        d = np.abs(e['time_frame'].to_numpy(int)[:, None]
                   - lab['time_frame'].to_numpy(int)[None, :]).astype(float)
        t = (lab[tol_col].to_numpy(float)[None, :] if tol_col else np.full((1, len(lab)), tol))
        ok = d <= t
        if ok.any():
            cost = np.where(ok, d, float(t.max()) * min(d.shape) + 1.0)   # cardinality first
            ri, ci = linear_sum_assignment(cost)
            hit = ok[ri, ci]
            ei += list(e.index[ri[hit]])
            li += list(lab.index[ci[hit]])
    return len(li), li, ei


def match_one_to_one(events, labels, tol=1):
    """The matched count only -- what `test_bursts.py` pins."""
    return match_events(events, labels, tol=tol)[0]


def prf(tp, n_events, n_labels):
    p = tp / n_events if n_events else np.nan
    r = tp / n_labels if n_labels else np.nan
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# Event tables: position along the trench, and the viewer union

# helps attach a burst to the nearest cell (based on centroid-y pixel values)
def add_relative_y(events, cells, track_col='track_id', frame_col='time_frame',
                   y_col='centroid_y'):
    """

    ``centroid_y_rel``              0 at the OPEN end, 1 at the closed end. Denominator is
                                    this trench's own occupied extent (see below), so it
                                    maps straight onto the viewer's y-axis as a fraction.
    ``cell_index_from_open_end``    0-based rank of this cell among the cells present in
                                    that frame, sorted by `centroid_y`. Denominator-free,
                                    so it survives a bad extent estimate.
    ``n_cells_in_frame``            how many cells that rank is out of -- "cell 3 of 8".

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
        track_col, frame_col, y_col: the event table's names for those three.

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

GEO_COLS = ['gene', 'centroid_x', 'centroid_y', 'axis_major_length',
            'axis_minor_length', 'area']

# writes out the bursts for V1-V3 into one CSV to be used downstream 
# time frame is the 0-based index within the trimmed stack (must not be offset by the time_dict start 
# hmtl viewer uses start only for printed tick labeskl 
def build_burst_csv(burst_sets, cells, path, raw_csv=None):
    
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

    carry = [c for c in ('fluor', 'threshold', 'route', 'baseline_scope')
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


# The build
def _check(got, want, what):
    assert got == want, f'{what}: expected {want}, got {got}'
    print(f'  OK  {what}: {got}')


def build_all(out_dir='data', models_dir='models', data_root='', verbose=True):
    """Regenerate every OUTPUT from the frozen INPUTS, asserting the pinned counts.

    Steps, in order: the clean cell table -> the hand labels and the manual layer -> v1 ->
    v2 -> v3 (splits, features, fit, score) -> the four-layer viewer union.
    Returns {name: DataFrame}.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = lambda k: os.path.join(out_dir, OUTPUTS[k])

    print('--- inputs ---')
    clean = clean_cells(data_root=data_root)
    _check(len(clean), EXPECTED['cells'], 'clean cell-frames')
    _check(clean.groupby(KYMO_KEY).ngroups, EXPECTED['trenches'], 'trenches')
    fiji = load_fiji(data_root=data_root)

    print('--- manual (hand labels) ---')
    labels, rejects = load_hand_labels(cells=clean, data_root=data_root, verbose=False)
    manual = build_manual(clean, labels)
    _check(len(manual), EXPECTED['manual'], 'resolved labels')
    _check(len(rejects), EXPECTED['unresolved_labels'], 'unresolvable labels')

    print('--- v1 ---')
    v1, _ = run_v1(clean, verbose=verbose)
    _check(len(v1), EXPECTED['v1'], 'v1 events')
    _check(v1['route'].value_counts().to_dict(), EXPECTED['v1_routes'], 'v1 routes')

    print('--- v2 ---')
    v2, _ = run_v2(clean, fiji, model_path=os.path.join(data_root, INPUTS['v2_model']),
                   verbose=verbose)
    _check(len(v2), EXPECTED['v2'], 'v2 events')

    print('--- v3 ---')
    v3, splits, _ = run_v3(clean, fiji, labels, verbose=verbose,
                           model_out=os.path.join(models_dir, OUTPUTS['v3_model']))
    _check(len(v3), EXPECTED['v3'], 'v3 events')
    _check(splits['split'].value_counts().to_dict(),
           {'dev': 167, 'test': 57, 'unreviewable': 19}, 'splits')

    print('--- writing ---')
    v1.to_csv(path('v1'), index=False)
    v2.to_csv(path('v2'), index=False)
    v3.to_csv(path('v3'), index=False)
    manual.to_csv(path('manual'), index=False)
    rejects.to_csv(path('unresolved_labels'), index=False)
    (splits[KYMO_KEY + ['gene', 'split', 'n_labels', 'n_cell_frames']]
     .sort_values(['gene', 'split'] + KYMO_KEY).to_csv(path('splits'), index=False))
    # v1 first so shared rows report v1's values in any carried column
    union = build_burst_csv({'v1': v1, 'v2': v2, 'v3': v3, 'manual': manual}, clean,
                            path('union'))
    _check(len(union), EXPECTED['union'], 'union rows')
    for k in ('v1', 'v2', 'v3', 'manual', 'unresolved_labels', 'splits'):
        print(f'  wrote {path(k)}')
    print(f'  wrote {os.path.join(models_dir, OUTPUTS["v3_model"])}')
    return {'clean': clean, 'labels': labels, 'manual': manual, 'v1': v1, 'v2': v2,
            'v3': v3, 'splits': splits, 'union': union}
