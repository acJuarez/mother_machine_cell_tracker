"""v4: v2's 43 features and unconstrained gradient boosting, refitted on the 110926 labels
with v3's whole-trench dev/test split and out-of-fold scoring.

Parked. Not part of `build_all`, not a viewer layer, not in the figures notebook. Run by
`legacy/build_v4.py`, which needs the repo root on `sys.path` and imports the shared halves
of the burst pipeline from `mmtrack.bursts`.

The 110926 labels carry four columns no model can take as input, because they exist only on
labelled frames while the model scores every cell-frame. They shape training and evaluation
instead: Confidence becomes the positive's sample weight, Duration the per-label decay mask
(the labelled frame may be any of the D frames the burst lasts), Notes decide whether the mask
follows the lineage into daughters, Intensity is evaluation only.
"""
import numpy as np
import pandas as pd

from mmtrack.bursts import (
    FLUOR_COL, FLUOR_COLS, FOLD_FLOOR_FRAC, FRAME_KEY, GEO_COLS, KYMO_KEY, LINK_DRIFT_PX,
    LINK_DT, LINK_SLACK_CELL_LENGTHS, SPLIT_SEED, SPLIT_TEST_FRAC, _safe_ratio,
    add_relative_y, cv_score_all, featurize, flags_to_events, load_hand_labels, make_splits,
    match_events, match_one_to_one, prf, trench_index,
)

# V4: v2's recipe (its 43 `featurize` features, unconstrained gradient boosting) refitted on
# the re-annotated labels in `bursts_annotated_110926.csv`, with v3's discipline (whole-trench
# dev/test split, out-of-fold scoring, so no trench is scored by a model that saw it).
#
# The 110926 file adds four columns per label. None can be a model INPUT -- they exist only
# on labelled frames, and the model has to score every cell-frame -- so they shape training
# and evaluation instead:
#   Confidence   -> the positive's sample weight (`V4_CONF_WEIGHTS`)
#   Duration     -> the per-label decay mask: +-D frames, because the labelled frame may be
#                   ANY of the D frames the burst lasts (the labeller's convention)
#   Notes        -> the mask follows the lineage into daughters ONLY when the note says the
#                   signal carries into division; otherwise tracking is not trusted
#   Intensity    -> evaluation only (recall per tier)
# v4 is built by `build_v4.py`, NOT by `build_all`: v1/v2/v3/manual and `EXPECTED` are
# untouched by it, and the viewer's four layers do not include it.

INPUTS_V4 = {'labels': 'data/bursts_annotated_110926.csv'}
OUTPUTS_V4 = {
    'events': 'bursts_v4_110926.csv',
    'labels': 'bursts_manual_v4_110926.csv',
    'unresolved_labels': 'burst_labels_recheck_v4_110926.csv',
    'splits': 'burst_splits_v4_110926.csv',
    'sweep': 'burst_v4_sweep_110926.csv',
    'model': 'burst_v4_110926.joblib',
    'union': 'bursts_viewer_v2v4manual_110926.csv',   # generate_kymograph_viewer --burst-layers v2v4manual
}
V4_CONF_LEVELS = ('high', 'medium', 'low')
V4_DEFAULT_CONF = 'medium'
V4_CONF_FIX = {'meium': 'medium', 'medium+': 'medium', 'mlow': 'low'}
V4_INTENSITY_FIX = {'mediuhm': 'medium', 'high': 'bright'}
V4_CARRIES_RE = r'carr+ies into division|divided between|signal split'
V4_SEED = 0
V4_LINK_DRIFT = LINK_DRIFT_PX
V4_BLOCKS_ALL = ('persistence', 'multiscale', 'spatial_rise')


def load_labels_v4(cells, path=INPUTS_V4['labels'], data_root='', verbose=True):
    """The 110926 labels, resolved exactly like `load_hand_labels`, plus the four new columns.

    Returns `(labels, rejects)`. Added columns: `confidence` (high/medium/low), `duration`
    (int frames), `intensity` (bright/medium/low/unknown), `note`, `carries_division`, and
    `defaulted` (which fields had to be filled). Typos are mapped (`V4_CONF_FIX`,
    `V4_INTENSITY_FIX`); a missing or unparseable confidence becomes `V4_DEFAULT_CONF`, a
    missing duration the file's median. Every defaulted row is printed.
    """
    d = pd.read_csv(os.path.join(data_root, path), dtype=str, keep_default_na=False)
    d.columns = [c.strip() for c in d.columns]
    d = d[[c for c in d.columns if c and not c.startswith('Unnamed')]]
    d = d.rename(columns={'Confidence': 'confidence', 'Duration (frames)': 'duration',
                          'Intensity': 'intensity', 'Notes': 'note'})

    conf = d['confidence'].str.strip().str.lower().replace(V4_CONF_FIX)
    intens = d['intensity'].str.strip().str.lower().replace(V4_INTENSITY_FIX)
    dur = pd.to_numeric(d['duration'].str.strip(), errors='coerce')
    med_dur = int(np.nanmedian(dur))
    bad_conf = ~conf.isin(V4_CONF_LEVELS)
    bad_dur = dur.isna() | (dur < 1)
    defaulted = np.where(bad_conf & bad_dur, 'confidence+duration',
                         np.where(bad_conf, 'confidence', np.where(bad_dur, 'duration', '')))
    d['confidence'] = conf.where(~bad_conf, V4_DEFAULT_CONF)
    d['duration'] = dur.where(~bad_dur, med_dur).astype(int).astype(str)
    d['intensity'] = intens.where(intens.isin(('bright', 'medium', 'low')), 'unknown')
    d['note'] = d['note'].str.strip()
    d['carries_division'] = d['note'].str.lower().str.contains(V4_CARRIES_RE, regex=True)
    d['defaulted'] = defaulted
    if verbose:
        n = int((d['defaulted'] != '').sum())
        print(f'  {n} label row(s) defaulted (confidence -> {V4_DEFAULT_CONF!r}, '
              f'duration -> median {med_dur}):')
        for r in d[d['defaulted'] != ''].itertuples():
            print(f'    {r.experiment_name} {r.FOV}/{r.trench_id} track {r.track_id} '
                  f'frame {r.frame}: {r.defaulted}')

    extra = ['confidence', 'duration', 'intensity', 'note', 'carries_division', 'defaulted']
    lab, rej = load_hand_labels(d, cells=cells, data_root=data_root, verbose=verbose,
                                extra_cols=extra)
    lab['duration'] = lab['duration'].astype(int)
    lab['carries_division'] = lab['carries_division'].astype(str).eq('True')
    return lab, rej


def build_training_table_v4(feat, labels, reviewed, conf_weights):
    """`y`, `sample_weight` and `cv_group` for every cell-frame of the `reviewed` trenches.

    y == 1       a labelled frame, weighted by `conf_weights[confidence]`
    weight 0     within +-`duration` frames of a label, on the labelled track -- and on its
                 ancestors/descendants too, but ONLY when the label carries into division.
                 These rows' true label is unknown (the labelled frame may be any of the D).
    y == 0       everything else in a reviewed trench, weight 1
    """
    keys = feat[KYMO_KEY].apply(tuple, axis=1)
    pool = feat[keys.isin(reviewed)].copy()
    pool['track_id'] = pool['track_id'].astype(int)
    pool['time_frame'] = pool['time_frame'].astype(int)
    lab = labels.astype({'track_id': int, 'time_frame': int})

    w_pos = {tuple(k): float(conf_weights[c]) for k, c in
             zip(lab[FRAME_KEY].values, lab['confidence'])}
    fk = list(map(tuple, pool[FRAME_KEY].values))
    pool['y'] = np.array([k in w_pos for k in fk], dtype=int)

    mask = np.zeros(len(pool), dtype=bool)
    idx = {k: i for i, k in enumerate(fk)}
    lab_by = dict(list(lab.groupby(KYMO_KEY, sort=False)))
    for key, g in pool.groupby(KYMO_KEY, sort=False):
        lb = lab_by.get(key)
        if lb is None:
            continue
        par = (g.groupby('track_id')['parent_track_id'].first()
               .apply(lambda p: int(p) if pd.notna(p) else None).to_dict())
        rel = _relatives(par)
        for r in lb.itertuples():
            fam = {int(r.track_id)}
            if r.carries_division:
                fam |= rel.get(int(r.track_id), set())
            sel = g[g.track_id.isin(fam)
                    & g.time_frame.between(r.time_frame - r.duration, r.time_frame + r.duration)]
            for k in map(tuple, sel[FRAME_KEY].values):
                mask[idx[k]] = True
    pool['in_decay_mask'] = mask & (pool['y'] == 0).to_numpy()
    pool['sample_weight'] = np.where(pool['y'] == 1, [w_pos.get(k, 0.0) for k in fk],
                                     np.where(pool['in_decay_mask'], 0.0, 1.0))
    pool['cv_group'] = pool[KYMO_KEY].apply(lambda r: '|'.join(r), axis=1)
    return pool


def _persistence(g, col, cap=10, frac=0.5, floor_frac=FOLD_FLOOR_FRAC):
    """Frames back / forward, along the lineage, that `col` stays >= `frac` x this frame's.

    Forward takes the MAX over daughters at each step (inherited signal), backward follows the
    parent chain. A cell-frame at or below 1% of its trench's max has nothing to persist and
    scores 0 both ways -- otherwise every dark frame would 'persist' forever.
    """
    idx, par, kid = _lineage_maps(g)
    v = g[col].to_numpy(dtype=float)
    tr = g['track_id'].to_numpy().astype(int)
    fr = g['time_frame'].to_numpy().astype(int)
    floor = floor_frac * max(np.nanmax(v) if len(v) else 0.0, 1e-9)
    back = np.zeros(len(g))
    fwd = np.zeros(len(g))
    for i in range(len(g)):
        if not v[i] > floor:
            continue
        bar = frac * v[i]
        t, f, n = int(tr[i]), int(fr[i]), 0
        while n < cap:
            f -= 1
            if (t, f) not in idx:
                p = par.get(t)
                if p is None or (p, f) not in idx:
                    break
                t = p
            if v[idx[(t, f)]] < bar:
                break
            n += 1
        back[i] = n
        front, n = [(int(tr[i]), int(fr[i]))], 0
        while n < cap:
            nxt = []
            for t, f in front:
                if (t, f + 1) in idx:
                    nxt.append((t, f + 1))
                else:
                    nxt += [(c, f + 1) for c in kid.get(t, []) if (c, f + 1) in idx]
            if not nxt or max(v[idx[c]] for c in nxt) < bar:
                break
            n += 1
            front = nxt
        fwd[i] = n
    return back, fwd


def _spatial_neighbour_max(g, col, k_max=3, slack_L=LINK_SLACK_CELL_LENGTHS,
                           drift=LINK_DRIFT_PX):
    """Track-free context: per cell-frame, the max `col` among cells within
    `slack_L * L + drift * k` px in y at frame t-k (and t+k), for k = 1..k_max.

    Returns `(back[k], fwd[k])`, arrays of shape (k_max, n). NaN where no cell is in range.
    The same tolerance as the event linker, so 'the same cell' means the same thing.
    """
    L = float(np.nanmedian(g['axis_major_length'].to_numpy(dtype=float)))
    L = L if np.isfinite(L) and L > 0 else 1.0
    v = g[col].to_numpy(dtype=float)
    y = g['centroid_y'].to_numpy(dtype=float)
    fr = g['time_frame'].to_numpy().astype(int)
    by_f = {f: np.flatnonzero(fr == f) for f in np.unique(fr)}
    back = np.full((k_max, len(g)), np.nan)
    fwd = np.full((k_max, len(g)), np.nan)
    for f, ii in by_f.items():
        for k in range(1, k_max + 1):
            tol = slack_L * L + drift * k
            for out, f2 in ((back, f - k), (fwd, f + k)):
                jj = by_f.get(f2)
                if jj is None:
                    continue
                near = np.abs(y[ii][:, None] - y[jj][None, :]) <= tol
                vals = np.where(near, v[jj][None, :], -np.inf).max(axis=1)
                out[k - 1, ii] = np.where(np.isfinite(vals), vals, np.nan)
    return back, fwd


def featurize_v4(df, blocks=V4_BLOCKS_ALL, verbose=True):
    """v2's `featurize`, plus opt-in blocks. Every block is computed inside one kymograph.

    `attrs['feature_cols']` is v2's 43 plus the requested blocks; `attrs['blocks']` maps
    'v2' and each block to its own columns, so a selection sweep can subset without
    recomputing.

      persistence   frames back / forward along the lineage that the signal stays >= 50% of
                    this frame (Duration: labelled bursts mostly last 3-6 frames)
      multiscale    v2's lineage windows at 4 and 6 frames next to its 3 and 5: fold over the
                    trailing median, rise over the trailing max, drop to the forward max
      spatial_rise  rise / fold / drop against the brightest cell within the linker's y
                    tolerance at t-1..t-3 and t+1..t+3 -- needs no track ids at all
    """
    d = featurize(df, verbose=False)
    blk = {'v2': list(d.attrs['feature_cols'])}
    g = d.groupby(KYMO_KEY, sort=False)
    new = {}
    for c in FLUOR_COLS:
        scale = np.maximum(g[c].transform('max').to_numpy(dtype=float), 1e-9)
        x = d[c].to_numpy(dtype=float)
        floor = FOLD_FLOOR_FRAC * scale
        if 'persistence' in blocks:
            pb, pf = np.zeros(len(d)), np.zeros(len(d))
            for _, gg in g:
                b_, f_ = _persistence(gg, c)
                pb[gg.index], pf[gg.index] = b_, f_
            new.setdefault('persistence', {}).update({
                f'{c}__persist_back': pb, f'{c}__persist_fwd': pf,
                f'{c}__persist_total': pb + pf})
        if 'multiscale' in blocks:
            cols = {}
            for _, gg in g:
                for k in (4, 6):
                    for nm, how, dr in (('bmed', 'median', 'back'), ('bmax', 'max', 'back'),
                                        ('fmax', 'max', 'fwd')):
                        cols.setdefault(f'{nm}{k}', np.full(len(d), np.nan))[gg.index] = \
                            _lineage_window(gg, c, k, how, dr)
            blkd = new.setdefault('multiscale', {})
            for k in (4, 6):
                blkd[f'{c}__fold_back{k}m'] = x / np.maximum(cols[f'bmed{k}'], floor)
                blkd[f'{c}__rise{k}m'] = (x - cols[f'bmax{k}']) / scale
                blkd[f'{c}__drop{k}m'] = (x - cols[f'fmax{k}']) / scale
        if 'spatial_rise' in blocks:
            B_ = np.full((3, len(d)), np.nan)
            F_ = np.full((3, len(d)), np.nan)
            for _, gg in g:
                b_, f_ = _spatial_neighbour_max(gg, c)
                B_[:, gg.index], F_[:, gg.index] = b_, f_
            with np.errstate(all='ignore'):
                b3 = np.nanmax(B_, axis=0)
                f3 = np.nanmax(F_, axis=0)
            new.setdefault('spatial_rise', {}).update({
                f'{c}__sp_rise1': (x - B_[0]) / scale,
                f'{c}__sp_rise3': (x - b3) / scale,
                f'{c}__sp_fold3': x / np.maximum(b3, floor),
                f'{c}__sp_drop3': (x - f3) / scale,
            })
    for b in blocks:
        cols = new.get(b, {})
        for k, v in cols.items():
            d[k] = v
        blk[b] = list(cols)
    d.attrs['blocks'] = blk
    d.attrs['feature_cols'] = [c for b in ('v2',) + tuple(blocks) for c in blk[b]]
    if verbose:
        print(f'featurize_v4: {len(d):,} cell-frames x {len(d.attrs["feature_cols"])} '
              f'features ({", ".join(f"{b} {len(blk[b])}" for b in blk)})')
    return d



def v4_events(feat, threshold, L_by=None):
    """Score -> events, the v3 linker (peak-reported, sum tolerance, drift 10 px/frame)."""
    if L_by is None:
        L_by = feat.groupby(KYMO_KEY)['axis_major_length'].median()
    return flags_to_events(feat, 'score', feat['score'] >= threshold, link_dt=LINK_DT,
                           slack=LINK_SLACK_CELL_LENGTHS, drift=V4_LINK_DRIFT, L_by=L_by,
                           report='peak')


def threshold_sweep(scored, labels, thresholds=None, tol=1):
    """Event P/R/F1 at +-`tol` per threshold, over the trenches `scored` covers."""
    thresholds = np.round(np.arange(0.50, 0.951, 0.05), 2) if thresholds is None else thresholds
    L_by = scored.groupby(KYMO_KEY)['axis_major_length'].median()
    keys = set(map(tuple, scored[KYMO_KEY].drop_duplicates().values))
    lab = labels[[tuple(k) in keys for k in labels[KYMO_KEY].values]]
    rows = []
    for th in thresholds:
        ev = v4_events(scored, th, L_by)
        tp = match_one_to_one(ev, lab, tol=tol)
        p, r, f = prf(tp, len(ev), len(lab))
        rows.append({'threshold': float(th), 'n_events': len(ev), 'n_labels': len(lab),
                     'tp': tp, 'P': p, 'R': r, 'F1': f})
    return pd.DataFrame(rows)


def run_v4(clean, fiji, labels, blocks=(), conf_weights=None, threshold=None,
           feat=None, splits=None, model_out=None, verbose=True):
    """Split, fit on dev, score every trench, link.

    Returns `(events, splits, bundle, feat, sweep)`; `sweep` is the dev out-of-fold
    threshold table.

    Dev cell-frames keep their out-of-fold score (5-fold, grouped by trench); test and
    unreviewable trenches get the dev-fitted model -- so no trench is scored by a model that
    saw it. `threshold=None` picks the max dev out-of-fold event F1 at +-1. Deterministic.
    Pass a precomputed `feat` (from `featurize_v4`) / `splits` to skip recomputing them.
    """
    import joblib
    conf_weights = conf_weights or {c: 1.0 for c in V4_CONF_LEVELS}
    if splits is None:
        splits = make_splits(trench_index(clean), labels, test_frac=SPLIT_TEST_FRAC,
                             seed=SPLIT_SEED)
    dev = set(map(tuple, splits.loc[splits['split'] == 'dev', KYMO_KEY].values))
    lab_dev = labels[[tuple(r) in dev for r in labels[KYMO_KEY].values]].copy()

    if feat is None:
        feat = featurize_v4(clean.merge(fiji, on=FRAME_KEY, how='left'),
                            blocks=V4_BLOCKS_ALL, verbose=verbose)
    feat = feat.copy()
    FEATS = [c for b in ('v2',) + tuple(blocks) for c in feat.attrs['blocks'][b]]

    tt = build_training_table_v4(feat, lab_dev, dev, conf_weights)
    oof, auc, ap = cv_score_all(tt, FEATS, seed=V4_SEED, monotone=False, use_weights=True)
    sweep = threshold_sweep(oof, lab_dev)
    if threshold is None:
        threshold = float(sweep.loc[sweep['F1'].idxmax(), 'threshold'])

    ttr = tt.reset_index(drop=True)
    fit_idx = np.flatnonzero((ttr['sample_weight'] > 0).to_numpy())
    model = make_model_v3(seed=V4_SEED, monotone=False)
    model.fit(ttr.loc[fit_idx, FEATS], ttr['y'].to_numpy().astype(int)[fit_idx],
              sample_weight=ttr['sample_weight'].to_numpy(dtype=float)[fit_idx])

    feat['score'] = model.predict_proba(feat[FEATS])[:, 1]
    k_oof = dict(zip(map(tuple, oof[FRAME_KEY].values), oof['score']))
    feat['score'] = [k_oof.get(k, s) for k, s in
                     zip(map(tuple, feat[FRAME_KEY].astype({'track_id': int,
                                                             'time_frame': int}).values),
                         feat['score'])]

    ev = v4_events(feat, threshold)
    geo = [c for c in GEO_COLS if c in clean.columns and c != 'gene']
    ev = ev.astype({'track_id': int, 'time_frame': int}).merge(
        clean.astype({'track_id': int, 'time_frame': int})[FRAME_KEY + geo],
        on=FRAME_KEY, how='left')
    assert ev['centroid_y'].notna().all(), 'a v4 event has no geometry'
    ev = add_relative_y(ev, clean).assign(burst_v4=True)

    bundle = {'model': model, 'feature_cols': list(FEATS), 'blocks': list(blocks),
              'threshold': threshold, 'conf_weights': dict(conf_weights),
              'mask_rule': '+-duration on the labelled track; + relatives iff carries_division',
              'link_dt': LINK_DT, 'slack_cell_lengths': LINK_SLACK_CELL_LENGTHS,
              'drift_px': V4_LINK_DRIFT, 'seed': V4_SEED, 'split_seed': SPLIT_SEED,
              'labels': INPUTS_V4['labels'], 'n_labels': int(len(lab_dev)),
              'dev_oof_auc': auc, 'dev_oof_ap': ap}
    if model_out is not None:
        os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
        joblib.dump(bundle, model_out)
    if verbose:
        best = sweep.loc[sweep['threshold'] == threshold].iloc[0]
        print(f'v4: {len(ev)} events at {threshold} | dev OOF AUC {auc:.4f} PR-AUC {ap:.4f} '
              f'event F1(+-1) {best.F1:.3f} | {len(FEATS)} features | {len(dev)} dev trenches, '
              f'{len(lab_dev)} dev labels')
    return ev, splits, bundle, feat, sweep

