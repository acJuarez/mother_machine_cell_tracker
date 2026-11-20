"""Build v4: v2's classifier refitted on the 110926 labels, with v3's training discipline.

    conda run -n mmct --no-capture-output python build_v4.py

Does NOT touch any build_bursts.py output or `bursts.EXPECTED`. bursts.md section 5b explains it.

  1. load the 110926 labels (+ confidence / duration / intensity / notes), a fresh trench split
  2. SELECTION on dev out-of-fold only: v2's 43 features, then each candidate block alone
     (kept if dev event F1 at +-1 rises >= MIN_GAIN), then the kept blocks together, then
     the confidence-weight sweep. Each config is scored at its own best threshold.
  3. fit the chosen config on all of dev, score everything (dev keeps its OOF score), link
  4. read the TEST split once: P/R/F1 at exact/+-1/+-3/+-duration, recall per intensity,
     confidence and gene, rates per division and the Spearman against the labels, and v2/v3
     on the same test trenches (v2 also without the trenches it was fitted on)

Writes under --out-dir (default data/): bursts_v4_110926.csv, bursts_manual_v4_110926.csv,
burst_labels_recheck_v4_110926.csv, burst_splits_v4_110926.csv, burst_v4_sweep_110926.csv,
bursts_viewer_v2v4manual_110926.csv (the viewer's three-layer union, also via --union-only);
and models/burst_v4_110926.joblib.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import mmtrack.bursts as B

# `legacy/mmtrack/` is not a package, so the archived halves of the burst pipeline are loaded
# by path and grafted onto `B`. Everything below then reads as it did when this was live.
def _graft(module_name):
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'mmtrack', module_name + '.py')
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if not name.startswith('_') and not hasattr(B, name):
            setattr(B, name, getattr(mod, name))

_graft('bursts_v4')


MIN_GAIN = 0.01
WEIGHT_GRID = {'1/1/1': {'high': 1.0, 'medium': 1.0, 'low': 1.0},
               '1/0.7/0.4': {'high': 1.0, 'medium': 0.7, 'low': 0.4},
               '1/0.5/0.2': {'high': 1.0, 'medium': 0.5, 'low': 0.2}}


def keys_of(d):
    return set(map(tuple, d[B.KYMO_KEY].drop_duplicates().values))


def within(d, keys):
    return d[[tuple(k) in keys for k in d[B.KYMO_KEY].values]]


def dev_cv(feat, lab_dev, dev, blocks, weights):
    feats = [c for b in ('v2',) + tuple(blocks) for c in feat.attrs['blocks'][b]]
    tt = B.build_training_table_v4(feat, lab_dev, dev, weights)
    oof, auc, ap = B.cv_score_all(tt, feats, seed=B.V4_SEED, monotone=False, use_weights=True)
    sw = B.threshold_sweep(oof, lab_dev)
    best = sw.loc[sw['F1'].idxmax()]
    return {'n_features': len(feats), 'auc': auc, 'pr_auc': ap,
            'threshold': best.threshold, 'P': best.P, 'R': best.R, 'F1': best.F1,
            'n_events': int(best.n_events)}


def score_block(ev, lab, empty, name):
    rows = []
    for tol, tc in ((0, None), (1, None), (3, None), ('D', 'duration')):
        tp = B.match_events(ev, lab, tol=0 if tol == 'D' else tol, tol_col=tc)[0]
        p, r, f = B.prf(tp, len(ev), len(lab))
        rows.append({'model': name, 'tol': f'+-{tol}', 'n_events': len(ev), 'n_labels': len(lab),
                     'tp': tp, 'P': p, 'R': r, 'F1': f})
    fp_empty = len(within(ev, empty))
    return rows, fp_empty


def read_ev(path):
    d = pd.read_csv(path, dtype=B.READ_DTYPES)
    d['FOV'] = d['FOV'].str.zfill(3)
    return d.astype({'track_id': int, 'time_frame': int})


def write_viewer_union(clean, manual, v2, v4, path):
    """The three-layer table `generate_kymograph_viewer.py --burst-layers v2v4manual` draws."""
    union = B.build_burst_csv({'manual': manual, 'v2': v2, 'v4': v4}, clean, path)
    for v, d in (('manual', manual), ('v2', v2), ('v4', v4)):
        assert int(union[f'burst_{v}'].sum()) == len(d), f'union lost {v} rows'
    return union


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--union-only', action='store_true',
                    help='only (re)write the viewer union from the already-written CSVs')
    args = ap.parse_args()
    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    pd.set_option('display.width', 200)
    out = lambda k: os.path.join(args.out_dir, B.OUTPUTS_V4[k])
    if args.union_only:
        write_viewer_union(B.clean_cells(), read_ev(out('labels')),
                           read_ev('data/bursts_v2_100226.csv'), read_ev(out('events')),
                           out('union'))
        return

    print('--- inputs ---')
    clean = B.clean_cells()
    fiji = B.load_fiji()
    labels, rejects = B.load_labels_v4(clean, verbose=True)
    print(f'  resolved {len(labels)} labels, {len(rejects)} unresolvable')
    print(f'  confidence {labels.confidence.value_counts().to_dict()}')
    print(f'  intensity  {labels.intensity.value_counts().to_dict()}')
    print(f'  duration   median {labels.duration.median():.0f}, '
          f'range {labels.duration.min()}-{labels.duration.max()}')
    print(f'  carries into division: {int(labels.carries_division.sum())}')

    splits = B.make_splits(B.trench_index(clean), labels, test_frac=B.SPLIT_TEST_FRAC,
                           seed=B.SPLIT_SEED)
    sc = splits['split'].value_counts().to_dict()
    dev = set(map(tuple, splits.loc[splits.split == 'dev', B.KYMO_KEY].values))
    test = set(map(tuple, splits.loc[splits.split == 'test', B.KYMO_KEY].values))
    assert not (dev & test), 'a trench is in both dev and test'
    lab_dev, lab_test = within(labels, dev), within(labels, test)
    print(f'  split {sc}: dev {len(lab_dev)} labels, test {len(lab_test)} labels')

    feat = B.featurize_v4(clean.merge(fiji, on=B.FRAME_KEY, how='left'))

    print('--- selection (dev out-of-fold only; each config at its best threshold) ---')
    rows = []
    w1 = WEIGHT_GRID['1/1/1']

    def run(name, blocks, wname):
        r = {'config': name, 'blocks': '+'.join(blocks) or '-', 'weights': wname,
             **dev_cv(feat, lab_dev, dev, blocks, WEIGHT_GRID[wname])}
        rows.append(r)
        print(f'  {name:<28} {r["n_features"]:>3} feats  AUC {r["auc"]:.4f}  '
              f'PR-AUC {r["pr_auc"]:.4f}  th {r["threshold"]:.2f}  '
              f'P {r["P"]:.3f} R {r["R"]:.3f} F1 {r["F1"]:.3f}')
        return r

    base = run('v2 features', (), '1/1/1')
    kept = [b for b in B.V4_BLOCKS_ALL
            if run(f'+ {b}', (b,), '1/1/1')['F1'] - base['F1'] >= MIN_GAIN]
    best_blocks, best = (), base
    if len(kept) == 1:
        best_blocks = tuple(kept)
        best = next(x for x in rows if x['blocks'] == kept[0])
    elif kept:
        r = run('+ ' + ' + '.join(kept), tuple(kept), '1/1/1')
        single = max((x for x in rows if x['blocks'] in kept), key=lambda x: x['F1'])
        if len(kept) > 1 and single['F1'] > r['F1']:
            best_blocks, best = (single['blocks'],), single
        else:
            best_blocks, best = tuple(kept), r
    best_w = '1/1/1'
    for wname in ('1/0.7/0.4', '1/0.5/0.2'):
        r = run(f'{"+".join(best_blocks) or "v2"} w={wname}', best_blocks, wname)
        if r['F1'] > best['F1']:
            best, best_w = r, wname
    print(f'  CHOSEN: blocks {best_blocks or "(none)"}, weights {best_w}, '
          f'threshold {best["threshold"]:.2f}, dev OOF F1 {best["F1"]:.3f}')
    pd.DataFrame(rows).to_csv(out('sweep'), index=False)

    print('--- final fit ---')
    ev, _, bundle, feat_s, sweep = B.run_v4(
        clean, fiji, labels, blocks=best_blocks, conf_weights=WEIGHT_GRID[best_w],
        threshold=float(best['threshold']), feat=feat, splits=splits,
        model_out=os.path.join(args.models_dir, B.OUTPUTS_V4['model']))
    assert len(ev.merge(clean.astype({'track_id': int, 'time_frame': int})[B.FRAME_KEY],
                        on=B.FRAME_KEY)) == len(ev), 'an event does not join the cell table'
    tt = B.build_training_table_v4(feat, labels, dev | test, WEIGHT_GRID[best_w])
    assert not (tt.in_decay_mask & (tt.sample_weight > 0)).any(), 'a masked row is weighted'

    print('--- TEST split, read once ---')
    empty = set(map(tuple, splits.loc[(splits.split == 'test') & (splits.n_labels == 0),
                                      B.KYMO_KEY].values))
    v2 = read_ev('data/bursts_v2_100226.csv')
    v3 = read_ev('data/bursts_v3_100226.csv')
    old_lab = read_ev('data/bursts_manual_100226.csv')
    v2_train = keys_of(pd.read_csv('data/bursts_lacZ_gfcE_091026.csv', dtype=str)
                       .assign(FOV=lambda d: d.FOV.str.zfill(3)))
    v3_dev = set(map(tuple, pd.read_csv('data/burst_splits_100226.csv', dtype=str)
                     .assign(FOV=lambda d: d.FOV.str.zfill(3))
                     .query('split == "dev"')[B.KYMO_KEY].values))
    print(f'  {len(test)} test trenches, {len(empty)} certified empty; '
          f'{len(test & v2_train)} were v2 training trenches, {len(test & v3_dev)} v3 dev '
          '(v3 dev trenches carry out-of-fold scores, so only v2 is contaminated)')
    res = []
    for ref_name, ref in (('110926 labels', labels), ('100226 labels', old_lab)):
        for scope_name, scope in (('all test', test), ('test minus v2-train', test - v2_train)):
            lab_t = within(ref, scope)
            if 'duration' not in lab_t:
                lab_t = lab_t.assign(duration=1)
            for name, e in (('v4', ev), ('v2', v2), ('v3', v3)):
                rws, fpe = score_block(within(e, scope), lab_t, empty & scope, name)
                for r in rws:
                    res.append({'reference': ref_name, 'scope': scope_name, **r,
                                'events_in_empty': fpe})
    R = pd.DataFrame(res)
    for (ref_name, scope_name), g in R.groupby(['reference', 'scope'], sort=False):
        print(f'\n  [{ref_name} | {scope_name}]  (+-D uses each label\'s duration; '
              '100226 has none, so +-D = +-1 there)')
        piv = g.pivot_table(index='model', columns='tol', values='F1', sort=False)
        extra = g.groupby('model', sort=False)[['n_events', 'n_labels', 'events_in_empty']].first()
        pr = g[g.tol == '+-1'].set_index('model')[['P', 'R']].add_suffix('(+-1)')
        print(pd.concat([extra, pr, piv.add_prefix('F1 ')], axis=1).round(3).to_string())

    print('\n  v4 recall at +-1 on the test split, by label attribute:')
    lab_t = within(labels, test)
    _, hit, _ = B.match_events(within(ev, test), lab_t, tol=1)
    lab_t = lab_t.assign(found=lab_t.index.isin(hit))
    allr = within(labels, dev | test)
    _, hit_all, _ = B.match_events(within(ev, dev | test), allr, tol=1)
    allr = allr.assign(found=allr.index.isin(hit_all))
    for col in ('intensity', 'confidence', 'gene', 'carries_division'):
        t = lab_t.groupby(col)['found'].agg(['size', 'mean']).rename(
            columns={'size': 'n_test', 'mean': 'recall_test'})
        a = allr.groupby(col)['found'].agg(['size', 'mean']).rename(
            columns={'size': 'n_all', 'mean': 'recall_all_oos'})
        print(pd.concat([t, a], axis=1).round(3).to_string(), '\n')

    print('  rates per observed division (all trenches; dev = OOF, test/unreviewed = dev model)')
    div = B.load_divisions().groupby('gene').size()
    rate = pd.DataFrame({'divisions': div,
                         'v4': ev.groupby('gene').size(), 'v2': v2.groupby('gene').size(),
                         'v3': v3.groupby('gene').size(),
                         'manual_110926': labels.groupby('gene').size(),
                         'manual_100226': old_lab.groupby('gene').size()}).fillna(0)
    for c in ('v4', 'v2', 'v3', 'manual_110926', 'manual_100226'):
        rate[c] = rate[c] / rate['divisions']
    rate = rate.sort_values('v4', ascending=False)
    rate.index = [g + (' *' if g in B.UNREVIEWED_GENES else '') for g in rate.index]
    print(rate.round(4).to_string())
    rv = rate.loc[[g for g in rate.index if not g.endswith('*')]]
    for c in ('v4', 'v2', 'v3', 'manual_100226'):
        print(f'  Spearman {c:<14} vs manual_110926 (7 reviewed genes): '
              f'{spearmanr(rv[c], rv["manual_110926"]).statistic:+.3f}')
    print('  * unreviewed: no labels, counts reported, not validated')

    print('\n--- writing ---')
    ev.to_csv(out('events'), index=False)
    B.build_manual(clean, labels).to_csv(out('labels'), index=False)
    rejects.to_csv(out('unresolved_labels'), index=False)
    (splits[B.KYMO_KEY + ['gene', 'split', 'n_labels', 'n_cell_frames']]
     .sort_values(['gene', 'split'] + B.KYMO_KEY).to_csv(out('splits'), index=False))
    R.to_csv(os.path.join(args.out_dir, 'burst_v4_holdout_110926.csv'), index=False)
    write_viewer_union(clean, read_ev(out('labels')), v2, ev, out('union'))
    for k in ('events', 'labels', 'unresolved_labels', 'splits', 'sweep', 'union'):
        print(f'  wrote {out(k)}')
    print(f'  wrote {os.path.join(args.out_dir, "burst_v4_holdout_110926.csv")}')
    print(f'  wrote {os.path.join(args.models_dir, B.OUTPUTS_V4["model"])}')
    print(f'v4: {len(ev)} events, by gene {ev.gene.value_counts().to_dict()}')


if __name__ == '__main__':
    main()
