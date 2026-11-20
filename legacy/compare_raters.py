"""Two labellers on the same trenches: is the training data biased toward labeller A?

    conda run -n mmct --no-capture-output python compare_raters.py

Labeller A is `data/annotated_bursts_100226.csv` (the ground truth every model is trained or
scored on). The second labeller is `bursts.SECOND_RATER`, covering araB, lacZ and chpS only;
its `confidence` and `comment` columns are ignored. Reads frozen inputs only and writes no
file `build_bursts.py` owns. The headline counts are pinned in EXPECTED and asserted.

  1. ingest       both files resolved onto cell-frames; the frame convention checked
  2. coverage     which trenches each labeller marked; empty-vs-not kappa
  3. counts       per-trench and per-division concordance (is one labeller more liberal?)
  4. timing       where each labeller puts a label relative to the burst's local peak
  5. agreement    one-to-one event F1, raw and peak-snapped, with a trench bootstrap
  6. disputes     what separates the events only one labeller marked (Holm-corrected)
  7. definitions  v1/v2/v3 scored against six references; is v2 > v3 labeller-specific?
  8. v2 recall    share of each labeller's labels inside a v2 event's linked span

Writes data/rater_events_102426.csv: one row per peak-snapped event, `cls` in
{both, A_only, H_only} -- the worklist for adjudicating the disputed events.

Matching is within +-tol frames AND 1.5 median cell lengths in y, so two cells in the same
frame never match (test_bursts.match_one_to_one is frame-only, fine for one detector against
one label set but not here). The "local peak" of a label is the brightest
intensity_mean_fluor cell-frame within +-5 frames and the 1.5 L + 10 px |dt| neighbourhood,
the tolerance the event linker uses.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score

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

_graft('second_rater')


G = list(B.SECOND_RATER_GENES)
F = 'intensity_mean_fluor'
PEAK_WINDOW = 5
N_BOOT = 2000
SEED = 20261024
OUT_EVENTS = 'data/rater_events_102426.csv'

EXPECTED = {'A': 257, 'H': 265, 'H_rejected': 4, 'trenches': 92,
            'raw_tp': {0: 51, 1: 106, 3: 161}, 'snapped': (246, 225), 'snapped_tp1': 168,
            'cls': {'both': 168, 'A_only': 78, 'H_only': 57}}
FAILED = []


def check(got, want, what):
    ok = got == want
    print(f'  {"PASS" if ok else "FAIL"}  {what}: {got}' + ('' if ok else f' (want {want})'))
    if not ok:
        FAILED.append(what)


def in_scope(d, scope):
    return d[[tuple(r) in scope for r in d[B.KYMO_KEY].values]].reset_index(drop=True)


def match(X, Y, tol, Lmed, yslack=1.5):
    """Maximum-cardinality one-to-one (index_X, index_Y) pairs, per trench."""
    pairs = []
    yb = dict(list(Y.groupby(B.KYMO_KEY, sort=False)))
    for k, x in X.groupby(B.KYMO_KEY, sort=False):
        y = yb.get(k)
        if y is None:
            continue
        L = Lmed[k]
        dt = np.abs(x.time_frame.to_numpy(int)[:, None]
                    - y.time_frame.to_numpy(int)[None, :]).astype(float)
        dy = np.abs(x.centroid_y.to_numpy(float)[:, None] - y.centroid_y.to_numpy(float)[None, :])
        ok = (dt <= tol) & (dy <= yslack * L)
        if not ok.any():
            continue
        ri, ci = linear_sum_assignment(np.where(ok, dt + dy / (100 * L), 1e6))
        pairs += [(x.index[r], y.index[c]) for r, c in zip(ri, ci) if ok[r, c]]
    return pairs


def counts(X, Y, tol, Lmed):
    return np.array([len(X), len(Y), len(match(X, Y, tol, Lmed)) if len(X) and len(Y) else 0])


def f1(c):
    return 2 * c[2] / max(c[0] + c[1], 1)


def describe(c):
    return f'{c[2]}/{c[0]},{c[1]}  P {c[2] / max(c[0], 1):.3f} R {c[2] / max(c[1], 1):.3f} F1 {f1(c):.3f}'


def bootstrap(per_trench, combine, rng, n=N_BOOT):
    """Resample trenches; `per_trench` maps trench -> dict of count vectors."""
    keys = list(per_trench)
    out = []
    for _ in range(n):
        ks = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        tot = {name: sum(per_trench[k][name] for k in ks) for name in per_trench[keys[0]]}
        out.append(combine(tot))
    return np.array(out)


def ci(v):
    return f'{np.median(v):+.3f} [{np.quantile(v, .025):+.3f}, {np.quantile(v, .975):+.3f}]'


def by_trench(d, keys):
    g = dict(list(d.groupby(B.KYMO_KEY, sort=False)))
    return {k: g.get(k, d.iloc[:0]) for k in keys}


def snap(L, byT, Lmed):
    """Attach each label's local peak and where the label sits relative to it."""
    rows = []
    for r in L.itertuples():
        k = (r.experiment_name, r.FOV, r.trench_id)
        g, Lk = byT[k], Lmed[k]
        dt = (g.time_frame - r.time_frame).abs()
        nb = g[(dt <= PEAK_WINDOW) & ((g.centroid_y - r.centroid_y).abs() <= 1.5 * Lk + 10 * dt)]
        pk = nb.loc[nb[F].idxmax()]
        me = g[(g.track_id == r.track_id) & (g.time_frame == r.time_frame)].iloc[0]
        rows.append({'pk_frame': int(pk.time_frame), 'pk_track': int(pk.track_id),
                     'pk_y': pk.centroid_y, 'off': int(r.time_frame - pk.time_frame),
                     'frac_of_peak': me[F] / pk[F],
                     'fpos': r.time_frame / max(g.time_frame.max(), 1)})
    return pd.concat([L.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def to_peak(d):
    s = d.assign(time_frame=d.pk_frame, track_id=d.pk_track, centroid_y=d.pk_y)
    return s.drop_duplicates(B.FRAME_KEY).reset_index(drop=True)


def load_events(v, geo, scope):
    d = pd.read_csv(f'data/bursts_{v}_100226.csv', dtype=B.READ_DTYPES)
    d['FOV'] = d['FOV'].str.zfill(3)
    d = d[d.gene.isin(G)].astype({'track_id': int, 'time_frame': int})
    if 'centroid_y' not in d:
        d = d.merge(geo, on=B.FRAME_KEY)
    return in_scope(d, scope)


def main():
    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
    rng = np.random.default_rng(SEED)

    print('1. ingest')
    cells = B.load_cells(drop_excluded=False)
    present = set(zip(cells.experiment_name, cells.FOV, cells.trench_id))
    stale = [t for t in B.FAILED_KYMOGRAPHS if tuple(t) not in present]
    if stale:
        print(f'  WARNING: FAILED_KYMOGRAPHS entries match no trench, ignored here: {stale}')
    cells = B.drop_excluded_kymographs(cells, 'cells', require_match=False, verbose=False)
    clean = cells[~cells.is_degenerate_mask].reset_index(drop=True)

    H, Hrej, h_empty, skipped = B.load_second_rater(clean, verbose=False)
    A, _ = B.load_hand_labels(cells=clean, genes=G, verbose=False)
    A = A[[tuple(r) not in skipped for r in A[B.KYMO_KEY].values]]
    raw = pd.read_csv(B.SECOND_RATER, dtype=str, keep_default_na=False)
    raw = raw[raw.track_id.str.upper() != 'N/A']
    raw = raw.assign(FOV=raw.FOV.str.zfill(3), track_id=raw.track_id.astype(int),
                     time_frame=raw.frame.astype(int))
    c = clean[B.FRAME_KEY].astype({'track_id': int, 'time_frame': int})
    print(f'  second labeller: {len(H)} resolved, {len(Hrej)} rejected; read as viewer frames. '
          f'Read as raw time_frame only {len(raw.merge(c, on=B.FRAME_KEY))} of {len(raw)} '
          f'would resolve.')
    print(f'  skipped trench(es), out of scope for both: {sorted(skipped)}')
    print(f'  reviewed with no burst: {sorted(h_empty)}')

    X = B.run_v2(clean, B.load_fiji(), verbose=False)[1]
    X = X.astype({'track_id': int, 'time_frame': int})
    geo = X[B.FRAME_KEY + ['centroid_y']]
    A = A.astype({'track_id': int, 'time_frame': int}).merge(geo, on=B.FRAME_KEY)
    H = H.astype({'track_id': int, 'time_frame': int}).merge(geo, on=B.FRAME_KEY)
    ti = B.trench_index(clean)
    ti = ti[ti.gene.isin(G)]
    scope = set(map(tuple, ti[B.KYMO_KEY].values)) - skipped
    keys = sorted(scope)
    X = in_scope(X, scope)
    Lmed = X.groupby(B.KYMO_KEY).axis_major_length.median()
    byT = dict(list(X.groupby(B.KYMO_KEY, sort=False)))
    check(len(A), EXPECTED['A'], 'labeller A labels in scope')
    check(len(H), EXPECTED['H'], 'second-labeller labels resolved')
    check(len(Hrej), EXPECTED['H_rejected'], 'second-labeller labels rejected')
    check(len(scope), EXPECTED['trenches'], 'trenches in scope')

    print('\n2. coverage')
    TA, TH = set(map(tuple, A[B.KYMO_KEY].values)), set(map(tuple, H[B.KYMO_KEY].values))
    for g in G:
        tg = set(map(tuple, ti.loc[ti.gene == g, B.KYMO_KEY].values)) & scope
        print(f'  {g}: {len(tg)} trenches | both labelled {len(TA & TH & tg)} | A only '
              f'{len((TA - TH) & tg)} | H only {len((TH - TA) & tg)} | neither {len(tg - TA - TH)}')
    pt = ti.set_index(B.KYMO_KEY)[['gene', 'n_cell_frames']].loc[keys]
    pt['nA'] = A.groupby(B.KYMO_KEY).size().reindex(pt.index).fillna(0)
    pt['nH'] = H.groupby(B.KYMO_KEY).size().reindex(pt.index).fillna(0)
    print(f'  empty-vs-not kappa {cohen_kappa_score(pt.nA > 0, pt.nH > 0):.3f}; empty for A '
          f'{int((pt.nA == 0).sum())}, for H {int((pt.nH == 0).sum())}, A-only-labelled '
          f'{int(((pt.nA > 0) & (pt.nH == 0)).sum())}, H-only-labelled '
          f'{int(((pt.nA == 0) & (pt.nH > 0)).sum())}')

    print('\n3. counts')
    for g in G + ['all']:
        d = pt if g == 'all' else pt[pt.gene == g]
        rho = stats.spearmanr(d.nA, d.nH).statistic
        w = stats.wilcoxon(d.nA, d.nH).pvalue
        diff = d.nA - d.nH
        print(f'  {g:4s} n={len(d):2d}  A {d.nA.sum():.0f}  H {d.nH.sum():.0f}  spearman {rho:.3f}  '
              f'wilcoxon p={w:.2g}  A-H per trench {diff.mean():+.2f} (LoA +-{1.96 * diff.std():.1f})')
    dv = B.load_divisions()
    dv['FOV'] = dv['FOV'].str.zfill(3)
    nd = in_scope(dv, scope).groupby('gene').size()
    print('  bursts per observed division, exact Poisson 95% CI:')
    for nm, L in (('A', A), ('H', H)):
        cells_ = []
        for g in G:
            n, dd = int((L.gene == g).sum()), nd[g]
            lo = stats.chi2.ppf(.025, 2 * n) / 2 / dd if n else 0.0
            hi = stats.chi2.ppf(.975, 2 * n + 2) / 2 / dd
            cells_.append(f'{g} {n / dd:.3f} [{lo:.3f}, {hi:.3f}]')
        print(f'    {nm}  ' + ' | '.join(cells_))

    print('\n4. timing (label frame minus local-peak frame)')
    A2, H2 = snap(A, byT, Lmed), snap(H, byT, Lmed)
    for nm, d in (('A', A2), ('H', H2)):
        hist = d.off.clip(-4, 4).value_counts(normalize=True).sort_index().round(3).to_dict()
        print(f'  {nm}: at peak {np.mean(d.off == 0):.3f}, median {d.off.median():+.0f}, mean '
              f'fraction of peak {d.frac_of_peak.mean():.3f}  {hist}')
    print(f'  Mann-Whitney offset p={stats.mannwhitneyu(A2.off, H2.off).pvalue:.2g}, fraction '
          f'of peak p={stats.mannwhitneyu(A2.frac_of_peak, H2.frac_of_peak).pvalue:.2g}')
    dts = [H.loc[b, 'time_frame'] - A.loc[a, 'time_frame'] for a, b in match(A, H, 3, Lmed)]
    print(f'  H - A over the +-3 matches: {pd.Series(dts).value_counts().sort_index().to_dict()}')

    print('\n5. agreement between the labellers')
    for tol in (0, 1, 3):
        cnt = counts(A, H, tol, Lmed)
        check(int(cnt[2]), EXPECTED['raw_tp'][tol], f'raw +-{tol} matches')
        print(f'      raw +-{tol}: {describe(cnt)}')
    As, Hs = to_peak(A2), to_peak(H2)
    check((len(As), len(Hs)), EXPECTED['snapped'], 'unique events after peak snapping')
    for tol in (0, 1, 3):
        print(f'  peak-snapped +-{tol}: {describe(counts(As, Hs, tol, Lmed))}')
    for g in G:
        print(f'    {g} +-1: {describe(counts(As[As.gene == g], Hs[Hs.gene == g], 1, Lmed))}')
    pairs = match(As, Hs, 1, Lmed)
    check(len(pairs), EXPECTED['snapped_tp1'], 'peak-snapped +-1 matches')

    v2 = load_events('v2', geo, scope)
    pa, ph, pv = by_trench(As, keys), by_trench(Hs, keys), by_trench(v2, keys)
    ra, rh = by_trench(A, keys), by_trench(H, keys)
    per = {k: {'AH': counts(pa[k], ph[k], 1, Lmed), 'vA': counts(pv[k], ra[k], 3, Lmed),
               'vH': counts(pv[k], rh[k], 3, Lmed)} for k in keys}
    bs = bootstrap(per, lambda t: (f1(t['AH']), f1(t['vA']), f1(t['vH'])), rng)
    print(f'  trench bootstrap ({N_BOOT} x {len(keys)} trenches):')
    print(f'    A~H peak-snapped +-1   {ci(bs[:, 0])}   <- the human ceiling')
    print(f'    v2~A raw +-3           {ci(bs[:, 1])}')
    print(f'    v2~H raw +-3           {ci(bs[:, 2])}')
    print(f'    (v2~A) - (A~H)         {ci(bs[:, 1] - bs[:, 0])}   >0: v2 agrees with A better '
          f'than a second human does')
    print(f'    (v2~A) - (v2~H)        {ci(bs[:, 1] - bs[:, 2])}')

    print('\n6. disputes (values at the peak cell-frame)')
    ia, ih = {a for a, _ in pairs}, {b for _, b in pairs}
    As['cls'] = np.where(As.index.isin(ia), 'both', 'A_only')
    Hs['cls'] = np.where(Hs.index.isin(ih), 'both', 'H_only')
    ev = pd.concat([As, Hs[Hs.cls == 'H_only']], ignore_index=True)
    check(ev.cls.value_counts().to_dict(), EXPECTED['cls'], 'event classes')
    feat = [F + '__pct', 'score', F, 'fiji_mean__fold_back5', F + '__fold_back5']
    ev = ev.drop(columns=[c for c in feat if c in ev]).merge(X[B.FRAME_KEY + feat],
                                                             on=B.FRAME_KEY, how='left')
    ev = B.add_relative_y(ev, X)
    cols = feat + ['centroid_y_rel', 'n_cells_in_frame', 'fpos']
    rows = []
    for c in cols:
        a, h, b = (ev.loc[ev.cls == k, c].dropna() for k in ('A_only', 'H_only', 'both'))
        rows.append({'feature': c, 'both': b.median(), 'A_only': a.median(),
                     'H_only': h.median(), 'p': stats.mannwhitneyu(a, h).pvalue})
    tab = pd.DataFrame(rows)
    order = tab.p.sort_values().index
    holm = (tab.p[order] * np.arange(len(tab), 0, -1)).cummax().clip(upper=1)
    tab['p_holm'] = holm.reindex(tab.index)
    print(tab.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    for k in ('both', 'A_only', 'H_only'):
        e = ev[ev.cls == k]
        print(f'  {k:6s}: v2 score >= 0.75 at the peak {np.mean(e.score >= 0.75):.3f}; '
              f'{e.gene.value_counts().to_dict()}')
    keep = B.FRAME_KEY + ['gene', 'cls', 'viewer_frame', 'off', 'frac_of_peak'] + cols
    ev[keep].sort_values(B.FRAME_KEY).to_csv(OUT_EVENTS, index=False)
    print(f'  wrote {OUT_EVENTS} ({len(ev)} events)')

    print('\n7. definitions against each reference')
    E = {v: (v2 if v == 'v2' else load_events(v, geo, scope)) for v in ('v1', 'v2', 'v3')}
    refs = {'A raw': A, 'H raw': H, 'A peak-snapped': As, 'H peak-snapped': Hs,
            'consensus A∩H': As[As.cls == 'both'], 'union A∪H': ev}
    Eb = {v: by_trench(e, keys) for v, e in E.items()}
    for tol in (1, 3):
        print(f'  +-{tol}            n     v1     v2     v3   v2 - v3 (trench bootstrap)')
        for name, L in refs.items():
            Lb = by_trench(L, keys)
            per = {k: {v: counts(Eb[v][k], Lb[k], tol, Lmed) for v in E} for k in keys}
            tot = {v: sum(per[k][v] for k in keys) for v in E}
            d = bootstrap(per, lambda t: f1(t['v2']) - f1(t['v3']), rng, n=N_BOOT // 2)
            print(f'  {name:15s} {len(L):4d}  {f1(tot["v1"]):.3f}  {f1(tot["v2"]):.3f}  '
                  f'{f1(tot["v3"]):.3f}   {ci(d)}')

    print('\n8. v2 span recall: label inside [first_frame - 1, last_frame + 1] of a v2 event')
    for nm, L in (('A', A), ('H', H)):
        parts = []
        for g in G + ['all']:
            Lg = L if g == 'all' else L[L.gene == g]
            Eg = v2 if g == 'all' else v2[v2.gene == g]
            eb, tp = dict(list(Eg.groupby(B.KYMO_KEY, sort=False))), 0
            for k, l in Lg.groupby(B.KYMO_KEY, sort=False):
                e = eb.get(k)
                if e is None:
                    continue
                t = l.time_frame.to_numpy()[:, None]
                ok = ((t >= e.first_frame.to_numpy()[None, :] - 1)
                      & (t <= e.last_frame.to_numpy()[None, :] + 1)
                      & (np.abs(l.centroid_y.to_numpy()[:, None] - e.centroid_y.to_numpy()[None, :])
                         <= 1.5 * Lmed[k] + 10 * PEAK_WINDOW))
                if ok.any():
                    ri, ci_ = linear_sum_assignment(np.where(ok, 0, 1))
                    tp += int(ok[ri, ci_].sum())
            parts.append(f'{g} {tp}/{len(Lg)} = {tp / len(Lg):.3f}')
        print(f'  {nm}: ' + ' | '.join(parts))

    print(f'\n{"ALL PASS" if not FAILED else f"{len(FAILED)} FAILURE(S): {FAILED}"}')
    return 1 if FAILED else 0


if __name__ == '__main__':
    raise SystemExit(main())
