# Reads what 07_build_bursts.py wrote and checks it against the pinned numbers.

"""The burst regression contract: v1, v2, v3, the manual layer, the union and the viewer.

    conda run -n mmct --no-capture-output python test_bursts.py

Reads the files `07_build_bursts.py` wrote (no recompute, a few seconds, no network share) and
checks them against the pinned numbers. Run it after every build. Exit code 0 = all pass.

  1. every layer's count, v1's route split and v3's per-gene split
  2. each union column is set-identical to its own layer file
  3. no event sits in a FAILED_KYMOGRAPHS trench, and every event carries its position
  4. the two models' metadata (v2 is loaded, v3 rebuilt)
  5. v3's HOLDOUT numbers -- the test split, matched one-to-one at +-1 frame
  6. the viewer's layer invariants
"""
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd

import mmtrack.bursts as B
from mmtrack.experiment_metadata import FAILED_KYMOGRAPHS

OUT = {k: os.path.join('models' if k == 'v3_model' else 'data', v)
       for k, v in B.OUTPUTS.items()}
EXCLUDED = {tuple(t) for t in FAILED_KYMOGRAPHS}
V3_TRENCHES = 228
V3_BY_GENE = {'baeS': 270, 'gfcE': 179, 'gldA': 116, 'araB': 111, 'chpS': 110,
              'alkA': 109, 'lacZ': 95, 'mazF': 56, 'hupA': 19}
# v3 on the 57 test trenches, measured once and never tuned against: P 0.560 R 0.700
HOLDOUT = {'n_labels': 213, 'n_events': 266, 'tp': 149, 'fp_empty': 2}
FAILED = []


def check(ok, msg):
    print(f'  {"PASS" if ok else "FAIL"}  {msg}')
    if not ok:
        FAILED.append(msg)


def read(path):
    d = pd.read_csv(path, dtype=B.READ_DTYPES)
    d['FOV'] = d['FOV'].str.zfill(3)
    return d


def keyset(df):
    return set(map(tuple, df[B.FRAME_KEY].astype({'track_id': int, 'time_frame': int}).values))


match_one_to_one = B.match_one_to_one


def main():
    E = B.EXPECTED
    layers = {v: read(OUT[v]) for v in ('v1', 'v2', 'v3', 'manual')}

    print('1. counts')
    for v, d in layers.items():
        check(len(d) == E[v], f'{v}: {len(d)} rows (want {E[v]})')
    got = layers['v1']['route'].value_counts().to_dict()
    check(got == E['v1_routes'], f'v1 routes {got}')
    v3 = layers['v3']
    check(v3.groupby(B.KYMO_KEY).ngroups == V3_TRENCHES,
          f'v3 events span {v3.groupby(B.KYMO_KEY).ngroups} trenches (want {V3_TRENCHES})')
    check(v3.gene.value_counts().to_dict() == V3_BY_GENE, 'v3 per-gene split')
    sp = read(OUT['splits'])
    got = sp['split'].value_counts().to_dict()
    check(got == {'dev': 167, 'test': 57, 'unreviewable': 19}, f'splits {got}')
    check(len(read(OUT['unresolved_labels'])) == E['unresolved_labels'],
          f'{E["unresolved_labels"]} unresolvable hand labels')

    print('2. the union')
    un = read(OUT['union'])
    check(len(un) == E['union'], f'{len(un)} union rows (want {E["union"]})')
    check(un[B.FRAME_KEY].duplicated().sum() == 0, 'no duplicate cell-frame keys')
    for v, d in layers.items():
        check(keyset(un[un[f'burst_{v}'].astype(bool)]) == keyset(d),
              f'burst_{v} is set-identical to {OUT[v]}')

    print('3. exclusions and positions')
    for v, d in layers.items():
        check(not (set(zip(d.experiment_name, d.FOV, d.trench_id)) & EXCLUDED),
              f'no {v} event in an excluded kymograph')
    for v in ('v1', 'v3', 'manual'):
        check(layers[v]['centroid_y_rel'].notna().all(), f'every {v} event has centroid_y_rel')

    print('4. models')
    m2 = joblib.load(B.INPUTS['v2_model'])
    check(m2['threshold'] == 0.75 and len(m2['feature_cols']) == 43,
          f'v2: threshold {m2["threshold"]}, {len(m2["feature_cols"])} features')
    m3 = joblib.load(OUT['v3_model'])
    check(m3['monotone'] is True and m3['threshold'] == B.V3_THRESHOLD
          and len(m3['feature_cols']) == 38 and m3['mask_window'] == B.V3_MASK_WINDOW,
          f'v3: monotone {m3["monotone"]}, threshold {m3["threshold"]}, '
          f'{len(m3["feature_cols"])} features, mask_window {m3["mask_window"]}')
    n_pos = sum(1 for s in m3['monotone_signs'].values() if s == 1)
    check(n_pos == 26, f'v3: {n_pos} features constrained +1 (want 26)')

    print('5. v3 holdout')
    test = set(map(tuple, sp.loc[sp['split'] == 'test', B.KYMO_KEY].values))
    in_test = lambda d: d[[tuple(r) in test for r in d[B.KYMO_KEY].values]]
    L, Ev = in_test(layers['manual']), in_test(v3)
    tp = match_one_to_one(Ev, L)
    # a certified-empty trench: reviewed gene, zero labels
    empty = set(map(tuple, sp.loc[(sp.split == 'test') & (sp.n_labels == 0),
                                  B.KYMO_KEY].values))
    fp = sum(1 for k in Ev[B.KYMO_KEY].values if tuple(k) in empty)
    got = {'n_labels': len(L), 'n_events': len(Ev), 'tp': tp, 'fp_empty': fp}
    check(got == HOLDOUT, f'{got}')
    p, r = tp / len(Ev), tp / len(L)
    print(f'        -> P {p:.3f}  R {r:.3f}  F1 {2 * p * r / (p + r):.3f} '
          f'({len(empty)} certified-empty test trenches)')

    print('6. the viewer')
    import importlib.util
    spec = importlib.util.spec_from_file_location('gkv', 'generate_kymograph_viewer.py')
    gkv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gkv)
    V, C, W = gkv.BURST_VERSIONS, gkv.BURST_COLORS, gkv.BURST_WIDTHS
    check(V == ('manual', 'v1', 'v2', 'v3'), f'four layers, widest first {V}')
    ws = [W[v] for v in V]
    check(all(a > b for a, b in zip(ws, ws[1:])),
          f'widths strictly decreasing {ws} -- agreeing layers draw the same contour, so '
          f'only nested strokes keep them distinguishable')
    check(set(V) <= set(C), 'every layer has a colour')
    # every layer set, with its generated controls and JS array filled in
    for name in gkv.BURST_LAYER_SETS:
        Vs, Cs, Ws = gkv._layer_globals(name)
        t = (gkv.HTML_TEMPLATE.replace('{burst_controls}', gkv.burst_controls_html(name))
             .replace('{burst_layers_js}', gkv.burst_layers_js(name)))
        seg = t[t.index('<div class="controls'):t.index('{gene_panels}')]
        check(seg.count('<div') == seg.count('</div>'),
              f'[{name}] .controls is balanced (an unclosed group kills the sticky bar and scrolling)')
        names = re.findall(r"'([a-z0-9]+)'",
                           re.search(r'const layers = \[(.*?)\];', t, re.S).group(1))
        ids = set(re.findall(r'id="chk-([a-z0-9]+)"', t))
        check(not [n for n in names if n not in ids], f'[{name}] every JS layer has a checkbox')
        check(all(f'burst{v}' in names and f'burst{v}' in ids for v in Vs),
              f'[{name}] every layer {Vs} has a checkbox AND a JS entry')

    print(f'\n{"ALL PASS" if not FAILED else f"{len(FAILED)} FAILURE(S)"}')
    for m in FAILED:
        print(f'  - {m}')
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
