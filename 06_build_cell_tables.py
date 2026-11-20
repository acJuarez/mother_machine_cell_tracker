"""Stage 6: turn the stage-5 pickles into the tables the burst build and the notebook read.

    conda run -n mmct --no-capture-output python 06_build_cell_tables.py \
        --base-path <dir holding trackastra_cell_data_*.pkl>

Writes, under --out-dir and tagged with --generation:

    all_data_<gen>.csv                combined, gene- and exposure-annotated cell table
    track_pairs_<gen>.csv             within_track + division pairs, jump-flagged
    fluor_filtered_median3_<gen>.csv  per-cell mean on median-filtered images
    fiji_fluor_<gen>.csv              per-cell values on the viewer's FIJI transform
    all_data_annotated_<gen>.csv      + the above, is_degenerate_mask, per-kymograph scale
    track_pairs_annotated_<gen>.csv   + endpoint ids and every endpoint's cell values
    divisions_<gen>.csv               one observed division per row

The two image-reading steps need /Volumes/mcovert. Point --filtered-csv and --fiji-csv at
existing files to reuse them, or pass --skip-fiji on a machine without the share. Any stage
can be skipped by naming its output: --cells-csv and --pairs-csv read instead of rebuild.

Every output is left alone if it already exists, unless --force.
"""
import argparse
import os

import pandas as pd

import mmtrack.cell_tables as CT


def _read(path, label):
    print(f'reading {label}: {path}')
    df = pd.read_csv(path, usecols=lambda c: c != 'coords', dtype=CT.DTYPES)
    df['FOV'] = df['FOV'].astype(str).str.zfill(3)
    return df


def _write(df, path, force):
    if os.path.exists(path) and not force:
        raise SystemExit(f'{path} exists; pass --force to overwrite it.')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    print(f'wrote {path}  ({len(df):,} rows x {df.shape[1]} cols, '
          f'{os.path.getsize(path) / 1e6:.1f} MB)\n')


def main(a):
    out = lambda name: os.path.join(a.out_dir, f'{name}_{a.generation}.csv')

    print('--- 6a. cells ---')
    if a.cells_csv:
        cells = _read(a.cells_csv, 'cells')
    else:
        if not a.base_path:
            raise SystemExit('pass --base-path (the stage-5 pickles) or --cells-csv.')
        cells = CT.build_cells(a.base_path, a.time_dict)
        _write(cells, out('all_data'), a.force)

    print('--- 6b. pairs ---')
    if a.pairs_csv:
        pairs = _read(a.pairs_csv, 'pairs')
    else:
        pairs = CT.build_pairs(cells, length_k=a.dy_length_k,
                               division_length_k=a.division_length_k)
        _write(pairs, out('track_pairs'), a.force)

    experiments = sorted(cells['experiment_name'].unique())

    print('\n--- 6c. filtered fluorescence ---')
    filt_path = a.filtered_csv or out('fluor_filtered_median3')
    if os.path.exists(filt_path):
        filtered = _read(filt_path, 'filtered fluor')
    else:
        print('measuring on median-filtered images; a few minutes ...')
        filtered = CT.compute_filtered_fluor(
            experiments, base_dir=a.base_dir, trackastra_dir=a.trackastra_dir,
            time_dict_path=a.time_dict)
        _write(filtered, filt_path, a.force)

    print('--- 6d. annotated tables ---')
    acells = CT.annotate_cells(cells, filtered)
    CT.check_filtered_alignment(acells)
    _write(acells, out('all_data_annotated'), a.force)

    apairs = CT.annotate_pairs(pairs, acells)
    _write(apairs, out('track_pairs_annotated'), a.force)

    print('--- 6e. divisions ---')
    # Scoped, unlike the two annotated tables: a failed acquisition or an excluded trench
    # has no usable lineage, and this table is read as a rate denominator.
    from mmtrack.bursts import drop_excluded_kymographs
    dcells = drop_excluded_kymographs(acells[~acells['is_failed_experiment']], 'cells')
    dpairs = drop_excluded_kymographs(apairs[~apairs['is_failed_experiment']], 'pairs')
    _write(CT.division_events(dpairs, dcells), out('divisions'), a.force)

    if a.skip_fiji:
        print('--- 6f. FIJI values: skipped ---')
        return
    print('--- 6f. FIJI values ---')
    fiji_path = a.fiji_csv or out('fiji_fluor')
    if os.path.exists(fiji_path) and not a.force:
        print(f'{fiji_path} exists; skipping.')
        return
    fiji = CT.compute_fiji_fluor(
        experiments, base_dir=a.base_dir, trackastra_dir=a.trackastra_dir,
        time_dict_path=a.time_dict, min_size=a.small_object_min_size)
    _write(fiji, fiji_path, a.force)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-path', help='Directory holding the stage-5 trackastra_cell_data_*.pkl.')
    p.add_argument('--out-dir', default='data')
    p.add_argument('--generation', default='082026',
                   help='Date suffix on every output filename. Default: 082026')
    p.add_argument('--time-dict', default='data/time_dict_082026.json',
                   help='Per-trench {start, end}. Pickles for experiments absent from it '
                        'are skipped, and the start offsets trim the fluor stacks.')
    p.add_argument('--cells-csv', help='Read the combined cell table instead of rebuilding it.')
    p.add_argument('--pairs-csv', help='Read the pair table instead of rebuilding it.')
    p.add_argument('--filtered-csv', help='Reuse an existing filtered-fluorescence table.')
    p.add_argument('--fiji-csv', help='Reuse an existing FIJI-values table.')
    p.add_argument('--skip-fiji', action='store_true',
                   help='Do not measure the FIJI values (they need the network share).')
    p.add_argument('--base-dir', default=CT.DEFAULT_BASE_DIR,
                   help='Root of the processed experiment folders.')
    p.add_argument('--trackastra-dir', default=CT.DEFAULT_TRACKASTRA_DIR,
                   help='Where stage 4 wrote the ctc masks.')
    p.add_argument('--dy-length-k', type=float, default=1.0,
                   help='Jump threshold for within_track pairs, in cell lengths.')
    p.add_argument('--division-length-k', type=float, default=1.5,
                   help='Jump threshold for division pairs, in cell lengths.')
    p.add_argument('--small-object-min-size', type=int, default=0,
                   help='Speckle removal on the FIJI transform. 0 reproduces FIJI exactly.')
    p.add_argument('--force', action='store_true', help='Overwrite existing outputs.')
    main(p.parse_args())
