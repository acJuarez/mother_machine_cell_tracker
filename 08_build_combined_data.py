"""
Stage 8: combine the per-experiment trackastra pickles into one annotated table.

Stage 6 writes one `trackastra_cell_data_{exp}.pkl` per experiment. This concatenates
them and adds the per-experiment annotations that stage 6 does not know about:

  gene                 from the strain/experiment/FOV mapping
  phase_exposure_ms    phase channel exposure, milliseconds
  fluor_exposure_ms    fluorescence channel exposure, milliseconds
  minutes_per_frame    acquisition interval
  time_min             time_frame * minutes_per_frame

All four come from mmtrack.experiment_metadata, which is the single source of truth
shared with the viewer and the analysis notebook.

Usage:
  poetry run python 08_build_combined_data.py \\
      --base-path <dir holding trackastra_cell_data_*.pkl> \\
      [--time-dict data/time_dict_082026.json] \\
      [--out-csv data/all_data_082026.csv]
"""

import argparse
import glob
import json
import os

import pandas as pd

from mmtrack.experiment_metadata import (
    EXPOSURE_MS,
    MINUTES_INTERVAL,
    STRAIN_EXP_DICT,
)

PKL_PREFIX = 'trackastra_cell_data_'


def load_pickles(base_path, time_dict_path):
    """Concatenate the per-experiment pickles that belong to the current generation.

    Pickles for experiments absent from the time_dict are skipped rather than silently
    folded in: reprocessing a trench can renumber its peak ids, so a leftover pickle
    from an earlier generation would contribute rows keyed to peak ids that no longer
    exist.
    """
    known = set(json.load(open(time_dict_path)))
    paths = sorted(glob.glob(os.path.join(base_path, f'{PKL_PREFIX}*.pkl')))
    if not paths:
        raise SystemExit(f'No {PKL_PREFIX}*.pkl found in {base_path}. Run stage 6 first.')

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


def annotate(df):
    """Add gene, exposure and time columns in place, and return the DataFrame."""
    # FOV is a zero-padded string in the pickles (it comes from the time_dict key), but
    # a CSV round-trip turns '005' into the integer 5. Normalize both sides so the match
    # holds regardless of which source the frame came from.
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


def build(base_path, out_csv, time_dict_path):
    df = annotate(load_pickles(base_path, time_dict_path))
    report_gaps(df)

    out_dir = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_trenches = df.groupby(['experiment_name', 'FOV', 'trench_id']).ngroups
    size_mb = os.path.getsize(out_csv) / (1024 * 1024)
    print(f'\nWrote {out_csv} ({size_mb:.1f} MB)')
    print(f'  {len(df):,} rows | {n_trenches} trenches | '
          f'{df["experiment_name"].nunique()} experiments | {len(df.columns)} columns')
    print('\nTrenches per gene:')
    per_gene = (df[['gene', 'experiment_name', 'FOV', 'trench_id']]
                .drop_duplicates().groupby('gene').size())
    print(per_gene.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 8: combine trackastra per-experiment pickles into one '
                    'gene- and exposure-annotated table.'
    )
    parser.add_argument(
        '--base-path', required=True, type=str,
        help='Directory holding the trackastra_cell_data_*.pkl files written by stage 6.'
    )
    parser.add_argument(
        '--time-dict', type=str, default='data/time_dict_082026.json',
        help='time_dict JSON defining the current generation of experiments. Pickles for '
             'experiments not listed here are skipped. Default: data/time_dict_082026.json'
    )
    parser.add_argument(
        '--out-csv', type=str, default='data/all_data_082026.csv',
        help='Output CSV path. Default: data/all_data_082026.csv'
    )

    args = parser.parse_args()
    build(args.base_path, args.out_csv, args.time_dict)
