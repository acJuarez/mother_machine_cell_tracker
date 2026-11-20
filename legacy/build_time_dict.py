# builds time_dict JSON from the image_segmentation kymographs sheet that was manually labeled 
# probably does not need to be its own script and can be embedded elsewhere when it is actually needed. 
# this is really just creatring the larger time_dict, since was originally just a function in a jupytr notebook

"""
The time_dict is the frame window for every trench that has been segmented:

    {experiment_name: {FOV: {trench_id: {"start": int, "end": int | None}}}}

Usage:
  python build_time_dict.py \\
      --sheet-csv data/image_segmentation_kymographs_081326.csv \\
      --out data/time_dict_082026.json \\
      --compare data/time_dict_080726.json
"""

import argparse
import json
import os

import pandas as pd

# The sheet's own column spelling, typo and trailing space included.
COL_TRACKED = 'trackasrta? '
COL_SEGMENTED = 'resegmented?'


def _as_bool(series):
    """Coerce the sheet's TRUE/FALSE text column to real booleans."""
    return series.astype(str).str.strip().str.upper() == 'TRUE'


def _as_frame(value):
    """Sheet frame number -> int, or None for the literal 'null'/blank open end."""
    text = str(value).strip()
    if text in ('', 'null', 'nan', 'NaN', 'None'):
        return None
    return int(float(text))


def build_time_dict(sheet_csv, untracked_only=False):
    """Return the nested time_dict for every segmented trench in the sheet."""
    sheet = pd.read_csv(sheet_csv)

    rows = sheet.dropna(subset=['Start_frame'])
    rows = rows[_as_bool(rows[COL_SEGMENTED])]
    if untracked_only:
        rows = rows[~_as_bool(rows[COL_TRACKED])]

    time_dict = {}
    for _, row in rows.iterrows():
        # FOV is unpadded in the sheet ('9') but zero-padded everywhere on disk ('009').
        fov = str(row['FOV']).strip().zfill(3)
        trench = str(row['trench_id']).strip()
        exp = str(row['experiment_name']).strip()

        start = _as_frame(row['Start_frame'])
        if start is None:
            continue
        time_dict.setdefault(exp, {}).setdefault(fov, {})[trench] = {
            'start': start,
            'end': _as_frame(row['End_frame']),
        }

    return time_dict


def count_peaks(time_dict):
    return sum(len(peaks) for fovs in time_dict.values() for peaks in fovs.values())


def report_diff(new, old, old_path):
    """Print the peak-level difference against a previous time_dict.

    Dropped peaks matter: an experiment or trench that quietly disappears here
    disappears from the combined CSV and the viewer too.
    """
    def peaks(d):
        return {(exp, fov, pk): info
                for exp, fovs in d.items()
                for fov, pks in fovs.items()
                for pk, info in pks.items()}

    new_p, old_p = peaks(new), peaks(old)

    print(f'\nDiff against {old_path} '
          f'({len(old)} experiments / {len(old_p)} peaks):')

    for label, keys in (('ADDED', sorted(set(new_p) - set(old_p))),
                        ('DROPPED', sorted(set(old_p) - set(new_p)))):
        if not keys:
            continue
        print(f'  {label} {len(keys)}:')
        for exp, fov, pk in keys:
            info = (new_p if label == 'ADDED' else old_p)[(exp, fov, pk)]
            print(f'    {exp} / FOV {fov} / trench {pk}  {info}')

    changed = [k for k in set(new_p) & set(old_p) if new_p[k] != old_p[k]]
    if changed:
        print(f'  CHANGED {len(changed)}:')
        for k in sorted(changed):
            print(f'    {k[0]} / FOV {k[1]} / trench {k[2]}: '
                  f'{old_p[k]} -> {new_p[k]}')

    if not (set(new_p) ^ set(old_p)) and not changed:
        print('  identical')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build a time_dict JSON from the image_segmentation_kymographs '
                    'sheet export.'
    )
    parser.add_argument(
        '--sheet-csv', type=str,
        default='data/image_segmentation_kymographs_081326.csv',
        help='Sheet export to read. Default: '
             'data/image_segmentation_kymographs_081326.csv'
    )
    parser.add_argument(
        '--out', required=True, type=str,
        help='Output JSON path, e.g. data/time_dict_082026.json'
    )
    parser.add_argument(
        '--compare', type=str, default=None,
        help='Previous time_dict JSON to diff against, so added/dropped/changed '
             'trenches are visible instead of silent.'
    )
    parser.add_argument(
        '--untracked-only', action='store_true',
        help=f'Emit only trenches with {COL_TRACKED.strip()!r} == FALSE, i.e. the '
             'delta still needing a trackastra run (input for '
             'Slurm/submit_trackastra_jobs.py). NOT suitable for stages 6/8 or the '
             'viewer, which treat this dict as the full whitelist.'
    )

    args = parser.parse_args()

    time_dict = build_time_dict(args.sheet_csv, untracked_only=args.untracked_only)
    if not time_dict:
        raise SystemExit(f'No segmented trenches found in {args.sheet_csv}.')

    print(f'{args.sheet_csv}: {len(time_dict)} experiments / '
          f'{count_peaks(time_dict)} peaks'
          + ('  (untracked-only)' if args.untracked_only else ''))

    if args.compare:
        with open(args.compare) as fh:
            report_diff(time_dict, json.load(fh), args.compare)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(time_dict, fh, separators=(',', ':'))

    print(f'\nWrote {args.out}')
