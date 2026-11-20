# this seems to be the script that regenerates all the bursts and their outputs 
# essentially just runs a few commands that then runs build_all from bursts.py
# generates all the output files in bursts.md and uses the inputs it mentioned 



"""Regenerate every burst output -- v1, v2, v3, the manual layer and the viewer union.

    conda run -n mmct --no-capture-output python 07_build_bursts.py
    conda run -n mmct --no-capture-output python 07_build_bursts.py --out-dir /tmp/rebuild

Reads the four frozen tables in `mmtrack.bursts.INPUTS` plus the v2 model, needs no network
share, and takes under a minute. Every pinned count (`mmtrack.bursts.EXPECTED`) is asserted
as it goes, so a run that finishes reproduced the shipped outputs. Then check them with
`python test_bursts.py`. `bursts.md` explains each definition.

Writes, under --out-dir (default data/):
    bursts_v1_100226.csv                    v1, the rule              161 events
    bursts_v2_100226.csv                    v2, the classifier      1,077 events
    bursts_v3_100226.csv                    v3, the monotone model  1,099 events
    bursts_manual_100226.csv                hand labels               837 (ground truth)
    burst_labels_recheck_100226.csv         the 19 labels that resolve to no cell-frame
    burst_splits_100226.csv                 v3's frozen dev/test/unreviewable trench split
    bursts_viewer_v1v2v3manual_100226.csv   the four-layer union the viewer draws
and under --models-dir (default models/):
    burst_v3_100226.joblib                  v3's model (rebuilt identically)
"""
import argparse
import warnings

from mmtrack.bursts import build_all


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out-dir', default='data', help='where the CSVs go (default: data)')
    ap.add_argument('--models-dir', default='models',
                    help='where the v3 model goes (default: models)')
    args = ap.parse_args()
    # The lineage walks take nanmax/nanmedian over cells with no descendants (the end of the
    # movie). The NaN is the intended "no forward lineage" value, so the warning is noise.
    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
    build_all(out_dir=args.out_dir, models_dir=args.models_dir)
    print('\nALL OK')


if __name__ == '__main__':
    main()
