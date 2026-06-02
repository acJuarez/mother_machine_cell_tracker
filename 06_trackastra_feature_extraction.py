"""
Stage 4+5 (Trackastra): Extract cell features from trackastra outputs and
assign lineages directly from the trackastra tracking graph.

This script replaces both 04_feature_extraction.py and 05_lineage_tracking.py
for experiments that were tracked with trackastra (run_trackastra_sherlock.py).

Workflow:
  1. Load CTC-format tracked masks and lineage graph from trackastra output dir.
  2. Extract morphological/intensity features per cell per time frame.
  3. Assign hierarchical lineage IDs from the trackastra division graph.
  4. Save phase, fluor, and mask kymograph TIFFs (same paths as Stage 4).
  5. Save per-experiment DataFrame: trackastra_cell_data_{exp}.pkl

Usage:
  poetry run python 06_trackastra_feature_extraction.py \\
      --base-path path to where experiments are stored  \\
      --trackastra-dir path to trackastra outputs \\
      --time-range-dict '{"EXP_name":{"FOV":{"peak_id":{"start":0,"end":null}}}}' \\
"""

import argparse
import glob
import json
import os

import pandas as pd
import tifffile

from mmtrack import plot_cells
from mmtrack.trackastra_feature_extraction import run_trackastra_feature_extraction


def plot_trackastra_kymographs(base_path, df):
    """
    Plot lineage-annotated kymographs for all experiment/FOV/trench combos
    found in df. Kymograph TIFFs must already exist (written during extraction).
    """
    unique_views = df[['experiment_name', 'FOV', 'trench_id']].drop_duplicates()

    for _, row in unique_views.iterrows():
        exp, fov, trench = row['experiment_name'], row['FOV'], row['trench_id']

        kymo_base = os.path.join(
            base_path, exp, 'hyperstacked', 'drift_corrected',
            'rotated', 'mm_channels', 'subtracted'
        )
        path_phase_kymo = os.path.join(kymo_base, f'{fov}_{trench}.tif')
        path_fluor_kymo = os.path.join(kymo_base, 'fluor_kymos', f'{fov}_{trench}.tif')

        if not (os.path.exists(path_phase_kymo) and os.path.exists(path_fluor_kymo)):
            print(f"  Skipping plot for {exp}/{fov}/{trench}: kymograph TIFFs not found.")
            continue

        phase_kymo = tifffile.imread(path_phase_kymo)
        fluor_kymo = tifffile.imread(path_fluor_kymo)

        df_view = df[
            (df['experiment_name'] == exp) &
            (df['FOV'] == fov) &
            (df['trench_id'] == trench)
        ].copy()

        print(f"  Plotting {exp} / FOV {fov} / trench {trench} "
              f"({df_view['predicted_lineage'].nunique()} lineages)...")

        plot_cells.plot_kymograph_cells_id(
            phase_kymo, fluor_kymo,
            df_view, exp, fov, trench,
            track_id_col='predicted_lineage',
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 4+5 (Trackastra): feature extraction and lineage assignment '
                    'from trackastra tracking outputs.'
    )

    parser.add_argument(
        '--base-path', required=True, type=str,
        help='Root directory containing all experiment folders '
             '(same as --base-dir in Stages 1-5).'
    )
    parser.add_argument(
        '--trackastra-dir', required=True, type=str,
        help='Directory holding trackastra output files '
             '(*_ctc_masks_*.npy, *_napari_tracks_graph_*.json, *_imgs_*.npy).'
    )
    parser.add_argument(
        '--time-range-dict', required=False, type=str, default='',
        help='JSON string matching the dict used when running trackastra. '
             'If omitted, outputs are auto-discovered (assumes start=0). '
             'Format: {"exp_folder": {"FOV": {"peak": {"start":0, "end":null}}}}'
    )
    parser.add_argument(
        '--phase-channel', type=str, default='0',
        help='Phase channel index used in subtracted TIFF filenames. Default: 0.'
    )
    parser.add_argument(
        '--fluor-channel', type=str, default='1',
        help='Fluorescence channel index used in subtracted TIFF filenames. Default: 1.'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Plot lineage-annotated kymographs after extraction.'
    )

    args = parser.parse_args()

    time_range_dict = {}
    if args.time_range_dict:
        try:
            time_range_dict = json.loads(args.time_range_dict)
        except json.JSONDecodeError:
            print('ERROR: --time-range-dict is not valid JSON. Exiting.')
            raise SystemExit(1)

    run_trackastra_feature_extraction(
        base_path=args.base_path,
        trackastra_dir=args.trackastra_dir,
        time_range_dict=time_range_dict,
        phase_c_str=args.phase_channel,
        fluor_c_str=args.fluor_channel,
    )

    if args.plot:
        pkl_files = sorted(
            glob.glob(os.path.join(args.base_path, 'trackastra_cell_data_*.pkl'))
        )
        if not pkl_files:
            print('No trackastra_cell_data_*.pkl files found for plotting.')
        else:
            for pkl in pkl_files:
                print(f'\nPlotting from {pkl}')
                df = pd.read_pickle(pkl)
                plot_trackastra_kymographs(args.base_path, df)
