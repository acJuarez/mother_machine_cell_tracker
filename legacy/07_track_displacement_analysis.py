"""
Stage 7: Track y-displacement QC + fluorescence-change analysis.

In Mother Machine trenches, cells only move along the trench y-axis between frames.
This stage pairs cells across frames, flags likely-mis-tracked pairs from the
y-displacement (Δy), and on the retained good pairs measures the change in average
fluorescence (Δfluor).

Two pair populations are built, tagged by the `pair_type` / `is_division` columns:

  within_track  consecutive frames of one track_id. Flagged when |Δy| exceeds one of its
                own cell lengths (--dy-length-k, default 1.0): a tracking error means the
                track landed on a different cell, and in a trench of stacked cells that
                takes roughly a cell length of apparent motion.
  division      parent cell A's last frame vs. each daughter's first frame. Previously
                these were not examined at all, so a mis-called division -- or a
                fluorescence burst landing exactly on a division frame -- was invisible.
                Flagged when |Δy| exceeds --division-length-k (default 1.5) times cell
                A's OWN length: a real division displaces the centroid further than one
                frame of growth, since each daughter starts about a quarter of a parent
                length off A's centroid on top of the usual advection.

Δy is advective and its spread scales with the frame interval, so a fixed pixel window is
not comparable across 5- and 10-minute acquisitions; normalising by cell length removes
the dependence on magnification and cell size, and is the only criterion this stage
applies. The per-row cutoff actually applied is recorded in `dy_threshold_cell_lengths`.

Consumes the aggregated cell×frame CSV produced during postprocessing
(data/all_data_082026.csv), which already carries a `gene` column.

Usage:
  poetry run python 07_track_displacement_analysis.py \\
      --gene lacZ \\
      [--data-csv data/all_data_082026.csv] \\
      [--out-dir notebooks/figures] \\
      [--dy-length-k 1.0] [--division-length-k 1.5] [--skip-divisions] \\
      [--bins 60]

Outputs (into --out-dir):
  {gene}_delta_y_hist.svg/.png           Δy histogram, within_track pairs
  {gene}_delta_fluor_hist.svg/.png       Δfluor histogram (all vs good within_track)
  {gene}_dy_vs_dfluor.svg/.png           Δy vs Δfluor scatter (good vs jump)
  {gene}_division_delta_y_hist.svg/.png  Δy histogram, division pairs (1.5× cutoff)
  {gene}_division_dy_vs_dfluor.svg/.png  Δy vs Δfluor scatter, division pairs
  {gene}_track_pairs.csv                 every pair, both types, with delta_y,
                                         delta_fluor, is_jump, pair_type, is_division
                                         (redirect with --pairs-csv)

The pair table carries the full cell identity (experiment_name, FOV, trench_id,
track_id, gene, predicted_lineage, node_id_from/to) plus from_track_id/to_track_id, the
per-endpoint merge keys. Division rows additionally carry parent_track_id,
sibling_track_id and n_siblings.

MERGE KEY: every row carries `from_track_id` and `to_track_id`, so each endpoint merges
back onto the cell×frame CSV on its own key and neither pair type needs special
handling (FOV is zero-padded in both):

  from = (experiment_name, FOV, trench_id, from_track_id, frame_from)
  to   = (experiment_name, FOV, trench_id, to_track_id,   frame_to)

They are equal on a within_track row and differ on a division row, where frame_from is
the PARENT's frame while track_id is the DAUGHTER's. `time_frame` (== frame_from, NaN on
division rows) is kept for the older 1:1 within_track merge, but a merge on it silently
drops every division row.

All genes in one run, writing the combined table next to the cell data:
  poetry run python 07_track_displacement_analysis.py --gene all \\
      --pairs-csv data/track_pairs_082026.csv
"""

import argparse

# `legacy/mmtrack/` is not a package; its modules are loaded by path. Run from the repo root.
def _legacy(module_name):
    import importlib.util, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'mmtrack', module_name + '.py')
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

run_track_displacement_analysis = _legacy('track_displacement').run_track_displacement_analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 7: per-track y-displacement QC and fluorescence-change '
                    'analysis from tracked cell data.'
    )
    parser.add_argument(
        '--data-csv', type=str, default='data/all_data_082026.csv',
        help='Aggregated cell×frame CSV (default: data/all_data_082026.csv).'
    )
    parser.add_argument(
        '--gene', type=str, default='lacZ',
        help='Gene/strain label to analyze (default: lacZ). Use "all" for no filter.'
    )
    parser.add_argument(
        '--out-dir', type=str, default='notebooks/figures',
        help='Directory for output figures and CSVs (default: notebooks/figures).'
    )
    parser.add_argument(
        '--pairs-csv', type=str, default=None,
        help='Explicit path for the pair table, overriding '
             '{out-dir}/{gene}_track_pairs.csv. Use this to drop the combined table '
             'next to the cell data, e.g. data/track_pairs_082026.csv.'
    )
    parser.add_argument(
        '--dy-length-k', type=float, default=1.0,
        help='Jump criterion: |Δy| > k × the pair\'s own cell length (mean '
             'axis_major_length of the two endpoints). A tracking error means the track '
             'landed on a different cell, which takes roughly one cell length of motion, '
             'so k=1.0 is the physical threshold and is independent of magnification, '
             'cell size and frame interval. Must be positive. Default: 1.0.'
    )
    parser.add_argument(
        '--division-length-k', type=float, default=1.5,
        help='Jump criterion for parent->daughter division pairs: |Δy| > k × cell A\'s '
             'own axis_major_length at its last frame. Wider than --dy-length-k because a '
             'real division displaces the centroid further than one frame of growth. '
             'Default: 1.5.'
    )
    parser.add_argument(
        '--skip-divisions', action='store_true',
        help='Only build within-track pairs, restoring the pre-division-check behaviour.'
    )
    parser.add_argument(
        '--fluor-col', type=str, default='intensity_mean_fluor',
        help='Fluorescence column to difference (default: intensity_mean_fluor).'
    )
    parser.add_argument(
        '--bins', type=int, default=60,
        help='Histogram bin count (default: 60).'
    )
    parser.add_argument(
        '--fluor-method', type=str, default='none',
        choices=['none', 'median', 'mean', 'gaussian', 'nlm', 'otsu'],
        help="Recompute fluorescence on filtered images before the fluor-change step. "
             "'none' uses the CSV's (unfiltered) intensity_mean_fluor. Outputs are "
             "named with the method. Default: none."
    )
    parser.add_argument(
        '--base-dir', type=str,
        default='/Volumes/mcovert/Instruments/Covert-lab-scope1/subgen_processed_data',
        help='Root dir with subtracted fluor TIFFs (for --fluor-method != none).'
    )
    parser.add_argument(
        '--trackastra-dir', type=str,
        default='/Volumes/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output',
        help='Dir with trackastra ctc_masks .npy outputs (for --fluor-method != none).'
    )
    parser.add_argument(
        '--time-dict', type=str, default='data/time_dict_082026.json',
        help='JSON time_dict giving per-trench start offsets (for filtered recompute).'
    )
    parser.add_argument('--filter-size', type=int, default=3,
                        help='median/mean/otsu footprint size (default: 3).')
    parser.add_argument('--filter-sigma', type=float, default=1.0,
                        help='gaussian sigma (default: 1.0).')
    parser.add_argument('--nlm-h', type=float, default=1.15,
                        help='non-local means strength multiplier (default: 1.15).')
    parser.add_argument('--otsu-scale', type=float, default=1.0,
                        help='global Otsu threshold multiplier (default: 1.0).')

    args = parser.parse_args()

    gene = None if args.gene.lower() == 'all' else args.gene

    run_track_displacement_analysis(
        csv_path=args.data_csv,
        gene=gene,
        out_dir=args.out_dir,
        dy_length_k=args.dy_length_k,
        division_length_k=args.division_length_k,
        include_divisions=not args.skip_divisions,
        fluor_col=args.fluor_col,
        bins=args.bins,
        fluor_method=args.fluor_method,
        base_dir=args.base_dir,
        trackastra_dir=args.trackastra_dir,
        time_dict_path=args.time_dict,
        filter_size=args.filter_size,
        filter_sigma=args.filter_sigma,
        nlm_h=args.nlm_h,
        otsu_scale=args.otsu_scale,
        pairs_csv=args.pairs_csv,
    )
