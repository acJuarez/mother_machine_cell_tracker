"""
Stage 7: Track y-displacement QC + fluorescence-change analysis.

In Mother Machine trenches, cells only move along the trench y-axis between frames.
This stage breaks every track into consecutive-frame pairs (connected by track_id),
uses the y-displacement (Δy) distribution to flag likely-mis-tracked pairs (big
jumps), and on the retained good pairs measures the change in average fluorescence
(Δfluor) to surface cells with significant fluorescence changes.

Consumes the aggregated cell×frame CSV produced during postprocessing
(data/all_data_070926.csv), which already carries a `gene` column.

Usage:
  poetry run python 07_track_displacement_analysis.py \\
      --gene lacZ \\
      [--data-csv data/all_data_070926.csv] \\
      [--out-dir notebooks/figures] \\
      [--dy-k 4.0] [--dy-abs-cutoff 40] \\
      [--fluor-k 4.0] [--bins 60]

Outputs (into --out-dir):
  {gene}_delta_y_hist.svg/.png        Δy histogram with jump cutoffs
  {gene}_delta_fluor_hist.svg/.png    Δfluor histogram (all vs good pairs)
  {gene}_dy_vs_dfluor.svg/.png        Δy vs Δfluor scatter (good vs jump)
  {gene}_track_pairs.csv              every pair with delta_y, delta_fluor, is_jump
  {gene}_significant_fluor_pairs.csv  good pairs flagged significant (for viewer QC)
"""

import argparse

from mmtrack.track_displacement import run_track_displacement_analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 7: per-track y-displacement QC and fluorescence-change '
                    'analysis from tracked cell data.'
    )
    parser.add_argument(
        '--data-csv', type=str, default='data/all_data_070926.csv',
        help='Aggregated cell×frame CSV (default: data/all_data_070926.csv).'
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
        '--dy-k', type=float, default=4.0,
        help='Robust cutoff multiplier for Δy jumps: keep |Δy - median| <= k·(1.4826·MAD). '
             'Default: 4.0.'
    )
    parser.add_argument(
        '--dy-abs-cutoff', type=float, default=None,
        help='Optional absolute Δy cutoff in px (|Δy| > cutoff = jump). Overrides --dy-k.'
    )
    parser.add_argument(
        '--fluor-k', type=float, default=4.0,
        help='Robust z-score threshold for a significant Δfluor on good pairs. Default: 4.0.'
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
        '--time-dict', type=str, default='data/time_dict_070826.json',
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
        dy_k=args.dy_k,
        dy_abs_cutoff=args.dy_abs_cutoff,
        fluor_k=args.fluor_k,
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
    )
