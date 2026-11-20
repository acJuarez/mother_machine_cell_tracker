"""Stage 4: run Trackastra on the hand-corrected masks to get lineages.

    conda run -n mmct --no-capture-output python 04_trackastra_tracking.py \
        --base-path <experiments dir> \
        --time-range-dict '{"Exp":{"FOV":{"Peak":{"start":0,"end":null}}}}'

On Sherlock this is driven per trench by Slurm/submit_trackastra_jobs.py, which submits one
sbatch job per (experiment, FOV, trench) through Slurm/track_astra.batch.

Input is napari_corrections/{FOV}_{peak}_corrected.tif from stage 3b, so stage 3b must have
run on every trench to be tracked; a trench without one is skipped with a warning. Output is
five files per trench under --output-dir, which stage 5 reads.

Keep the same --time-range-dict for stage 5: `start` trims the stack here and stage 5
re-slices the fluorescence TIFF with it.
"""
import argparse

from mmtrack.run_trackastra import DEFAULT_OUTPUT_DIR, run_trackastra

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-path', required=True,
                   help='Directory containing all microscopy experiments.')
    p.add_argument('--time-range-dict', required=True,
                   help='JSON: {"Exp":{"FOV":{"Peak":{"start":N,"end":N|null}}}}. '
                        '`end` null means the last frame of the phase stack.')
    p.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                   help=f'Where to write the per-trench outputs. Default: {DEFAULT_OUTPUT_DIR}')
    p.add_argument('--phase-channel', default='0')
    args = p.parse_args()
    run_trackastra(args.base_path, args.time_range_dict, args.output_dir,
                   phase_c=args.phase_channel)
