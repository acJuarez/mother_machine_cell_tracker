"""Stage 4: Trackastra lineage tracking, one trench at a time.

Tracks the hand-corrected masks from stage 3b against the phase stack and writes, per
trench, into `output_dir`:

    {exp}_ctc_tracks_{fov}_{peak}.csv          label, t1, t2, parent -- the lineage
    {exp}_ctc_masks_{fov}_{peak}.npy           relabelled tracked masks
    {exp}_imgs_{fov}_{peak}.npy                the trimmed phase stack
    {exp}_napari_tracks_{fov}_{peak}.npy       display only
    {exp}_napari_tracks_graph_{fov}_{peak}.json

STAGE 3B IS MANDATORY. The input mask is `napari_corrections/{fov}_{peak}_corrected.tif`,
not the raw `mm3_segmented_*.tif`. A trench with no corrected mask is skipped with a
warning, so a missing 3b pass shows up as absent output, never as an error.

The stack is trimmed to `[start:end]` before tracking, and `end` defaults to T-1, so the
last frame is dropped. Stage 5 must be given the SAME time_range_dict: it re-slices the
fluorescence TIFF with those indices while taking phase from the already-trimmed
`imgs.npy`, and a mismatched `start` silently pairs each mask with the wrong frame.
"""
import argparse
import json
import os
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import tifffile

DEFAULT_OUTPUT_DIR = ('/oak/stanford/groups/mcovert/Instruments/Covert-lab-scope1/'
                      'track_test/track_astra_output')
MODEL_NAME = "general_2d_w_SAM2_features"


def get_tiff_frame_count(file_path):
    """Index of the last time frame (T-1), or 0 if the file cannot be read."""
    try:
        shape = tifffile.imread(file_path).shape
        return shape[0] - 1 if len(shape) >= 3 else 0
    except FileNotFoundError:
        print(f"Error: TIFF file not found at {file_path}")
        return 0
    except Exception as e:
        print(f"Error reading TIFF file {file_path}: {e}")
        return 0


def run_trackastra(base_path, time_range_dict, output_dir=DEFAULT_OUTPUT_DIR,
                   phase_c='0', device="automatic"):
    """Track every (experiment, FOV, trench) named in `time_range_dict`.

    `time_range_dict` is a JSON string or a dict:
        {"Exp_name": {"FOV": {"Peak_ID": {"start": 62, "end": null}}}}
    """
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks

    if isinstance(time_range_dict, str):
        time_range_dict = json.loads(time_range_dict)
    os.makedirs(output_dir, exist_ok=True)
    model = Trackastra.from_pretrained(MODEL_NAME, device=device)

    for folder, fov_dict in time_range_dict.items():
        for fov_id, trench_time_ranges in fov_dict.items():
            print(f"  FOV: {fov_id}, Trenches: {list(trench_time_ranges.keys())}")

            for peak_id, time_info in trench_time_ranges.items():
                subtracted = os.path.join(base_path, folder, 'hyperstacked',
                                          'drift_corrected', 'rotated', 'mm_channels',
                                          'subtracted')
                path_phase = os.path.join(
                    subtracted, f'subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_c}.tif')
                path_labeled = os.path.join(
                    subtracted, 'napari_corrections', f'{fov_id}_{peak_id}_corrected.tif')

                start = time_info['start']
                end = time_info.get('end')
                if end is None:
                    end = get_tiff_frame_count(path_phase)
                    if end == 0:
                        print(f"    WARNING: no frame count for {peak_id}. Skipping.")
                        continue

                try:
                    stack_phase = tifffile.imread(path_phase)
                    stack_labeled = tifffile.imread(path_labeled)
                except FileNotFoundError as e:
                    print(f"    WARNING: required file not found for {peak_id}: {e}. "
                          f"Skipping (has stage 3b run on this trench?).")
                    continue

                print(f"{peak_id} start: {start} end: {end}")
                phase = stack_phase[start:end, :, :].copy()
                labeled = stack_labeled[start:end, :, :].copy()
                track_graph, masks_tracked = model.track(phase, labeled, mode="ilp")

                # graph_to_ctc insists on writing a CTC directory; it is not an output of
                # this stage, so it goes to a temp dir that disappears.
                with tempfile.TemporaryDirectory() as tmp:
                    ctc_tracks, ctc_masks = graph_to_ctc(track_graph, masks_tracked,
                                                         outdir=tmp)
                napari_tracks, napari_graph, _ = graph_to_napari_tracks(track_graph)

                stem = os.path.join(output_dir, f'{folder}_{{}}_{fov_id}_{peak_id}')
                ctc_tracks.to_csv(stem.format('ctc_tracks') + '.csv')
                np.save(stem.format('ctc_masks') + '.npy', ctc_masks)
                np.save(stem.format('imgs') + '.npy', phase)
                np.save(stem.format('napari_tracks') + '.npy', napari_tracks)
                with open(stem.format('napari_tracks_graph') + '.json', 'w') as f:
                    json.dump(napari_graph, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trackastra tracking algorithm")
    parser.add_argument('--base-path', required=True, type=str,
                        help="path to directory containing all microscopy experiments")
    parser.add_argument('--time-range-dict', required=True, type=str,
                        help='JSON string defining start/end frames per trench. Format: '
                             '{"Exp_name":{"FOV":{"Peak_ID":{"start": 62, "end": null}}}}')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Where to write the per-trench outputs. '
                             f'Default: {DEFAULT_OUTPUT_DIR}')
    args = parser.parse_args()
    run_trackastra(args.base_path, args.time_range_dict, args.output_dir)
