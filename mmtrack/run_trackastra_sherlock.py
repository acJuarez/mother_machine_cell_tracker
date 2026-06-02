import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks, write_to_geff
from trackastra.data import example_data_bacteria
import tifffile
import napari
import json
import pandas as pd
import numpy as np
import argparse


def get_tiff_frame_count(file_path):
    """
    Reads a TIFF file using imread and returns the index of the last time frame (T - 1).
    NOTE: This loads the entire file into memory.
    """
    try:
        img_stack = tifffile.imread(file_path)
        shape = img_stack.shape

        if len(shape) >= 3:
            return shape[0] - 1
        else:
            return 0

    except FileNotFoundError:
        print(f"Error: TIFF file not found at {file_path}")
        return 0
    except Exception as e:
        print(f"Error reading TIFF file {file_path}: {e}")
        return 0
    
def stack_to_kymograph(stack):
    """
    takes a numpyarray of an image in the format (t, y, x) and converts it into a kymograph
    """

    kymograph_gray = []
    for i in range(stack.shape[0]):
        frame = stack[i]
        if frame.ndim == 3:
            kymograph_gray.append(frame, axis=2)
        else:
            kymograph_gray.append(frame)
    
    kymograph = np.concatenate(kymograph_gray, axis =1)
    return kymograph

def run_track_astra(base_path, time_range_dict):
    time_range_dict = json.loads(time_range_dict)
    print(len(time_range_dict))
    
    phase_c_str = '0'
    fluor_c_str = '1'
    
    device = "automatic" 
    model = Trackastra.from_pretrained("general_2d_w_SAM2_features", device=device)

    for folder, fov_dict in time_range_dict.items():    
        for fov_id in fov_dict.keys():
            trench_time_ranges = fov_dict[fov_id]
            print(f"  FOV: {fov_id}, Trenches: {list(trench_time_ranges.keys())}")
    
            for peak_id, time_info in trench_time_ranges.items():
                base_file_path = os.path.join(base_path, folder, 'hyperstacked', 'drift_corrected', 'rotated',
                                                'mm_channels', 'subtracted')
                path_to_phase_stack = os.path.join(base_file_path,
                                                    f'subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_c_str}.tif')
                path_to_labeled_stack = os.path.join(base_file_path,
                                                        f'napari_corrections/{fov_id}_{peak_id}_corrected.tif')
                path_to_fluor_stack = os.path.join(base_file_path,
                                                    f'subtracted_FOV_{fov_id}_region_{peak_id}_c_{fluor_c_str}.tif')
                # --- Dynamic Time Range Assignment ---
                start = time_info['start']
                end = time_info.get('end') 
            
                
                if end is None:
                    end = get_tiff_frame_count(path_to_phase_stack)
                    if end == 0:
                        print(f"    WARNING: Could not determine frame count for {peak_id}. Skipping.")
                        continue
    
                # 2. Read Image Stacks
                try:
                    stack_phase = tifffile.imread(path_to_phase_stack)
                    stack_labeled = tifffile.imread(path_to_labeled_stack)
                    stack_fluor = tifffile.imread(path_to_fluor_stack)
                except FileNotFoundError as e:
                    print(f"    WARNING: Required file not found for {peak_id}: {e}. Skipping.")
                    continue
                print(f"{peak_id} start: {start} end: {end}")
                stack_phase_trimmed = stack_phase[start:end, :, :].copy()
                stack_labeled_trimmed = stack_labeled[start:end, :, :].copy()   
                track_graph, masks_tracked = model.track(stack_phase_trimmed, stack_labeled_trimmed, mode="ilp")
                ctc_tracks, ctc_masks = graph_to_ctc(
                    track_graph, masks_tracked, outdir=f'test_040726')
                
                napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)

                #save outputs to be plotted elsewhere
                ctc_tracks.to_csv(f'/oak/stanford/groups/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output/{folder}_ctc_tracks_{fov_id}_{peak_id}.csv')
                np.save(f'/oak/stanford/groups/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output/{folder}_ctc_masks_{fov_id}_{peak_id}.npy', ctc_masks)
                np.save(f'/oak/stanford/groups/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output/{folder}_imgs_{fov_id}_{peak_id}.npy', stack_phase_trimmed)
                np.save(f'/oak/stanford/groups/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output/{folder}_napari_tracks_{fov_id}_{peak_id}.npy', napari_tracks)
                with open(f'/oak/stanford/groups/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output/{folder}_napari_tracks_graph_{fov_id}_{peak_id}.json', "w") as f:
                    json.dump(napari_tracks_graph, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="trackastra tracking algorithm"
    )

    parser.add_argument(
        '--base-path',
        required=True,
        type=str,
        help="path to directory containing all microscopy experiments"
    )

    parser.add_argument(
        '--time-range-dict',
        required=False,
        type=str,
        default='',
        help='JSON string defining start/end time frames to trim image before applying trackastra. Format: '
             '{"Exp_name":{"FOV":{"Peak_ID":{"start": 62, "end": null}}}}'
    )

    args = parser.parse_args()

    run_track_astra(args.base_path, args.time_range_dict)