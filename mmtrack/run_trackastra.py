import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks, write_to_geff
from trackastra.data import example_data_bacteria

def run_track_astra():
    path_all_lineages_df = '/Users/adrianjuarez/Documents/Covert_lab/Projects/Operon/tracked_all_cell_data_aggregate_032626.pkl'
    path_all_cell_data_df = '/Users/adrianjuarez/Documents/Covert_lab/Projects/Operon/all_cell_data_aggregate_032626.pkl'
    all_lineages_df = pd.read_pickle(path_all_lineages_df)
    all_cell_data_df = pd.read_pickle(path_all_cell_data_df)

    experiment = ''
    base_path =f'/Volumes/mcovert/Instruments/Covert-lab-scope1/subgen_processed_data/{experiment}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted'
    path_to_phase_stack_dir=f'{base_path}'
    path_to_labels_stack_dir =f'{base_path}/mask_kymos'
    # phase_list = os.listdir(path_to_phase_stack_dir)
    # mask_list =os.listdir(path_to_labels_stack_dir)


if __name__ == "__main__":
    run_track_astra()