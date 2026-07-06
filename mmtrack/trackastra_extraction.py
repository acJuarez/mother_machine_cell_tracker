import torch
import os
import sys
sys.path.insert(0, "/Users/adrianjuarez/Documents/Covert_lab/Repos/mother_machine_cell_tracker")
import mmtrack.pre_process_mm as pre
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks, write_to_geff
from trackastra.data import example_data_bacteria
import tifffile
import napari
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/adrianjuarez/Documents/Covert_lab/Repos/mother_machine_cell_tracker")
from mmtrack import plot_cells


def plot_trackastra_kymograph(imgs, ctc_masks, napari_tracks, napari_tracks_graph):
    t, y, x = imgs.shape
    
    # Transpose to (y, t, x) then reshape to (y, t*x)
    kymo_imgs = imgs.transpose(1, 0, 2).reshape(y, -1)  # (400, 540)
    kymo_masks = ctc_masks.transpose(1, 0, 2).reshape(y, -1)

    kymo_tracks = napari_tracks.copy()
    # New x position: time_frame * width + original_x
    new_x = napari_tracks[:, 1] * x + napari_tracks[:, 3]
    kymo_tracks = np.column_stack([
        napari_tracks[:, 0],  # track_id
        np.zeros(len(napari_tracks)),  # dummy time (all in same frame)
        napari_tracks[:, 2],  # y stays the same
        new_x  # new x position
    ])

    v = napari.Viewer()
    v.add_image(kymo_imgs, name='kymograph')
    v.add_labels(kymo_masks, name='masks_kymo')
    v.add_tracks(data=kymo_tracks, graph=napari_tracks_graph, name='tracks_kymo')

def main():
    # load all data, note that path files may need to be updated
    imgs = np.load('/Volumes/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output/DUMM_giTG059_068_SC_093025_imgs_020_589.npy')
    napari_tracks = np.load("/Volumes/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output/DUMM_giTG059_068_SC_093025_napari_tracks_020_589.npy")
    ctc_masks = np.load("/Volumes/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output/DUMM_giTG059_068_SC_093025_ctc_masks_020_589.npy")


    with open("/Volumes/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output/DUMM_giTG059_068_SC_093025_napari_tracks_graph_020_589.json") as f:
        napari_tracks_graph = json.load(f)
    # (optional) convert keys/values back to int if needed:
    napari_tracks_graph = {int(k): int(v) for k, v in napari_tracks_graph.items()}

    # Get all unique cell tracks
    unique_tracks = np.unique(napari_tracks[:, 0])
    print(f"Total number of cells tracked: {len(unique_tracks)}")

    # For each cell track
    for track_id in unique_tracks:
        # Get all timepoints for this cell
        cell_data = napari_tracks[napari_tracks[:, 0] == track_id]
        
        # Extract information
        timepoints = cell_data[:, 1]  # time
        y_coords = cell_data[:, 2]     # y position
        x_coords = cell_data[:, 3]     # x position
        
        print(f"Track {track_id}: {len(timepoints)} timepoints, "
            f"t={timepoints.min():.0f}-{timepoints.max():.0f}")
    def get_lineage_info(napari_tracks_graph):
        """Extract parent-daughter relationships"""
        lineages = []
        
        for daughter_id, parent_id in napari_tracks_graph.items():
            lineages.append({
                'parent': parent_id,
                'daughter': daughter_id
            })
        
        return lineages

    # Find all division events
    lineages = get_lineage_info(napari_tracks_graph)
    print(f"Number of division events: {len(lineages)}")

    # Find all daughters of a specific parent
    def get_daughters(parent_id, napari_tracks_graph):
        return [d for d, p in napari_tracks_graph.items() if p == parent_id]

    # Example: find sisters (cells with same parent)
    from collections import defaultdict
    parent_to_daughters = defaultdict(list)
    for daughter, parent in napari_tracks_graph.items():
        parent_to_daughters[parent].append(daughter)

    for parent, daughters in parent_to_daughters.items():
        if len(daughters) > 1:
            print(f"Parent {parent} -> Daughters {daughters}")