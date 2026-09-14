"""
Feature extraction doing bulk of the work for 06_trackastra_feature_extraction.py

Extracts per-cell morphological/intensity features from CTC-format tracked masks,
assigns hierarchical lineage IDs, generates kymograph TIFFs, and saves a
per-experiment DataFrame compatible with the existing plot_cells.py functions. Uses regionprops
like previous iteration of pipeline used to extract cell features 

Expected trackastra output files (produced by run_trackastra_sherlock.py):
    {exp}_ctc_masks_{fov}_{peak}.npy          -- (T, Y, X) tracked label masks
    {exp}_imgs_{fov}_{peak}.npy               -- (T, Y, X) trimmed phase stack
    {exp}_ctc_tracks_{fov}_{peak}.csv         -- CTC track table: label, t1, t2, parent
                                                 (parent == 0 means root). THE lineage source.

TWO LABEL SPACES -- do not mix them. `track_id` here is the ctc_masks label. The sibling
file {exp}_napari_tracks_{fov}_{peak}.npy uses a *different* numbering, and
{exp}_napari_tracks_graph_{fov}_{peak}.json is keyed in that napari space. Mapping the
napari graph onto track_id therefore attributes each division to the wrong cells; it
disagreed with the CTC parents in 99 of 293 trenches (~34%) on the 082026 generation, and
produced impossible parent->daughter frame gaps (a "parent" alive in the same frame as its
"daughter"). ctc_tracks_*.csv is already in ctc_masks label space, so it is used instead.
Translating the napari graph through a centroid-matched label map reproduces the CTC
parents exactly, which is what confirmed the two files disagree only by relabeling.

Output DataFrame columns (one row per cell × time frame):
    node_id, track_id, time_frame
    centroid_y, centroid_x         (kymograph-space coordinates)
    area, axis_major_length, axis_minor_length
    intensity_mean_phase, intensity_max_phase, intensity_min_phase
    intensity_mean_fluor, intensity_max_fluor, intensity_min_fluor
    coords                         (pixel coords in kymograph space, shape (N,2))
    predicted_lineage              (hierarchical ID string matching Stage 5 convention)
    parent_track_id                (direct parent, NaN for root tracks)
    experiment_name, FOV, trench_id
"""

import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

from mmtrack import plot_cells

def load_ctc_lineage_graph(ctc_tracks_csv):
    """
    Build {daughter_label: parent_label} from a CTC tracks CSV.

    The CSV has one row per track: label, t1 (first frame), t2 (last frame), parent
    (0 == root). Labels are ctc_masks values, i.e. the same space as `track_id`, which is
    why this and not napari_tracks_graph_*.json is the lineage source (see module
    docstring). Root rows are dropped so the returned dict has the same shape as the old
    napari graph and can be handed straight to assign_lineage_ids.
    """
    tracks = pd.read_csv(ctc_tracks_csv)
    missing = {'label', 'parent'} - set(tracks.columns)
    if missing:
        raise ValueError(f"{ctc_tracks_csv} is missing column(s) {sorted(missing)}")
    return {
        int(row.label): int(row.parent)
        for row in tracks.itertuples()
        if int(row.parent) != 0
    }


def assign_lineage_ids(lineage_graph, all_track_ids):
    """
    Build hierarchical lineage ID strings matching Stage 5 naming convention.

    - Root tracks (no parent): "1", "2", ...
    - Single child of a track (continuation): inherits parent's ID
    - Two+ children of a track (division): daughters get "parent_id.1", "parent_id.2", ...

    Args:
        lineage_graph: dict {daughter_track_id (int): parent_track_id (int)}, in the SAME
                       label space as all_track_ids (i.e. ctc_masks labels, as returned by
                       load_ctc_lineage_graph -- not the napari graph JSON).
        all_track_ids: iterable of all track IDs present in the data

    Returns:
        dict {track_id (int): lineage_id_string}
    """
    parent_to_children = defaultdict(list)
    for daughter, parent in lineage_graph.items():
        parent_to_children[parent].append(daughter)
    for parent in parent_to_children:
        parent_to_children[parent].sort()

    daughters = set(lineage_graph.keys())
    roots = sorted(t for t in all_track_ids if t not in daughters)

    track_to_lineage = {}

    def _dfs(track_id, lineage_id):
        track_to_lineage[track_id] = lineage_id
        children = parent_to_children.get(track_id, [])
        for i, child in enumerate(children, 1):
            # continuation (one child) keeps the same ID; division branches
            child_lineage = f"{lineage_id}.{i}" if len(children) > 1 else lineage_id
            _dfs(child, child_lineage)

    for idx, root in enumerate(roots, 1):
        _dfs(root, str(idx))

    # Fallback: assign IDs to any orphan tracks unreachable from roots
    fallback = len(roots) + 1
    for t in all_track_ids:
        if t not in track_to_lineage:
            track_to_lineage[t] = str(fallback)
            fallback += 1

    return track_to_lineage


def _extract_frame_features(mask_frame, phase_frame, fluor_frame, time_index, frame_width):
    """
    Run regionprops on one time frame and return a DataFrame with
    kymograph-space coordinates. Returns empty DataFrame if no cells.

    """
    if mask_frame.max() == 0:
        return pd.DataFrame()

    props_phase = regionprops_table(
        mask_frame.astype(np.int32), phase_frame,
        properties=[
            'label', 'area', 'coords', 'centroid',
            'axis_major_length', 'axis_minor_length',
            'intensity_mean', 'intensity_max', 'intensity_min',
        ]
    )
    props_fluor = regionprops_table(
        mask_frame.astype(np.int32), fluor_frame,
        properties=['label', 'intensity_mean', 'intensity_max', 'intensity_min']
    )

    df = pd.DataFrame(props_phase).merge(
        pd.DataFrame(props_fluor), on='label', suffixes=('_phase', '_fluor')
    )

    # Map centroid and pixel coords from image space to kymograph space.
    # Kymograph x-position for a cell at time t: t * frame_width + image_x
    x_offset = time_index * frame_width
    df['time_frame'] = time_index
    df['track_id'] = df['label']
    df['centroid_y'] = df['centroid-0']
    df['centroid_x'] = df['centroid-1'] + x_offset

    # Transform each cell's pixel coords to kymograph space (in-place via apply)
    df['coords'] = df['coords'].apply(
        lambda c: np.column_stack([c[:, 0], c[:, 1] + x_offset])
    )

    df.drop(columns=['centroid-0', 'centroid-1', 'label'], inplace=True)
    return df


# runs pipeline on one kymograph
def process_trench(
    ctc_masks,
    phase_stack,
    fluor_stack,
    lineage_graph,
    folder,
    fov_id,
    peak_id,
    kymo_output_dir,
):
    """
    Full per kymograph pipeline: extract features, assign lineages, save kymographs.

    Args:
        ctc_masks: (T, Y, X) ndarray — trackastra tracked label masks (label == track_id)
        phase_stack: (T, Y, X) ndarray — phase images, trimmed to match ctc_masks
        fluor_stack: (T, Y, X) ndarray — fluorescence images, trimmed to match ctc_masks
        lineage_graph: dict {daughter_track_id: parent_track_id} in ctc_masks label
                       space (see load_ctc_lineage_graph)
        folder, fov_id, peak_id: experiment metadata identifiers
        kymo_output_dir: directory where kymograph TIFFs are written

    Returns:
        DataFrame with one row per (track_id, time_frame), or empty DataFrame.
    """
    T, Y, X = ctc_masks.shape

    # --- Kymograph TIFFs (compatible naming with Stage 4/5) ---
    os.makedirs(kymo_output_dir, exist_ok=True)
    fluor_kymo_dir = os.path.join(kymo_output_dir, 'fluor_kymos')
    mask_kymo_dir = os.path.join(kymo_output_dir, 'mask_kymos')
    os.makedirs(fluor_kymo_dir, exist_ok=True)
    os.makedirs(mask_kymo_dir, exist_ok=True)

    plot_cells.create_kymograph(phase_stack, 0, T, fov_id, peak_id, kymo_output_dir)
    plot_cells.create_kymograph(fluor_stack, 0, T, fov_id, peak_id, fluor_kymo_dir)
    plot_cells.create_kymograph(ctc_masks.astype(np.uint16), 0, T, fov_id, peak_id, mask_kymo_dir)

    # --- Per-frame feature extraction ---
    frame_dfs = [
        _extract_frame_features(ctc_masks[t], phase_stack[t], fluor_stack[t], t, X)
        for t in range(T)
    ]
    frame_dfs = [d for d in frame_dfs if not d.empty]

    if not frame_dfs:
        print(f"      WARNING: no cells found in any frame for {folder}/{fov_id}/{peak_id}")
        return pd.DataFrame()

    df = pd.concat(frame_dfs, ignore_index=True)

    # --- Lineage assignment ---
    all_track_ids = df['track_id'].unique().tolist()
    track_to_lineage = assign_lineage_ids(lineage_graph, all_track_ids)
    df['predicted_lineage'] = df['track_id'].map(track_to_lineage).fillna('untracked')
    # lineage_graph is keyed by ctc_masks label, the same space as track_id, so this map is
    # label-for-label. NaN means root.
    df['parent_track_id'] = df['track_id'].map(lineage_graph)

    # --- Metadata ---
    df['experiment_name'] = folder
    df['FOV'] = fov_id
    df['trench_id'] = peak_id

    return df


# load trackastra outputs, only runs if not time_dict is provided
def discover_trackastra_outputs(trackastra_dir):
    """
    search trackastra_dir for ctc_masks files and build a time_range_dict
    suitable for run_trackastra_feature_extraction.

    Provide an explicit
    time_range_dict if a non-zero start offset was used.

    Filename format for ctc_masks: {exp}_ctc_masks_{fov}_{peak}.npy
    """
    pattern = os.path.join(trackastra_dir, '*_ctc_masks_*_*.npy')
    files = sorted(glob.glob(pattern))

    time_range_dict = {}
    for f in files:
        name = os.path.basename(f)[:-4]  # strip .npy
        if '_ctc_masks_' not in name:
            continue
        folder, rest = name.split('_ctc_masks_', 1)
        # rest = "{fov}_{peak}", peak may contain underscores so split from right
        parts = rest.rsplit('_', 1)
        if len(parts) != 2:
            print(f"  WARNING: cannot parse fov/peak from {name}. Skipping.")
            continue
        fov_id, peak_id = parts

        time_range_dict.setdefault(folder, {}).setdefault(fov_id, {})[peak_id] = {
            'start': 0, 'end': None
        }

    print(f"Auto-discovered {sum(len(p) for f in time_range_dict.values() for p in f.values())} "
          f"trenches across {len(time_range_dict)} experiments.")
    return time_range_dict


def run_trackastra_feature_extraction(
    base_path,
    trackastra_dir,
    time_range_dict=None,
    phase_c_str='0',
    fluor_c_str='1',
):
    """
    Iterate over all experiments/FOVs/trenches, extract features from trackastra
    outputs, and save one DataFrame per experiment.

    Phase images are loaded preferentially from the saved imgs.npy (guaranteed
    to be the exact trimmed stack trackastra used). Fluorescence is always loaded
    from the original TIFF using the start/end range in time_range_dict so make sure those are the same.

    Args:
        base_path: root directory containing experiment folders
        trackastra_dir: directory holding trackastra output .npy / .csv files
        time_range_dict: nested dict {exp: {fov: {peak: {'start': int, 'end': int|None}}}}
                         If None or empty, outputs are auto-discovered with start=0.
        phase_c_str: phase channel index string (for original TIFF filename)
        fluor_c_str: fluorescence channel index string

    Saves:
        {base_path}/trackastra_cell_data_{exp}.pkl  per experiment
    """
    if not time_range_dict:
        print("No time_range_dict provided — auto-discovering trackastra outputs.")
        time_range_dict = discover_trackastra_outputs(trackastra_dir)

    if not time_range_dict:
        print("ERROR: no trackastra outputs found. Exiting.")
        return

    for folder, fov_dict in time_range_dict.items():
        print(f"\n=== Experiment: {folder} ===")
        exp_dfs = []

        for fov_id, trench_dict in fov_dict.items():
            print(f"FOV: {fov_id}")

            for peak_id, time_info in trench_dict.items():

                # --- Load trackastra outputs ---
                ctc_path = os.path.join(
                    trackastra_dir, f'{folder}_ctc_masks_{fov_id}_{peak_id}.npy'
                )
                # Lineage comes from the CTC track table, which is in the same label
                # space as ctc_masks. There is no fallback to the napari graph JSON: it is
                # keyed in the napari track-id space and is wrong for ~1/3 of trenches
                # (see module docstring), so a silent fallback would reintroduce the bug.
                tracks_path = os.path.join(
                    trackastra_dir, f'{folder}_ctc_tracks_{fov_id}_{peak_id}.csv'
                )
                try:
                    ctc_masks = np.load(ctc_path)
                    lineage_graph = load_ctc_lineage_graph(tracks_path)
                except FileNotFoundError as exc:
                    print(f"WARNING: {exc}. Skipping trench {peak_id}.")
                    continue

                # --- Load phase (prefer saved imgs.npy; fallback to original TIFF) ---
                imgs_path = os.path.join(
                    trackastra_dir, f'{folder}_imgs_{fov_id}_{peak_id}.npy'
                )
                subtracted_dir = os.path.join(
                    base_path, folder, 'hyperstacked', 'drift_corrected',
                    'rotated', 'mm_channels', 'subtracted'
                )
                path_phase = os.path.join(
                    subtracted_dir,
                    f'subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_c_str}.tif'
                )
                path_fluor = os.path.join(
                    subtracted_dir,
                    f'subtracted_FOV_{fov_id}_region_{peak_id}_c_{fluor_c_str}.tif'
                )

                if os.path.exists(imgs_path):
                    phase_trimmed = np.load(imgs_path)
                else:
                    try:
                        start = time_info['start']
                        end = time_info.get('end') or start + ctc_masks.shape[0]
                        phase_trimmed = tifffile.imread(path_phase)[start:end]
                    except FileNotFoundError as exc:
                        print(f"    WARNING: {exc}. Skipping trench {peak_id}.")
                        continue

                # --- Load fluorescence from original TIFF ---
                try:
                    start = time_info['start']
                    end = time_info.get('end') or start + ctc_masks.shape[0]
                    fluor_trimmed = tifffile.imread(path_fluor)[start:end]
                except FileNotFoundError as exc:
                    print(f"    WARNING: {exc}. Skipping trench {peak_id}.")
                    continue

                # --- Reconcile frame counts defensively ---
                n = min(ctc_masks.shape[0], phase_trimmed.shape[0], fluor_trimmed.shape[0])
                ctc_masks = ctc_masks[:n]
                phase_trimmed = phase_trimmed[:n]
                fluor_trimmed = fluor_trimmed[:n]

                print(f"Trench {peak_id}: {n} frames, mask shape {ctc_masks.shape}")

                # --- Run per-trench pipeline ---
                df_trench = process_trench(
                    ctc_masks, phase_trimmed, fluor_trimmed,
                    lineage_graph,
                    folder, fov_id, peak_id,
                    kymo_output_dir=subtracted_dir,
                )

                if not df_trench.empty:
                    exp_dfs.append(df_trench)
                    print(f"{len(df_trench)} cell×frame observations, "
                          f"{df_trench['track_id'].nunique()} unique tracks, "
                          f"{df_trench['predicted_lineage'].nunique()} lineages.")

        # --- Save per-experiment DataFrame ---
        if exp_dfs:
            df_exp = pd.concat(exp_dfs, ignore_index=True)
            df_exp['node_id'] = df_exp.index
            out_path = os.path.join(base_path, f'trackastra_cell_data_{folder}.pkl')
            df_exp.to_pickle(out_path)
            print(f"\n  Saved {len(df_exp)} rows to {out_path}")
        else:
            print(f"\n  WARNING: no data extracted for {folder}.")

    print("\nTrackastra feature extraction complete.")
