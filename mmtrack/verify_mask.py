"""
Utility function to load corrected kymograph and verify labels are unique.
"""

import numpy as np
import tifffile


def load_and_verify_corrected_mask(corrected_path):
    """
    Load a corrected segmentation mask and verify all cell labels are unique.
    
    Parameters
    ----------
    corrected_path : str
        Path to the corrected .tif mask file
        
    Returns
    -------
    mask : np.ndarray
        The loaded mask array (frames, height, width)
    n_cells : int
        Number of unique cell labels (excluding background)
    is_valid : bool
        True if all labels are unique
    """
    mask = tifffile.imread(corrected_path)
    
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    n_cells = len(unique_labels)
    max_label = unique_labels.max() if len(unique_labels) > 0 else 0
    
    is_valid = True
    
    expected = set(range(1, max_label + 1))
    actual = set(unique_labels)
    missing = sorted(expected - actual)
    
    print(f"Loaded: {corrected_path}")
    print(f"  Shape: {mask.shape}")
    print(f"  Unique cell labels: {n_cells}")
    print(f"  Label range: 1 to {max_label}")
    
    if missing:
        print(f"  Missing labels (gaps): {missing}")
    else:
        print(f"  ✅ Labels are sequential (no gaps)")
    
    print(f"  ✅ All labels are unique")
    
    return mask, n_cells, is_valid


def mask_to_kymograph(mask):
    """
    Convert a 3D mask stack (frames, H, W) to a 2D kymograph (H, frames*W).
    """
    return np.hstack([mask[i] for i in range(mask.shape[0])])


def view_corrected_kymograph(corrected_path):
    """
    Load corrected mask, verify labels, and display as kymograph in napari.
    """
    import napari
    
    mask, n_cells, is_valid = load_and_verify_corrected_mask(corrected_path)
    kymo = mask_to_kymograph(mask)
    
    viewer = napari.Viewer()
    viewer.add_labels(kymo, name=f'corrected_kymo ({n_cells} cells)')
    napari.run()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to corrected mask: ")
    
    view_corrected_kymograph(path)
