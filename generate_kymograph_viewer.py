#!/usr/bin/env python
"""
Generate a self-contained HTML kymograph viewer with napari-like layer toggling.

Iterates over all trackastra-processed kymographs, renders phase/fluorescence/mask/track
layers, and embeds them into a single HTML file with interactive layer controls.

Everything is displayed in the ctc_masks label space, so the track numbers printed here
are the same `track_id` as in all_data_*.csv and the stage-7 pair tables -- a QC hit is
findable by number. That needs an explicit translation: napari_tracks_*.npy (which
supplies the polyline geometry) uses its own numbering, different from ctc_masks in ~1/3
of trenches, and napari_tracks_graph_*.json is keyed in that napari space. The mask layer
and the fluorescence charts have always been in ctc space, so before this translation the
printed labels and the chart series disagreed for those trenches.

Usage:
    python generate_kymograph_viewer.py [--time-dict data/time_dict_082026.json] \
        [--output data/kymograph_viewer.html]
"""

import argparse
import base64
import io
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Strain-experiment mapping
# ---------------------------------------------------------------------------

# Gene/timing metadata lives in one place so the viewer and the analysis notebook
# cannot drift apart. See mmtrack/experiment_metadata.py.
from mmtrack.experiment_metadata import STRAIN_EXP_DICT  # noqa: E402


# ---------------------------------------------------------------------------
# Fluorescence denoising
# ---------------------------------------------------------------------------

# Filter implementations are shared with the track-displacement analysis so both
# apply identical denoising. See mmtrack/fluor_filters.py.
from mmtrack.fluor_filters import (  # noqa: E402
    denoise_fluor,
    from_kymograph,
    imagej_median,
    remove_speckle,
    to_kymograph,
)

# Lineage graph loader is shared with stage 6 so the viewer and the cell table can never
# disagree about which cell is whose parent.
from mmtrack.trackastra_feature_extraction import load_ctc_lineage_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_kymograph_list(combined_dict, time_dict, trackastra_dir):
    """Return sorted list of (gene, exp, fov, peak) tuples for which data exists."""
    kymos = []
    for gene in sorted(combined_dict):
        for exp, fovs in sorted(combined_dict[gene].items()):
            for fov in sorted(fovs):
                if exp not in time_dict or fov not in time_dict.get(exp, {}):
                    continue
                for peak in sorted(time_dict[exp][fov]):
                    imgs_path = os.path.join(
                        trackastra_dir, f"{exp}_imgs_{fov}_{peak}.npy"
                    )
                    if os.path.exists(imgs_path):
                        kymos.append((gene, exp, fov, peak))
    return kymos


def relabel_tracks_to_ctc(tracks, ctc_masks):
    """Rewrite napari_tracks' track ids into ctc_masks label space.

    `tracks` is (N, 4) of [napari_track_id, t, y, x], where (y, x) is the cell's centroid
    in image space -- the same centroid regionprops computes for the ctc label, so for the
    true pairing the distance is ~0.

    Matching is done per frame as a one-to-one assignment (Hungarian, on squared centroid
    distance) rather than independent nearest-neighbour lookups. In a 20-px-wide trench of
    stacked cells, greedy matching lets two adjacent tracks claim the same label and the
    frame silently loses one; the assignment constraint is what makes the result a genuine
    relabelling. Each napari id then takes the ctc label it matched most often.

    Returns a copy with column 0 rewritten. Napari ids with no match keep their original
    value rather than being dropped, so geometry is never silently lost.
    """
    votes = defaultdict(Counter)
    for t in range(ctc_masks.shape[0]):
        rows = np.flatnonzero(tracks[:, 1].astype(int) == t)
        if rows.size == 0:
            continue
        mask_t = ctc_masks[t]
        labels = np.unique(mask_t)
        labels = labels[labels != 0]
        if labels.size == 0:
            continue
        centroids = np.array(
            ndimage.center_of_mass(np.ones_like(mask_t), labels=mask_t, index=labels)
        ).reshape(-1, 2)

        points = tracks[np.ix_(rows, [2, 3])]
        cost = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        pi, ci = linear_sum_assignment(cost)
        for r, c in zip(pi, ci):
            votes[int(tracks[rows[r], 0])][int(labels[c])] += 1

    mapping = {tid: c.most_common(1)[0][0] for tid, c in votes.items() if c}
    unmapped = sorted(set(tracks[:, 0].astype(int)) - set(mapping))
    if unmapped:
        print(f"    WARNING: {len(unmapped)} napari track id(s) had no ctc label match "
              f"{unmapped[:5]}; left unrelabelled.")
    collisions = len(mapping) - len(set(mapping.values()))
    if collisions:
        print(f"    WARNING: {collisions} napari id(s) map onto an already-used ctc label; "
              f"the two labelings are not a clean permutation for this trench.")

    out = tracks.copy()
    out[:, 0] = [mapping.get(int(tid), int(tid)) for tid in tracks[:, 0]]
    return out


def load_kymograph_data(exp, fov, peak, time_dict, base_dir, trackastra_dir,
                        fluor_filter="none", filter_size=3, filter_sigma=1.0,
                        nlm_h=1.15, otsu_scale=1.0, filter_radius=2.0):
    """Load all arrays for a single kymograph. Returns dict or None on failure."""
    imgs_path = os.path.join(trackastra_dir, f"{exp}_imgs_{fov}_{peak}.npy")
    ctc_path = os.path.join(trackastra_dir, f"{exp}_ctc_masks_{fov}_{peak}.npy")
    tracks_path = os.path.join(trackastra_dir, f"{exp}_napari_tracks_{fov}_{peak}.npy")
    # CTC track table, not napari_tracks_graph_*.json: it is already in the ctc_masks
    # label space (see module docstring).
    ctc_tracks_path = os.path.join(trackastra_dir, f"{exp}_ctc_tracks_{fov}_{peak}.csv")

    try:
        imgs = np.load(imgs_path)
        ctc_masks = np.load(ctc_path)
    except Exception as e:
        print(f"  SKIP {exp}/{fov}/{peak}: {e}")
        return None

    tracks = None
    graph = {}
    if os.path.exists(tracks_path):
        try:
            tracks = np.load(tracks_path)
        except Exception:
            pass
    if os.path.exists(ctc_tracks_path):
        try:
            graph = load_ctc_lineage_graph(ctc_tracks_path)
        except Exception:
            pass

    # Relabel the track geometry into ctc space so it agrees with `graph`, the mask layer
    # and the fluorescence charts.
    if tracks is not None and len(tracks):
        tracks = relabel_tracks_to_ctc(tracks, ctc_masks)

    fluor = None
    subtracted_dir = os.path.join(
        base_dir, exp, "hyperstacked", "drift_corrected",
        "rotated", "mm_channels", "subtracted",
    )
    fluor_path = os.path.join(
        subtracted_dir, f"subtracted_FOV_{fov}_region_{peak}_c_1.tif"
    )
    if os.path.exists(fluor_path):
        try:
            raw_fluor = tifffile.imread(fluor_path)
            start = time_dict[exp][fov][peak].get("start", 0)
            n_frames = imgs.shape[0]
            fluor = raw_fluor[start : start + n_frames]
            if fluor.shape[0] != n_frames:
                fluor = fluor[: min(fluor.shape[0], n_frames)]
        except Exception:
            fluor = None

    # Denoised copy for the "filtered" layer. Charts intentionally keep using the
    # raw `fluor` below; the two image layers are the visual assessment tool.
    #
    # "fiji_median" is filtered at RENDER time instead, on the assembled kymograph --
    # see render_fluor_fiji. Pre-filtering the stack here would be both wasted work and
    # the wrong image, because a per-frame kernel cannot reach across a frame seam.
    if fluor_filter == "fiji_median":
        fluor_filtered = None
    else:
        # Keyword args on purpose: this call used to be positional, which made every
        # new parameter on denoise_fluor a silent-misbinding hazard.
        fluor_filtered = denoise_fluor(
            fluor,
            method=fluor_filter,
            size=filter_size,
            sigma=filter_sigma,
            nlm_h=nlm_h,
            otsu_scale=otsu_scale,
            radius=filter_radius,
        )

    start_frame = time_dict[exp][fov][peak].get("start", 0)
    end_frame = start_frame + imgs.shape[0] - 1

    return {
        "imgs": imgs,
        "ctc_masks": ctc_masks,
        "tracks": tracks,
        "graph": graph,
        "fluor": fluor,
        "fluor_filtered": fluor_filtered,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _normalize_to_uint8(arr, vmin=None, vmax=None):
    """Linear contrast stretch to uint8.

    With no explicit range this stretches `arr`'s own 1st-99th percentile, which is what
    the phase layer and the raw fluorescence layer use. Pass `vmin`/`vmax` to impose a
    range computed elsewhere; `fluor_display_range` builds the one the filtered layer
    uses.
    """
    arr = arr.astype(np.float32)
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(arr, (1, 99))
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - vmin) / (vmax - vmin) * 255, 0, 255)
    return arr.astype(np.uint8)


# The kymograph layout now lives in mmtrack.fluor_filters, so the per-cell extractor in
# mmtrack.track_displacement assembles frames exactly the way this renderer does. A second
# copy here is precisely how the two would silently diverge.
_to_kymograph = to_kymograph


def _img_to_base64(pil_img, fmt="JPEG", quality=80):
    buf = io.BytesIO()
    if fmt == "JPEG":
        pil_img = pil_img.convert("RGB")
        pil_img.save(buf, format=fmt, quality=quality)
    else:
        pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_phase(imgs, downscale, quality, start_frame):
    from PIL import ImageDraw, ImageFont

    kymo = _to_kymograph(imgs)
    kymo8 = _normalize_to_uint8(kymo)
    pil = Image.fromarray(kymo8, mode="L")
    if downscale != 1.0:
        new_size = (max(1, int(pil.width * downscale)), max(1, int(pil.height * downscale)))
        pil = pil.resize(new_size, Image.LANCZOS)

    img_h = pil.height
    num_frames = imgs.shape[0]
    frame_width = imgs.shape[2]
    scaled_fw = frame_width * downscale
    label_bar_h = 16
    canvas = Image.new("L", (pil.width, img_h + label_bar_h), 0)
    canvas.paste(pil, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial", 9)
    except OSError:
        font = ImageFont.load_default(size=9)
    for i in range(num_frames):
        x = int(i * scaled_fw + scaled_fw / 2)
        draw.text((x, img_h + 2), str(start_frame + i), fill=255, font=font, anchor="mt")

    return _img_to_base64(canvas, "JPEG", quality), kymo.shape, label_bar_h


def fluor_display_range(fluor, low_frac, high_pct):
    """(vmin, vmax) for the filtered layer -- the manual FIJI Brightness/Contrast step.

    vmax is the `high_pct` percentile of the trench's raw pixels; 100 is the true max,
    where FIJI's slider sits when you open the image. vmin is `low_frac` x vmax -- a
    fraction of the RANGE, which is what "set the minimum to 25% of the max" means.

    NOT a percentile of the pixel values. On this background-subtracted data the 25th
    percentile is 0 while 0.25 x max is 8.5, and the two differ on 78% of rendered pixels
    by up to 67 grey levels, taking the black fraction from 21.9% to 98.9%. The
    percentile reading keeps the entire 0-9 count band that the FIJI setting deliberately
    crushes to black.

    Rounded half-up to an integer, because that is what you type into FIJI's Set Display
    Range: 0.25 x 34 = 8.5 -> 9. Using 8.5 shifts 1.7% of pixels by <=5 grey levels.
    Python's round() is banker's rounding and would give 8, hence the explicit + 0.5.

    Read from the RAW stack, before filtering, because that is FIJI's order (set B&C,
    then filter); by the clip/median commutation it is the same image either way.
    Percentiles pool all of (T, Y, X), so the kymograph cannot flicker column to column.
    """
    if fluor is None:
        return None, None
    vmax = float(np.percentile(fluor, high_pct))
    vmin = float(int(low_frac * vmax + 0.5))
    return vmin, vmax


def render_fluor(fluor, downscale, quality, vmin=None, vmax=None):
    kymo = _to_kymograph(fluor)
    kymo8 = _normalize_to_uint8(kymo, vmin, vmax)
    pil = Image.fromarray(kymo8, mode="L")
    if downscale != 1.0:
        new_size = (max(1, int(pil.width * downscale)), max(1, int(pil.height * downscale)))
        pil = pil.resize(new_size, Image.LANCZOS)
    return _img_to_base64(pil, "JPEG", quality)


def render_fluor_fiji(fluor, radius, downscale, vmin=None, vmax=None, min_size=0,
                      thr=None):
    """The filtered layer, filtered the way FIJI filters it: on the ASSEMBLED kymograph.

    The workflow this reproduces is a median run in FIJI on
    `subtracted/fluor_kymos/<fov>_<peak>.tif`. That file is written by
    `create_kymograph` (mmtrack/plot_cells.py:332) as a plain `np.concatenate` along x
    with no padding, separators or normalization -- bit-identical in layout to
    `_to_kymograph`, verified against the trench stack. So assembling first and
    filtering second is what FIJI does, and the kernel therefore spans frame seams: at
    a boundary the median is computed partly from the adjacent TIME POINT.

    That is not a rounding difference. Frames are 20 px wide and the kernel spans +/-2,
    so columns 0, 1, 18 and 19 of every frame -- 20% of the trench width -- are exposed.
    Measured on 008/1488, filtering per frame instead changes 8.3% of all pixels
    (56,692 of 682,200) by up to 67 grey levels after the display map, and leaves every
    interior column bit-identical.

    Emitted as lossless PNG, not JPEG. At quality 80 the codec alone moves 78% of pixels
    on 008/1488 (max 22 grey levels), because that trench's p99 is 12 counts and max 34,
    so the [p25, p100] map amplifies one raw count into ~7.5 grey levels and the codec's
    mush reads as texture that is not in the data. `--jpeg-quality` does not apply here.

    `min_size` adds a small-object removal pass after the median -- BEYOND the FIJI
    workflow, so with it enabled the layer is not bit-identical to FIJI. Pass 0 to
    disable and get the exact FIJI reproduction, which is how the 1:1 regression test
    runs. `thr` is the object threshold and should be `vmin`: objects then mean *visible*
    ones, and an Otsu threshold would leave the step inert under a 0.25 x max black
    point. It runs on the
    assembled kymograph too, which is what lets real cells survive it: a cell chains
    across frame seams into a component of ~10,000 px instead of standing alone at ~300.
    """
    kymo = imagej_median(_to_kymograph(fluor), radius)
    kymo = remove_speckle(kymo, min_size, thr=thr)
    kymo8 = _normalize_to_uint8(kymo, vmin, vmax)
    pil = Image.fromarray(kymo8)
    if downscale != 1.0:
        new_size = (max(1, int(pil.width * downscale)), max(1, int(pil.height * downscale)))
        pil = pil.resize(new_size, Image.LANCZOS)
    return _img_to_base64(pil, "PNG")


def render_mask(ctc_masks, downscale):
    kymo = _to_kymograph(ctc_masks)
    import matplotlib.pyplot as plt
    tab20 = plt.colormaps["tab20"]
    unique_labels = np.unique(kymo)
    rgba = np.zeros((*kymo.shape, 4), dtype=np.uint8)
    for label in unique_labels:
        if label == 0:
            continue
        color = tab20(int(label) % 20)
        mask = kymo == label
        rgba[mask, 0] = int(color[0] * 255)
        rgba[mask, 1] = int(color[1] * 255)
        rgba[mask, 2] = int(color[2] * 255)
        rgba[mask, 3] = 180
    pil = Image.fromarray(rgba, mode="RGBA")
    if downscale != 1.0:
        new_size = (max(1, int(pil.width * downscale)), max(1, int(pil.height * downscale)))
        pil = pil.resize(new_size, Image.NEAREST)
    return _img_to_base64(pil, "PNG")


def render_fluor_normalized(ctc_masks, series, start_frame, downscale):
    """Each cell filled with its own mean fluorescence, rescaled min->max WITHIN this trench.

    The point of this layer. Trench-to-trench variation is large -- within one
    (gene, experiment) the median of trench-median fluorescence spans a median of 2.11x, and
    up to 10.99x for mazF in DUMM_mazF_murQ_121625 -- so a cell that looks unremarkable
    against its gene can be the brightest thing in its own trench, and vice versa. Detection
    thresholds are (gene, experiment)-scoped, so this layer shows the other view: what the
    data looks like if every trench is normalized to itself.

    Note this is CELL-level, not pixel-level. The phase and fluorescence image layers are
    already per-trench normalized (`_normalize_to_uint8` does a 1st-99th percentile stretch on
    each trench's own kymograph), so a pixel min/max layer would differ from them only by
    being more outlier-sensitive. What was missing was a view of the per-cell quantity
    detection actually consumes.

    Reuses `compute_track_fluorescence`, whose mean over `ctc_masks == label` is identical to
    the `intensity_mean_fluor` feature, so this layer cannot disagree with the fluorescence
    charts drawn beside it.

    Two things that look like discrepancies and are not:

      * `series` carries RAW fluorescence, not `fluor_per_ms`. Min/max rescaling is invariant
        under a constant factor and exposure is one value per experiment, so the normalized
        value is identical either way.
      * This is not bit-identical to the export's `fluor_per_ms_kymo_minmax`. Same definition,
        slightly different population: the export drops degenerate (sub-area-floor) masks
        before computing the min and max, this normalizes over every label present.

    Viridis, not the tab20 used by `render_mask`: the quantity is sequential, and reusing a
    categorical palette here would imply cells belong to classes. Returns "" when there is no
    fluorescence to show.
    """
    if series is None or ctc_masks is None:
        return ""
    vals = [v for pts in series.values() for _, v in pts]
    if not vals:
        return ""
    lo, hi = float(min(vals)), float(max(vals))
    span = hi - lo

    import matplotlib.pyplot as plt
    cmap = plt.colormaps["viridis"]

    # (column, label) -> normalized value. `series` is keyed on ABSOLUTE frames; the mask
    # stack is indexed by kymograph column, so the start offset has to come back off.
    norm = {}
    for lab, pts in series.items():
        for abs_frame, v in pts:
            col = int(abs_frame) - int(start_frame)
            if 0 <= col < ctc_masks.shape[0]:
                norm[(col, int(lab))] = 0.5 if span <= 0 else (float(v) - lo) / span

    kymo = _to_kymograph(ctc_masks)
    frame_w = ctc_masks.shape[2]
    rgba = np.zeros((*kymo.shape, 4), dtype=np.uint8)
    for (col, lab), value in norm.items():
        x0 = col * frame_w
        sub = kymo[:, x0:x0 + frame_w]
        sel = sub == lab
        if not sel.any():
            continue
        r, g, b, _ = cmap(value)
        block = rgba[:, x0:x0 + frame_w]
        block[sel, 0] = int(r * 255)
        block[sel, 1] = int(g * 255)
        block[sel, 2] = int(b * 255)
        block[sel, 3] = 235

    pil = Image.fromarray(rgba, mode="RGBA")
    if downscale != 1.0:
        new_size = (max(1, int(pil.width * downscale)), max(1, int(pil.height * downscale)))
        pil = pil.resize(new_size, Image.NEAREST)
    return _img_to_base64(pil, "PNG")


def build_track_color_map(track_ids):
    """Map each track_id to an "rgb(r,g,b)" string via tab20 (position-based).

    Shared by the tracks overlay, labels layer, and fluorescence charts so a given
    lineage has the same color everywhere. IDs are sorted ascending (matches
    np.unique) before assigning colors.
    """
    import matplotlib.pyplot as plt
    tab20 = plt.colormaps["tab20"]
    color_map = {}
    for i, tid in enumerate(np.unique(np.asarray(list(track_ids)))):
        c = tab20(int(i) % 20)
        color_map[tid] = f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
    return color_map


def render_tracks_svg(tracks, graph, img_width, kymo_shape, downscale, color_map):
    """Return an inline SVG string with track polylines and division connectors."""
    if tracks is None or len(tracks) == 0:
        return ""

    h, w = kymo_shape
    sx = downscale
    sy = downscale
    svg_w = int(w * sx)
    svg_h = int(h * sy)

    # Transform track coords to kymograph space
    track_ids = tracks[:, 0]
    new_x = tracks[:, 1] * img_width + tracks[:, 3]
    y_coords = tracks[:, 2]

    unique_ids = np.unique(track_ids)

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}" style="position:absolute;top:0;left:0;">'
    )

    track_points = {}
    for tid in unique_ids:
        mask = track_ids == tid
        xs = new_x[mask] * sx
        ys = y_coords[mask] * sy
        order = np.argsort(tracks[mask, 1])
        xs = xs[order]
        ys = ys[order]
        track_points[tid] = (xs, ys)
        if len(xs) < 2:
            continue
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        lines.append(
            f'<polyline points="{pts}" fill="none" stroke="{color_map[tid]}" '
            f'stroke-width="1.5" stroke-opacity="0.9"/>'
        )

    for daughter, parent in graph.items():
        if parent in track_points and daughter in track_points:
            px, py = track_points[parent]
            dx, dy = track_points[daughter]
            if len(px) > 0 and len(dx) > 0:
                lines.append(
                    f'<line x1="{px[-1]:.1f}" y1="{py[-1]:.1f}" '
                    f'x2="{dx[0]:.1f}" y2="{dy[0]:.1f}" '
                    f'stroke="{color_map.get(parent, "#999")}" '
                    f'stroke-width="1" stroke-dasharray="4,3" stroke-opacity="0.7"/>'
                )

    lines.append("</svg>")
    return "\n".join(lines)


def render_labels_svg(tracks, img_width, kymo_shape, downscale, color_map):
    """Return an inline SVG overlay with a track_id text label on EVERY cell-frame.

    One label per cell-frame, not one per lineage. Through 090926 this mirrored the napari
    'track_labels' points layer and drew a single label at each lineage's FIRST appearance,
    which is unusable for identifying a cell mid-movie: to learn the id of a cell at frame 80
    you had to trace its polyline back through division connectors to wherever that lineage
    started. Reading a `track_id` straight off the screen is what a human needs in order to
    hand-record a burst the automated definitions missed (see mmtrack/burst_labels.py), so
    every cell-frame now carries its own number.

    `tracks` is the napari array [track_id, t, y, x] with exactly one row per cell-frame --
    verified, 300 rows against 300 cell-frames on DUMM_giTG060_064_121425 013/153 -- so this
    is a loop change, not new data. Its ids are the ctc mask labels, i.e. the same `track_id`
    the cell tables use, because relabel_tracks_to_ctc has already run; a number read off
    this layer therefore joins directly to all_data_annotated_082026.csv. That equality is
    the entire point of the layer, so do not "optimise" it to use the raw napari ids.

    Font drops to 8 px from 10: frames are 20 px wide at --downscale 1.0, and a 3-digit id at
    10 px overruns its own frame into the neighbouring time point. `text-anchor=middle`
    centres each number on its cell for the same reason. Colour still matches the track and
    chart colour, and the black paint-order stroke keeps it readable over any layer beneath.
    """
    if tracks is None or len(tracks) == 0:
        return ""

    h, w = kymo_shape
    svg_w = int(w * downscale)
    svg_h = int(h * downscale)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}" style="position:absolute;top:0;left:0;">'
    ]
    for row in tracks:
        tid, t, cy, cx = row[0], row[1], row[2], row[3]
        x = (t * img_width + cx) * downscale
        y = cy * downscale
        color = color_map.get(tid, "#ffff00")
        lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="8" fill="{color}" '
            f'style="paint-order:stroke;stroke:#000;stroke-width:2px;stroke-linejoin:round;" '
            f'text-anchor="middle" dominant-baseline="middle">{int(tid)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Burst outlines (from the notebook's burst CSV)
# ---------------------------------------------------------------------------

# The definitions the viewer draws -- TWO layers, kept to the live decision boundary:
# v7, the most PRECISE rule available (P 0.838 against the hand labels), and v12, the trained
# classifier that beats every rule on recall (P 0.607 at R 0.739, out of fold, one-to-one
# matched at +-1 frame). A rule that is right when it fires beside a model that finds three
# times as much: yellow-without-magenta is something the rule sees and the model missed,
# magenta-without-yellow is the recall v12 buys.
#
# Two, not more: more than three outline layers cannot be read on one kymograph, and every
# other definition is either nested inside these or superseded. v5 is a STRICT SUBSET of v7,
# so every v5 outline was already inside a v7 one; v4/v6 are the versions v5/v7 are supersets
# of; v1-v3 are the frozen regression fixture; v9/v10 are superseded by v11, and v11 by v12.
# All of them stay in mmtrack.bursts -- this list is only what gets DRAWN. Per-definition
# detail: bursts.md sections 7 and 14.
#
# Every definition stays a COLUMN in the CSV, and all of them stay in mmtrack.bursts as the
# regression fixture against data/bursts_082026.csv -- this list is only what gets a layer.
#
# ADDING A VERSION TAKES SIX EDITS, NOT ONE. This tuple drives the SVG generation and the
# `layer-burst<v>` divs, but the checkbox panel, the JS `layers` array and the --burst-csv help
# text below are all hardcoded. The six places are:
#   1. this tuple
#   2. BURST_COLORS just below (a bare dict index in render_bursts_svg -- KeyError if missed)
#   3. the chk-burst<v> input+label in HTML_TEMPLATE
#   4. ... and that ctrl-group's OWN closing </div>. The insertion point sits immediately
#      above the .controls closer, so it is very easy to let the new group swallow it: that
#      happened when v9 was added, leaving .controls unclosed, which silently killed the
#      sticky control bar AND horizontal kymograph scrolling (every .gene-panel became a flex
#      item of .controls). _assert_controls_balanced below now fails the build instead.
#   5. the `layers` JS array in HTML_TEMPLATE
#   6. the --burst-csv help string in main()
#
# DO 3 BEFORE 5. The toggle JS calls getElementById with no null guard, so a `layers` entry
# with no matching checkbox throws TypeError and aborts the whole IIFE -- applyLayers() never
# runs, so NO layer initialises and the lineage-highlight handlers never register. That is
# worse than the reverse mistake, which only leaves outlines hidden: measured when v6 was
# added, 101 `layer-burstv6` divs and 0 `chk-burstv6` checkboxes.
#
# After building, grep the HTML: `chk-burstv<n>` should appear exactly twice (the id and the
# for), `burstv<n>` once inside `const layers`, and `layer-burstv<n>` once per trench that has
# an event.
#
# load_burst_csv defaults any burst_* column the CSV lacks to False, so a viewer built with
# this list against an older CSV renders an empty layer rather than raising.
#
# ORDER IS MOST-NUMEROUS FIRST, and it is load-bearing: the outlines are true mask contours,
# so two definitions that agree on a cell draw the SAME path and only the topmost is visible.
# v12 (1,091 events) is emitted last so it wins the z-order over v7 (164). Reversed, v7 would
# vanish underneath v12 wherever the two agree -- which is most of the time. Do not reorder.
BURST_VERSIONS = ("v7", "v12")

# One stroke colour per definition, so they can be overlaid and told apart. These two hexes
# are the ones the 091026 build shipped; keep them, so a screenshot from either build reads
# the same way.
BURST_COLORS = {"v7":  "rgb(255,235,0)",     # yellow   the precise RULE
                "v12": "rgb(255,0,200)"}     # magenta  the trained CLASSIFIER




def load_burst_csv(path):
    """Load the notebook's burst table, keyed to match this viewer's loop variables.

    dtypes are forced because a bare read_csv turns FOV '015' into 15 and trench_id '1488'
    into 1488, which then match no card -- the same trap documented in
    mmtrack/track_displacement.py:load_cell_data.
    """
    if not path:
        return {}
    df = pd.read_csv(
        path, dtype={"experiment_name": str, "FOV": str, "trench_id": str, "gene": str}
    )
    df["FOV"] = df["FOV"].str.zfill(3)
    for v in BURST_VERSIONS:
        col = f"burst_{v}"
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].astype(str).str.lower().isin(("true", "1", "t", "yes"))
    df["track_id"] = df["track_id"].astype(int)
    df["time_frame"] = df["time_frame"].astype(int)

    by_trench = {}
    for (exp, fov, peak), g in df.groupby(["experiment_name", "FOV", "trench_id"]):
        by_trench[(exp, fov, peak)] = g.reset_index(drop=True)
    n = {v: int(df[f"burst_{v}"].sum()) for v in BURST_VERSIONS}
    print(f"  burst CSV: {len(df)} rows over {len(by_trench)} trenches | " +
          " | ".join(f"{v}={n[v]}" for v in BURST_VERSIONS))
    return by_trench


def render_bursts_svg(bursts, ctc_masks, img_width, kymo_shape, downscale, version):
    """SVG outlining each burst cell for one definition, as a true contour from ctc_masks.

    Geometry mirrors render_labels_svg: kymograph x = time_frame * img_width + image_x, then
    everything scaled by `downscale`. Two things that are easy to get wrong and are load-bearing:

      * `time_frame` is the 0-based index into the trimmed stack, i.e. the kymograph COLUMN.
        It must NOT be offset by the time_dict `start` -- start is only used for the printed
        tick labels, so adding it would slide every outline sideways.
      * find_contours returns (row, col) = (y, x). Swapping them transposes the outline.

    Each outline is captioned `<track_id> y<centroid_y_rel>` when the CSV carries
    `centroid_y_rel` (see `mmtrack.bursts.add_relative_y`): the outline says WHERE, and the
    caption says which cell and how far along the trench, so a burst read off the image can be
    matched back to a table row without counting cells by eye. `centroid_y_rel` runs 0 at the
    open end to 1 at the closed (mother) end.

    Returns ("", n_missing) when there is nothing to draw for this definition.
    """
    from skimage import measure

    if bursts is None or len(bursts) == 0 or ctc_masks is None:
        return "", 0
    sel = bursts[bursts[f"burst_{version}"]]
    if sel.empty:
        return "", 0
    has_rel = "centroid_y_rel" in sel.columns

    h, w = kymo_shape
    color = BURST_COLORS[version]
    n_frames = ctc_masks.shape[0]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(w * downscale)}" '
        f'height="{int(h * downscale)}" viewBox="0 0 {int(w * downscale)} '
        f'{int(h * downscale)}" style="position:absolute;top:0;left:0;">'
    ]
    n_missing = 0
    for row in sel.itertuples():
        t, tid = int(row.time_frame), int(row.track_id)
        if not (0 <= t < n_frames):
            n_missing += 1
            continue
        binary = ctc_masks[t] == tid
        if not binary.any():
            n_missing += 1
            continue
        drew = False
        for contour in measure.find_contours(binary.astype(float), 0.5):
            pts = " ".join(
                f"{((cx + t * img_width) * downscale):.1f},{(cy * downscale):.1f}"
                for cy, cx in contour            # (row, col) -> (y, x)
            )
            parts.append(
                f'<polygon points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="1.5" stroke-opacity="0.95"/>'
            )
            drew = True
        if not drew:
            n_missing += 1
            continue
        # Caption centred on the cell and sat just above it. Centred, not right-aligned:
        # a kymograph frame is only ~20 px wide and the caption is ~45, so a right-aligned
        # one would hide the next two frames entirely -- centring splits that over both
        # sides and keeps the bursting cell itself clear. Placed from the mask rather than
        # from centroid_x, so it cannot drift if the two disagree.
        ys, xs = np.nonzero(binary)
        cap = f"{tid}"
        if has_rel and pd.notna(getattr(row, "centroid_y_rel", None)):
            cap += f" y{float(row.centroid_y_rel):.2f}"
        tx = (xs.mean() + t * img_width) * downscale
        ty = ys.min() * downscale - 3.0
        # Flip below the cell when it is too near the top to caption above it.
        if ty < 9.0:
            ty = min(ys.max() * downscale + 10.0, h * downscale - 2.0)
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" fill="{color}" font-size="9" '
            f'font-family="monospace" text-anchor="middle" paint-order="stroke" '
            f'stroke="rgba(0,0,0,0.85)" stroke-width="2.5">{cap}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts), n_missing


# ---------------------------------------------------------------------------
# Fluorescence-vs-time charts (mirrors notebook plot_track_fluorescence)
# ---------------------------------------------------------------------------

def compute_track_fluorescence(ctc_masks, fluor, start_frame):
    """Per-track mean fluorescence over time.

    Returns {track_id: [(abs_frame, mean_val), ...]} sorted by frame, or None when
    no fluorescence is available. Mean over fluor pixels where ctc_masks == label is
    identical to the 'intensity_mean_fluor' feature (mask label == track_id).
    """
    if fluor is None:
        return None

    n = min(ctc_masks.shape[0], fluor.shape[0])
    series = defaultdict(list)
    for t in range(n):
        mask_t = ctc_masks[t]
        labels = np.unique(mask_t)
        labels = labels[labels != 0]
        if labels.size == 0:
            continue
        means = ndimage.mean(fluor[t], labels=mask_t, index=labels)
        for lab, m in zip(labels, np.atleast_1d(means)):
            series[int(lab)].append((start_frame + t, float(m)))

    if not series:
        return None
    return {tid: sorted(pts) for tid, pts in series.items()}


def _linspace_ticks(lo, hi, n=4):
    if hi <= lo:
        return [lo]
    step = (hi - lo) / n
    return [lo + step * i for i in range(n + 1)]


def _render_one_panel(series, color_map, title, x_of, x_lo, x_hi, y_lo, y_hi,
                      width=460, height=150):
    """Render a single line-chart panel as an <svg>. x_of maps (frame, track_pts)->x value."""
    ml, mr, mt, mb = 52, 12, 24, 34  # margins
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    def sx(xv):
        if x_hi <= x_lo:
            return ml
        return ml + (xv - x_lo) / (x_hi - x_lo) * plot_w

    def sy(yv):
        if y_hi <= y_lo:
            return mt + plot_h
        return mt + plot_h - (yv - y_lo) / (y_hi - y_lo) * plot_h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" class="fluor-svg">'
    ]
    out.append(f'<text x="{ml}" y="14" class="fl-title">{title}</text>')
    # axes
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" class="fl-axis"/>')
    out.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" class="fl-axis"/>')
    # y ticks
    for yv in _linspace_ticks(y_lo, y_hi):
        yy = sy(yv)
        out.append(f'<line x1="{ml-3}" y1="{yy:.1f}" x2="{ml}" y2="{yy:.1f}" class="fl-axis"/>')
        out.append(f'<text x="{ml-5}" y="{yy+3:.1f}" class="fl-tick" text-anchor="end">{yv:.0f}</text>')
    # x ticks
    for xv in _linspace_ticks(x_lo, x_hi):
        xx = sx(xv)
        out.append(f'<line x1="{xx:.1f}" y1="{mt+plot_h}" x2="{xx:.1f}" y2="{mt+plot_h+3}" class="fl-axis"/>')
        out.append(f'<text x="{xx:.1f}" y="{mt+plot_h+14:.1f}" class="fl-tick" text-anchor="middle">{xv:.0f}</text>')

    for tid in sorted(series):
        pts = series[tid]
        color = color_map.get(tid, "#888")
        coords = " ".join(f"{sx(x_of(fr, pts)):.1f},{sy(val):.1f}" for fr, val in pts)
        out.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" stroke-opacity="0.9" data-track="{tid}" data-color="{color}"/>'
        )
    out.append("</svg>")
    return "\n".join(out)


def _assert_controls_balanced(html):
    """Fail the build if the .controls block does not close before {gene_panels}.

    Guards a bug that is silent but wrecks the layout: `.controls` is `position: sticky` and
    `.kymo-container` is `overflow-x: auto`, and BOTH stop working if `.controls` is left
    open, because every `.gene-panel` then becomes a flex item of a sticky box that spans the
    whole document -- zero sticky travel, and a max-content panel width that never forces the
    kymograph to scroll. It cost two releases (090826e, 090926) to notice, and the CSS was
    never at fault: a new ctrl-group had swallowed the `.controls` closer.
    """
    a = html.index('<div class="controls"')
    b = html.index('<div class="gene-panel')
    seg = html[a:b]
    bal = seg.count('<div') - seg.count('</div>')
    assert bal == 0, (
        f'.controls block is unbalanced by {bal:+d} <div> before the first .gene-panel. '
        'A ctrl-group is probably missing its closing </div> (see the FIVE EDITS comment). '
        'Left unfixed this silently disables the sticky control bar and kymograph scrolling.')


def render_fluor_charts(series, color_map, start_frame, end_frame, card_id,
                        filt_series=None, filt_title=None):
    """Raw-time panel, an optional filtered-values panel, and one cell_id dropdown.

    Returns HTML for the right-hand fluorescence panel of a kymograph card.

    `filt_series` adds a SECOND panel below the first, in the same shape (per-track mean vs
    absolute frame) but measured on the FIJI-transformed image, so a feature lines up
    vertically between the two and against the image layer beside them.

    Both panels go inside ONE `.fluor-panel` with ONE `card_id`, and that is load-bearing:
    the `.cell-select` handler does `panel.querySelectorAll('polyline')` over the whole
    panel, so a single dropdown highlights both traces with no JS change. Emitting a second
    `.fluor-panel` with the same `data-card` would half-break it -- `querySelector` is
    singular, so the second panel's lines would never respond to the selector.

    Y axes are INDEPENDENT on purpose. The raw panel is in a.u.; the filtered panel is in
    counts above that trench's own display floor, where most cells read exactly 0. Forcing a
    shared scale would flatten the filtered trace, and its being empty except during a burst
    is the informative part.
    """
    if not series:
        return (
            f'<div class="fluor-panel" data-card="{card_id}">'
            f'<div class="fluor-none">No fluorescence data</div></div>'
        )

    all_vals = [v for pts in series.values() for _, v in pts]
    y_lo, y_hi = min(all_vals), max(all_vals)

    raw_panel = _render_one_panel(
        series, color_map, "Mean fluorescence — raw time",
        x_of=lambda fr, pts: fr, x_lo=start_frame, x_hi=end_frame,
        y_lo=y_lo, y_hi=y_hi,
    )

    panels = [raw_panel]
    if filt_series:
        _fv = [v for pts in filt_series.values() for _, v in pts]
        # y_lo pinned at 0, not min(_fv): the floor IS zero for this quantity, and letting it
        # float would hide how much of the trace sits on the axis.
        panels.append(_render_one_panel(
            filt_series, color_map,
            filt_title or "Mean fluorescence — FIJI clip + median r2",
            x_of=lambda fr, pts: fr, x_lo=start_frame, x_hi=end_frame,
            y_lo=0.0, y_hi=max(_fv) if _fv else 0.0,
        ))

    options = ['<option value="">All tracks</option>']
    options += [f'<option value="{tid}">cell {tid}</option>' for tid in sorted(series)]

    return f"""\
<div class="fluor-panel" data-card="{card_id}">
  <div class="fluor-select">
    <label>Highlight lineage:
      <select class="cell-select" data-card="{card_id}">{''.join(options)}</select>
    </label>
  </div>
  {''.join(panels)}
</div>"""


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<title>DuMM Kymograph Viewer</title>
<style>
  :root {{
    --bg: #1a1a2e; --fg: #e0e0e0; --card-bg: #16213e;
    --tab-bg: #0f3460; --tab-active: #e94560; --border: #333;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg: #f5f5f5; --fg: #222; --card-bg: #fff;
      --tab-bg: #ddd; --tab-active: #e94560; --border: #ccc;
    }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--fg); }}
  header {{ padding: 16px 24px; font-size: 1.4em; font-weight: 700;
            border-bottom: 2px solid var(--border); }}
  header span {{ font-size: 0.6em; font-weight: 400; opacity: 0.6; margin-left: 12px; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 10px 24px;
         border-bottom: 1px solid var(--border); }}
  nav button {{ padding: 8px 18px; border: none; border-radius: 6px 6px 0 0;
                cursor: pointer; font-size: 0.95em; font-weight: 600;
                background: var(--tab-bg); color: var(--fg); transition: background 0.15s; }}
  nav button.active {{ background: var(--tab-active); color: #fff; }}
  nav button:hover:not(.active) {{ filter: brightness(1.2); }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 18px; padding: 12px 24px;
               align-items: center; border-bottom: 1px solid var(--border);
               position: sticky; top: 0; background: var(--bg); z-index: 10; }}
  .ctrl-group {{ display: flex; align-items: center; gap: 6px; }}
  .ctrl-group label {{ font-size: 0.85em; cursor: pointer; user-select: none; }}
  .gene-panel {{ display: none; padding: 16px 24px; }}
  .gene-panel.active {{ display: block; }}
  .kymo-card {{ margin-bottom: 24px; background: var(--card-bg);
                border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }}
  .kymo-card h3 {{ padding: 10px 16px; font-size: 0.85em; font-weight: 500;
                   border-bottom: 1px solid var(--border); }}
  .kymo-card h3 strong {{ color: var(--tab-active); }}
  .kymo-body {{ display: flex; gap: 16px; align-items: flex-start; padding: 12px 16px; }}
  .kymo-container {{ position: relative; overflow-x: auto; overflow-y: hidden;
                     line-height: 0; flex: 1 1 auto; min-width: 0; }}
  .kymo-container img, .kymo-container svg {{
    display: block; max-width: none;
  }}
  .kymo-container .layer {{ position: absolute; top: 0; left: 0; }}
  .fluor-panel {{ flex: 0 0 auto; }}
  .fluor-select {{ font-size: 0.85em; margin-bottom: 6px; }}
  .fluor-select select {{ background: var(--tab-bg); color: var(--fg);
                          border: 1px solid var(--border); border-radius: 4px; padding: 3px 6px; }}
  .fluor-svg {{ display: block; }}
  .fluor-svg .fl-axis {{ stroke: var(--fg); stroke-opacity: 0.5; stroke-width: 1; }}
  .fluor-svg .fl-tick {{ fill: var(--fg); font-size: 9px; opacity: 0.7; }}
  .fluor-svg .fl-title {{ fill: var(--fg); font-size: 11px; font-weight: 600; }}
  .fluor-none {{ font-size: 0.85em; opacity: 0.6; padding: 8px; }}
  .summary {{ padding: 8px 24px; font-size: 0.8em; opacity: 0.6; }}
</style>

<header>DuMM Bacteria Tracker &mdash; Kymograph Viewer<span>{n_kymos} kymographs across {n_genes} genes</span></header>
<nav id="gene-tabs">{gene_tabs}</nav>
<div class="controls" id="layer-controls">
  <div class="ctrl-group">
    <input type="checkbox" id="chk-phase" checked>
    <label for="chk-phase">Phase</label>
  </div>
  <div class="ctrl-group">
    <input type="checkbox" id="chk-fluor">
    <label for="chk-fluor">Fluorescence</label>
  </div>
  <div class="ctrl-group">
    <input type="checkbox" id="chk-fluorfilt">
    <label for="chk-fluorfilt" title="ImageJ/FIJI Process &gt; Filters &gt; Median at
radius 2.0 -- a 21-point disk, not a 3x3 square and not skimage's disk(2), which is 13
points and equals ImageJ radius 1.75. Edges replicate, as ImageJ does. Run on the
ASSEMBLED kymograph, matching a FIJI median on fluor_kymos, so at frame boundaries the
kernel reaches into the neighbouring TIME POINT -- columns 0, 1, 18 and 19 of every
20-px frame. Small-object removal is OFF by default (--small-object-min-size 0), so this
layer is the pure two-step FIJI chain, clip then median. Raising it zeroes visible
components below that many pixels, which is BEYOND the FIJI workflow and acts as a
detection parameter rather than a denoiser -- at 10 px it cost 4.4% of visible real-cell
pixels on a dim trench and 0.0% on bright ones. Lossless PNG, unlike every other image layer.
Displayed over [0.25 x max, max] of this trench's RAW stack pooled over every frame --
FIJI's "minimum at 25% of the max", rounded half-up as you would type it, and NOT a
25th-percentile clip, which is 0 on this background-subtracted data. DISPLAY ONLY -- the fluor_filtered column in the CSVs and burst definitions
v4-v7 all still use the 3x3 median.">Fluorescence (FIJI median r2)</label>
  </div>
  <div class="ctrl-group">
    <input type="checkbox" id="chk-fluornorm">
    <label for="chk-fluornorm" title="Each cell's mean fluorescence rescaled min-max within
this trench only. Dark = dimmest cell in this trench, bright yellow = brightest. Per-CELL, not
per-pixel, and independent of the (gene, experiment) thresholds detection uses.">Fluor
(trench-normalized) <span style="font-family:monospace;font-size:10px;">[<span
style="color:#440154;">&#9608;</span><span style="color:#31688e;">&#9608;</span><span
style="color:#35b779;">&#9608;</span><span style="color:#fde725;">&#9608;</span>]</span></label>
  </div>
  <div class="ctrl-group">
    <input type="checkbox" id="chk-mask" checked>
    <label for="chk-mask">Masks</label>
  </div>
  <div class="ctrl-group">
    <input type="checkbox" id="chk-tracks" checked>
    <label for="chk-tracks">Tracks</label>
  </div>
  <div class="ctrl-group">
    <input type="checkbox" id="chk-labels">
    <label for="chk-labels">Labels</label>
  </div>
  <div class="ctrl-group">
    <input type="checkbox" id="chk-burstv7">
    <label for="chk-burstv7" style="color:#ffeb00;"
      title="BURST v7 -- the most PRECISE definition available: P 0.838 against the hand
labels, at about a quarter of v12's recall. A closed-form rule, so when it fires you can say
exactly why: it is v6 (height bar + MAD prominence + two fold routes) plus a fourth route,
the per-trench Tukey fence Q3 + 4.0*IQR. 164 events.
CAVEAT: v7 is the first definition that does NOT pass the constitutive control cleanly --
one hupA event, at 012/1165 frame 81, bright (7.73 against a 6.17 bar) but NOT prominent,
which is exactly the failure a marginal distribution cannot see. Check a yellow-only outline
by eye before trusting it. bursts.md section 7.">Burst v7 (rule)</label>
  </div>
  <div class="ctrl-group">
    <input type="checkbox" id="chk-burstv12">
    <label for="chk-burstv12" style="color:#ff00c8;"
      title="BURST v12 -- the TRAINED CLASSIFIER, at its shipped 0.75 threshold. 1,091
events. Out of fold, one-to-one matched at +-1 frame: P 0.607 / R 0.739, against v11's
0.592 / 0.429 -- it dominates every rule on recall. It has NO closed-form definition and no
threshold to quote; the honest gloss from local-occlusion ranking is "brightest in its frame,
top few percent of its trench, raised above its own recent past", all within-trench.
TWO KNOWN FAILURES, reported not hidden: it fires 2 events in the certified burst-free trench
at 0.75 (0 at 0.90), and it is far worse than the rules on the constitutive controls --
hupA 25 / mazF 57 / chpS 94 against v11's 0 / 9 / 34 -- because it has NEVER SEEN a control
gene. Labelling a few hupA/mazF/chpS trenches is the single highest-return fix.
bursts.md section 14.">Burst v12 (classifier)</label>
  </div>
</div>
{gene_panels}
<script>
(function() {{
  const tabs = document.querySelectorAll('#gene-tabs button');
  const panels = document.querySelectorAll('.gene-panel');
  const layers = ['phase','fluor','fluorfilt','fluornorm','mask','tracks','labels',
                  'burstv7','burstv12'];

  function switchTab(gene) {{
    tabs.forEach(t => t.classList.toggle('active', t.dataset.gene === gene));
    panels.forEach(p => p.classList.toggle('active', p.id === 'panel-' + gene));
  }}
  tabs.forEach(t => t.addEventListener('click', () => switchTab(t.dataset.gene)));

  function applyLayers() {{
    layers.forEach(layer => {{
      const chk = document.getElementById('chk-' + layer);
      const show = chk.checked;
      document.querySelectorAll('.layer-' + layer).forEach(el => {{
        el.style.display = show ? '' : 'none';
      }});
    }});
  }}
  layers.forEach(layer => {{
    document.getElementById('chk-' + layer).addEventListener('change', applyLayers);
  }});
  applyLayers();

  // Per-card lineage highlight in the fluorescence charts (plot only).
  document.querySelectorAll('.cell-select').forEach(sel => {{
    sel.addEventListener('change', () => {{
      const card = sel.dataset.card;
      const sel_id = sel.value;
      const panel = document.querySelector('.fluor-panel[data-card="' + card + '"]');
      if (!panel) return;
      panel.querySelectorAll('polyline').forEach(pl => {{
        const color = pl.dataset.color;
        if (sel_id === '') {{
          pl.setAttribute('stroke', color);
          pl.setAttribute('stroke-width', '1.5');
          pl.setAttribute('stroke-opacity', '0.9');
        }} else if (pl.dataset.track === sel_id) {{
          pl.setAttribute('stroke', color);
          pl.setAttribute('stroke-width', '2.5');
          pl.setAttribute('stroke-opacity', '1');
        }} else {{
          pl.setAttribute('stroke', '#888');
          pl.setAttribute('stroke-width', '1');
          pl.setAttribute('stroke-opacity', '0.3');
        }}
      }});
    }});
  }});
}})();
</script>
"""


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_viewer(args):
    # Relative --time-dict paths resolve against the repo, not the caller's cwd, so the
    # documented command works from anywhere.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    burst_csv_path = getattr(args, "burst_csv", None)
    if burst_csv_path and not os.path.isabs(burst_csv_path):
        burst_csv_path = os.path.join(script_dir, burst_csv_path)
    burst_by_trench = load_burst_csv(burst_csv_path)
    burst_missing_total = 0
    time_dict_path = args.time_dict
    if not os.path.isabs(time_dict_path):
        time_dict_path = os.path.join(script_dir, time_dict_path)

    with open(time_dict_path) as f:
        time_dict = json.load(f)
    print(f"Time dict: {time_dict_path} ({len(time_dict)} experiments)")

    combined_dict = STRAIN_EXP_DICT

    if getattr(args, "genes", None):
        wanted = {g.strip() for g in args.genes.split(",") if g.strip()}
        combined_dict = {g: v for g, v in combined_dict.items() if g in wanted}
        missing = wanted - set(combined_dict)
        if missing:
            print(f"WARNING: requested genes not found in mapping: {sorted(missing)}")
        print(f"Restricting to genes: {sorted(combined_dict)}")

    kymo_list = build_kymograph_list(combined_dict, time_dict, args.trackastra_dir)
    print(f"Found {len(kymo_list)} kymographs to render")
    # Fail here, clearly, rather than 200 lines later. With nothing to render no .gene-panel is
    # emitted, and _assert_controls_balanced then dies on `html.index('<div class="gene-panel')`
    # with `ValueError: substring not found` -- which points at the control-bar markup and is
    # the wrong diagnosis entirely. The real cause is almost always that the SMB share holding
    # the .npy stacks has dropped, since that is the only input this list is built from.
    if not kymo_list:
        raise SystemExit(
            f'No kymographs to render. Nothing was built from:\n'
            f'  --trackastra-dir {args.trackastra_dir}\n'
            f'  --base-dir       {args.base_dir}\n'
            f'  --time-dict      {args.time_dict}\n'
            'Check the share is mounted (ls /Volumes/) before blaming the code -- this is the '
            'usual symptom of an unmounted volume, not of a viewer bug.')
    print(f"Fluor filter: {args.fluor_filter} "
          f"(radius={args.filter_radius}, size={args.filter_size}, "
          f"sigma={args.filter_sigma}, nlm_h={args.nlm_h}, "
          f"otsu_scale={args.otsu_scale})")
    print(f"Filtered-layer display range: vmin = {args.fluor_low_frac} x vmax "
          f"(rounded half-up), vmax = p{args.fluor_high_pct} of each trench's raw "
          f"fluor stack")
    print(f"Small-object removal: min_size={args.small_object_min_size} px on "
          f"visible components"
          + ("  (DISABLED -- exact FIJI)" if args.small_object_min_size <= 0 else ""))

    kymos_by_gene = defaultdict(list)
    for gene, exp, fov, peak in kymo_list:
        kymos_by_gene[gene].append((exp, fov, peak))

    gene_tabs_html = ""
    gene_panels_html = ""
    genes = sorted(kymos_by_gene.keys())
    total_rendered = 0

    for gi, gene in enumerate(genes):
        active = " active" if gi == 0 else ""
        gene_tabs_html += f'<button data-gene="{gene}" class="{active.strip()}">{gene} ({len(kymos_by_gene[gene])})</button>\n'

        cards_html = ""
        entries = kymos_by_gene[gene]
        for exp, fov, peak in tqdm(entries, desc=f"  {gene}", leave=False):
            data = load_kymograph_data(
                exp, fov, peak, time_dict, args.base_dir, args.trackastra_dir,
                fluor_filter=args.fluor_filter,
                filter_size=args.filter_size,
                filter_sigma=args.filter_sigma,
                nlm_h=args.nlm_h,
                otsu_scale=args.otsu_scale,
                filter_radius=args.filter_radius,
            )
            if data is None:
                continue

            imgs = data["imgs"]
            ctc_masks = data["ctc_masks"]
            tracks = data["tracks"]
            graph = data["graph"]
            fluor = data["fluor"]
            fluor_filtered = data["fluor_filtered"]
            start_frame = data["start_frame"]
            end_frame = data["end_frame"]

            color_map = build_track_color_map(tracks[:, 0]) if tracks is not None and len(tracks) else {}

            phase_b64, kymo_shape, label_bar_h = render_phase(imgs, args.downscale, args.jpeg_quality, start_frame)
            fluor_b64 = render_fluor(fluor, args.downscale, args.jpeg_quality) if fluor is not None else None
            # FIJI display range, read off the RAW stack and applied to the filtered
            # layer only. The raw layer keeps its own 1-99 stretch, so this cannot
            # change how it looks.
            filt_vmin, filt_vmax = fluor_display_range(
                fluor, args.fluor_low_frac, args.fluor_high_pct
            )
            # fiji_median filters the assembled kymograph and is emitted lossless, so
            # the layer's encoding varies -- the <img> tag below can no longer hardcode
            # jpeg.
            fluor_filt_b64, fluor_filt_fmt = None, "jpeg"
            if fluor is not None and args.fluor_filter == "fiji_median":
                fluor_filt_b64 = render_fluor_fiji(
                    fluor, args.filter_radius, args.downscale, filt_vmin, filt_vmax,
                    min_size=args.small_object_min_size, thr=filt_vmin,
                )
                fluor_filt_fmt = "png"
            elif fluor_filtered is not None and args.fluor_filter != "none":
                fluor_filt_b64 = render_fluor(
                    fluor_filtered, args.downscale, args.jpeg_quality,
                    filt_vmin, filt_vmax
                )
            mask_b64 = render_mask(ctc_masks, args.downscale)
            tracks_svg = render_tracks_svg(tracks, graph, imgs.shape[2], kymo_shape, args.downscale, color_map)
            labels_svg = render_labels_svg(tracks, imgs.shape[2], kymo_shape, args.downscale, color_map)

            # Burst outlines: one layer per definition so two can be overlaid to see
            # where they disagree. Keyed on (exp, fov, peak) exactly as the card is.
            trench_bursts = burst_by_trench.get((exp, fov, peak))
            burst_svgs = {}
            for _bv in BURST_VERSIONS:
                _svg, _miss = render_bursts_svg(
                    trench_bursts, ctc_masks, imgs.shape[2], kymo_shape,
                    args.downscale, _bv,
                )
                burst_svgs[_bv] = _svg
                burst_missing_total += _miss

            fluor_series = compute_track_fluorescence(ctc_masks, fluor, start_frame)
            fluor_norm_b64 = render_fluor_normalized(
                ctc_masks, fluor_series, start_frame, args.downscale
            )
            card_id = f"{exp}_{fov}_{peak}"
            # Second chart: the same per-track mean, measured on the FIJI-transformed image.
            # Rebuilt here rather than read from data["fluor_filtered"], which is None under
            # --fluor-filter fiji_median by design (the filtering happens at render time, on
            # the assembled kymograph). This is byte-for-byte the array compute_fiji_fluor
            # measures, so this chart and the fiji_mean column cannot disagree.
            fluor_filt_series = None
            if fluor is not None and args.fluor_filter == "fiji_median":
                _k = imagej_median(to_kymograph(fluor), args.filter_radius)
                _k = remove_speckle(_k, args.small_object_min_size, thr=filt_vmin)
                _fiji_stack = from_kymograph(
                    np.clip(_k.astype(np.float32) - filt_vmin, 0, None), fluor.shape[0]
                )
                fluor_filt_series = compute_track_fluorescence(
                    ctc_masks, _fiji_stack, start_frame
                )
            fluor_charts = render_fluor_charts(
                fluor_series, color_map, start_frame, end_frame, card_id,
                filt_series=fluor_filt_series,
            )

            kymo_w = int(kymo_shape[1] * args.downscale)
            kymo_h = int(kymo_shape[0] * args.downscale)
            total_h = kymo_h + label_bar_h

            fluor_img = ""
            if fluor_b64:
                fluor_img = (
                    f'<img class="layer layer-fluor" '
                    f'src="data:image/jpeg;base64,{fluor_b64}" '
                    f'style="display:none;height:{kymo_h}px;" />'
                )

            fluor_filt_img = ""
            if fluor_filt_b64:
                fluor_filt_img = (
                    f'<img class="layer layer-fluorfilt" '
                    f'src="data:image/{fluor_filt_fmt};base64,{fluor_filt_b64}" '
                    f'style="display:none;height:{kymo_h}px;" />'
                )

            fluor_norm_img = ""
            if fluor_norm_b64:
                fluor_norm_img = (
                    f'<img class="layer layer-fluornorm" '
                    f'src="data:image/png;base64,{fluor_norm_b64}" '
                    f'style="display:none;height:{kymo_h}px;" />'
                )

            tracks_div = ""
            if tracks_svg:
                tracks_div = f'<div class="layer layer-tracks" style="height:{kymo_h}px;">{tracks_svg}</div>'

            burst_divs = ""
            for _bv in BURST_VERSIONS:
                if burst_svgs.get(_bv):
                    burst_divs += (
                        f'<div class="layer layer-burst{_bv}" '
                        f'style="height:{kymo_h}px;display:none;">{burst_svgs[_bv]}</div>'
                    )

            labels_div = ""
            if labels_svg:
                labels_div = (
                    f'<div class="layer layer-labels" '
                    f'style="height:{kymo_h}px;display:none;">{labels_svg}</div>'
                )

            cards_html += f"""\
<div class="kymo-card">
  <h3><strong>{gene}</strong> &mdash; {exp} | FOV {fov} | Trench {peak} | Frames {start_frame}&ndash;{end_frame}</h3>
  <div class="kymo-body">
    <div class="kymo-container">
      <div style="width:{kymo_w}px;height:{total_h}px;"></div>
      <img class="layer layer-phase" src="data:image/jpeg;base64,{phase_b64}" />
      {fluor_img}
      {fluor_filt_img}
      {fluor_norm_img}
      <img class="layer layer-mask" src="data:image/png;base64,{mask_b64}" style="height:{kymo_h}px;" />
      {tracks_div}
      {labels_div}
      {burst_divs}
    </div>
    {fluor_charts}
  </div>
</div>
"""
            total_rendered += 1

        gene_panels_html += f'<div class="gene-panel{active}" id="panel-{gene}">{cards_html}</div>\n'

    if burst_by_trench and burst_missing_total:
        print(f"  WARNING: {burst_missing_total} burst row(s) had no matching label in "
              f"ctc_masks[time_frame] and were not outlined")

    html = HTML_TEMPLATE.format(
        n_kymos=total_rendered,
        n_genes=len(genes),
        gene_tabs=gene_tabs_html,
        gene_panels=gene_panels_html,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    _assert_controls_balanced(html)
    with open(args.output, "w") as f:
        f.write(html)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone! Wrote {args.output} ({file_size_mb:.1f} MB, {total_rendered} kymographs)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML kymograph viewer with layer toggling"
    )
    parser.add_argument(
        "--base-dir",
        default="/Volumes/mcovert/Instruments/Covert-lab-scope1/subgen_processed_data",
        help="Root directory for subtracted image data",
    )
    parser.add_argument(
        "--trackastra-dir",
        default="/Volumes/mcovert/Instruments/Covert-lab-scope1/track_test/track_astra_output",
        help="Directory containing trackastra .npy outputs",
    )
    parser.add_argument(
        "--time-dict",
        default="data/time_dict_082026.json",
        help="time_dict JSON giving the start/end frame window per trench. Only "
             "experiments and FOVs present here are rendered.",
    )
    parser.add_argument(
        "--output",
        default="data/kymograph_viewer.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=1.0,
        help="Downscale factor for images (0.5 = half size, 1.0 = full)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for the phase and raw-fluorescence layers (1-100). Does "
             "NOT apply to the filtered layer under 'fiji_median', which is emitted as "
             "lossless PNG.",
    )
    parser.add_argument(
        "--fluor-filter",
        choices=["none", "median", "mean", "gaussian", "nlm", "otsu", "fiji_median"],
        default="fiji_median",
        help="Denoising filter for the 'Fluorescence (FIJI median r2)' layer. "
             "'fiji_median' = ImageJ Process>Filters>Median at --filter-radius, i.e. "
             "the manual FIJI workflow (default); applied to the assembled kymograph, "
             "so the kernel spans frame seams as FIJI's does, and emitted lossless. "
             "'nlm' = non-local means. "
             "'otsu' = median + global Otsu background subtraction. "
             "'none' disables the filtered layer.",
    )
    parser.add_argument(
        "--filter-radius",
        type=float,
        default=2.0,
        help="Disk radius for 'fiji_median', in ImageJ RankFilters semantics: 2.0 is a "
             "21-point disk (x^2 + y^2 <= 5), not a 5x5 square and not skimage's "
             "disk(2), which is 13 points and equals ImageJ radius 1.75.",
    )
    parser.add_argument(
        "--small-object-min-size",
        type=int,
        default=0,
        help="Small-object removal after the median: connected components smaller than "
             "this many pixels are zeroed. DEFAULT 0 = OFF, so the layer is the pure "
             "two-step FIJI chain (clip + median) and reproduces FIJI exactly. It shipped "
             "at 10 up to 090826 and was turned off because it had become a DETECTION "
             "parameter rather than a denoiser: trenches read as fully dark only because "
             "sub-10px components were deleted. Objects are defined by the layer's own "
             "display minimum, so a nonzero value acts on what is VISIBLE -- at 10 px it "
             "cost 4.4%% of visible real-cell pixels on a dim trench and 0.0%% on bright "
             "ones.",
    )
    parser.add_argument(
        "--fluor-low-frac",
        type=float,
        default=0.25,
        help="Black point for the filtered layer as a FRACTION of the white point, i.e. "
             "FIJI's 'set the minimum to 25%% of the max'. Rounded half-up to an integer, "
             "as you would type it: 0.25 x max 34 -> 9. This is deliberately NOT a "
             "percentile of the pixel values -- that reading gives 0 on "
             "background-subtracted data and differs on 78%% of rendered pixels.",
    )
    parser.add_argument(
        "--fluor-high-pct",
        type=float,
        default=100.0,
        help="White point for the filtered layer, same pooling. 100 = the true max, so "
             "nothing saturates and bright cells keep their gradient (FIJI slider "
             "behaviour). Drop to 99.9 if a surviving hot pixel makes a trench look "
             "dim. The old raw-layer value, 99, flattens the top 1%%.",
    )
    parser.add_argument(
        "--filter-size",
        type=int,
        default=3,
        help="Square footprint size for the median / mean / otsu filters. Ignored by "
             "'fiji_median', which uses --filter-radius. Stays an int because scipy's "
             "size= wants one.",
    )
    parser.add_argument(
        "--filter-sigma",
        type=float,
        default=1.0,
        help="Standard deviation for the gaussian filter",
    )
    parser.add_argument(
        "--nlm-h",
        type=float,
        default=1.15,
        help="Non-local means strength multiplier (higher = smoother)",
    )
    parser.add_argument(
        "--otsu-scale",
        type=float,
        default=1.0,
        help="Multiplier on the global Otsu threshold for the 'otsu' filter "
             "(>1 = more aggressive background removal, <1 = keeps more dim signal)",
    )
    parser.add_argument(
        "--burst-csv",
        default=None,
        help="Burst table from the analysis notebook (data/bursts_viewer_v12_091026.csv). "
             "When given, adds one toggleable outline layer per definition in "
             "BURST_VERSIONS -- v7 (the precise rule) and v12 (the trained classifier), one "
             "outline per EVENT -- drawn as a contour cut from ctc_masks around each bursting "
             "cell and captioned '<track_id> y<centroid_y_rel>'. Columns the CSV lacks "
             "default to False, so an older table renders an empty layer rather than "
             "raising, and extra burst_* columns it carries are simply ignored.",
    )
    parser.add_argument(
        "--genes",
        default=None,
        help="Comma-separated genes to render (e.g. 'lacZ'). Default: all genes.",
    )
    args = parser.parse_args()
    generate_viewer(args)


if __name__ == "__main__":
    main()
