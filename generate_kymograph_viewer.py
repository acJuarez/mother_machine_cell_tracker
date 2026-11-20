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
from mmtrack.experiment_metadata import STRAIN_EXP_DICT, FAILED_KYMOGRAPHS  # noqa: E402


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
                    # Individually excluded kymographs are never rendered. See
                    # experiment_metadata.FAILED_KYMOGRAPHS for the reason per entry.
                    if (exp, fov, peak) in set(FAILED_KYMOGRAPHS):
                        continue
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
# mmtrack.cell_tables assembles frames exactly the way this renderer does. A second
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


def render_labels_svg(tracks, img_width, kymo_shape, downscale, color_map, font_size=12.0):
    """Return an inline SVG overlay with a track_id text label on EVERY cell-frame.

    One label per cell-frame, not one per lineage. Through 090926 this mirrored the napari
    'track_labels' points layer and drew a single label at each lineage's FIRST appearance,
    which is unusable for identifying a cell mid-movie: to learn the id of a cell at frame 80
    you had to trace its polyline back through division connectors to wherever that lineage
    started. Reading a `track_id` straight off the screen is what a human needs in order to
    hand-record a burst the automated definitions missed (the `manual` layer), so
    every cell-frame now carries its own number.

    `tracks` is the napari array [track_id, t, y, x] with exactly one row per cell-frame --
    verified, 300 rows against 300 cell-frames on DUMM_giTG060_064_121425 013/153 -- so this
    is a loop change, not new data. Its ids are the ctc mask labels, i.e. the same `track_id`
    the cell tables use, because relabel_tracks_to_ctc has already run; a number read off
    this layer therefore joins directly to all_data_annotated_082026.csv. That equality is
    the entire point of the layer, so do not "optimise" it to use the raw napari ids.

    Font size is `--label-font-size`, default 12 px. It was 8 px through 100226, which keeps a
    3-digit id inside its own 20 px frame at --downscale 1.0 but proved too small to read; at
    12 px adjacent frames' ids can overlap, so drop back to 8 if they collide. `text-anchor=
    middle` centres each number on its cell. Colour still matches the track and chart colour,
    and the black paint-order stroke (scaled with the font) keeps it readable over any layer.
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
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{font_size:g}" fill="{color}" '
            f'style="paint-order:stroke;stroke:#000;stroke-width:{font_size / 4:g}px;'
            f'stroke-linejoin:round;" '
            f'text-anchor="middle" dominant-baseline="middle">{int(tid)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Burst outlines (from the notebook's burst CSV)
# ---------------------------------------------------------------------------

# What the viewer draws: the three DEFINITIONS plus the hand-annotated GROUND TRUTH, all from
# the union table `07_build_bursts.py` writes (mmtrack.bursts.OUTPUTS['union']).
#
# `manual` is NOT a definition -- nothing detects it; it is the reference the three are
# judged against: 837 hand-marked bursts over 208 trenches and 7 genes. At the exact frame v1
# lands on 92 of them and misses 745, v2 lands on 555 and misses 282, v3 lands on 407 and
# misses 430 (v3 is closer within +-1 frame -- see bursts.md section 7).
#
#   v1  the four-route rule: the most PRECISE, at about a quarter of v2's recall
#   v2  the classifier: the best RECALL
#   v3  the within-trench monotone classifier: the best per-gene ORDERING
#
# Per-definition detail: bursts.md sections 3, 4 and 5.
#
# The burst outline layers come in SETS, chosen with --burst-layers. A set is an ordered tuple
# of BurstLayer specs, WIDEST FIRST: the checkboxes, the JS `layers` array, the colours and the
# widths are all generated from it, so adding a layer is ONE edit here (plus its column in the
# burst CSV). That replaced a seven-edit recipe whose two traps -- a JS entry with no checkbox
# aborting the whole IIFE, and a new ctrl-group swallowing the `.controls` closer -- both shipped.
#
# ORDER IS WIDEST FIRST, and that is load-bearing. Outlines are true mask contours, so two layers
# that agree on a cell draw the IDENTICAL polygon; only strictly decreasing stroke widths keep an
# agreement legible as concentric haloes (cyan alone = a real burst every definition missed).
# FOUR layers is the ceiling: the narrowest stroke is already 1.0 px.
#
#   v1v2v3manual (default)  manual 100226 / v1 rule / v2 classifier / v3 monotone model
#
# load_burst_csv defaults any burst_* column the CSV lacks to False, so a set built against an
# older CSV renders an empty layer rather than raising (but ALL layers empty is an error).
from collections import namedtuple

BurstLayer = namedtuple("BurstLayer", "version color hex width checked label title")

BURST_LAYER_SETS = {
  "v1v2v3manual": (
    BurstLayer('manual', 'rgb(0,229,255)', '#00e5ff', 5.0, True,
               'Burst manual (hand-annotated)',
               'HAND-ANNOTATED BURSTS -- the GROUND TRUTH, not a definition. 837 bursts marked by\neye on this viewer, over 208 trenches and 7 genes (baeS 282, gfcE 129, chpS 100, alkA 93,\naraB 89, gldA 74, lacZ 70). Nothing detects these; they are the reference the three definitions\nare judged against, and they are drawn WIDEST so the others nest inside them.\nHOW TO READ IT: cyan halo ALONE = a real burst BOTH definitions missed. cyan+magenta = v2\ncaught it, v1 missed it. cyan+yellow+magenta = both caught it. magenta with NO cyan = v2 fired\nwhere nothing was annotated. Measured at the exact frame, v1 lands on 92 of the 837 and misses\n745; v2 lands on 555 and misses 282; v3 lands on 407 and misses 430.\nSource data/annotated_bursts_100226.csv -> data/bursts_manual_100226.csv; 19 rows did not\nresolve to a cell-frame and are in data/burst_labels_recheck_100226.csv for re-checking.\nbursts.md section 6.'),
    BurstLayer('v1', 'rgb(255,235,0)', '#ffeb00', 3.5, False,
               'Burst v1 (rule)',
               "BURST v1 -- THE RULE, and the most PRECISE definition available:\nP 0.838 against the hand labels, at about a quarter of v2's recall. Closed-form, so when it\nfires you can say exactly why: four OR'd routes, and the `route` column says which. 158\nevents = absolute 97 (height bar + MAD prominence) + fold_left 36 + fold 18 + tukey 7 (the\nper-trench fence Q3 + 4.0*IQR).\nCAVEAT: v1 does NOT pass the constitutive control perfectly -- one hupA event, at 012/1165\nframe 81, bright (7.73 against a 6.17 bar) but NOT prominent, which is exactly the failure a\nmarginal distribution cannot see. Check a yellow-only outline by eye before trusting it.\nbursts.md section 3."),
    BurstLayer('v2', 'rgb(255,0,200)', '#ff00c8', 2.25, False,
               'Burst v2 (classifier)',
               'BURST v2 -- THE TRAINED CLASSIFIER, at its shipped 0.75 threshold.\n1,041 events. Out of fold, one-to-one matched at +-1 frame: P 0.607 / R 0.739 against v1\'s\n0.838 / 0.281 -- it beats every rule on recall and is beaten by v1 on precision. It has NO\nclosed-form definition and no threshold to quote; the honest gloss from local-occlusion\nranking is "brightest in its frame, top few percent of its trench, raised above its own\nrecent past", all within-trench.\nIt was trained on lacZ and gfcE only. Its mazF/hupA events (57/25) are unvalidated:\nthose genes were never hand-reviewed, so nothing is known about them in either direction.\nbursts.md section 4.'),
    BurstLayer('v3', 'rgb(0,230,118)', '#00e676', 1.0, False,
               'Burst v3 (within-trench)',
               'BURST v3 -- THE WITHIN-TRENCH MONOTONE CLASSIFIER, at threshold 0.90.\n1,065 events. Every feature is computed inside a single kymograph and every one of the 26\nconstrained features has a physically-defensible sign (brighter, risen, fallen, brightest in\nframe => more burst-like). Held out: P 0.560 / R 0.700 / F1 0.622 on 57 trenches never\ntrained on, against dev 0.628 -- it generalises.\nIt does NOT beat v2 on F1 (0.719 on the same holdout). What it does beat it on is the per-gene\nORDERING: Spearman against the hand labels is 0.714 for v3, 0.607 for v2 and -0.393 for v1.\nNo closed form -- a depth-3 surrogate reaches only 0.405, so do not quote a threshold.\nbursts.md section 5.'),
  ),
}
DEFAULT_BURST_LAYERS = "v1v2v3manual"


def _layer_globals(name):
    layers = BURST_LAYER_SETS[name]
    ws = [l.width for l in layers]
    assert all(a > b for a, b in zip(ws, ws[1:])), f"{name}: widths must strictly decrease"
    assert len(layers) <= 4, f"{name}: four layers is the ceiling"
    return (tuple(l.version for l in layers), {l.version: l.color for l in layers},
            {l.version: l.width for l in layers})


# Module globals are the ACTIVE set (default until generate_viewer rebinds them); every renderer
# reads these, and render_bursts_svg indexes them bare so a missing entry is a loud KeyError.
BURST_VERSIONS, BURST_COLORS, BURST_WIDTHS = _layer_globals(DEFAULT_BURST_LAYERS)


def burst_controls_html(name):
    return "".join(
        f'  <div class="ctrl-group">\n    <input type="checkbox" id="chk-burst{l.version}"'
        f'{" checked" if l.checked else ""}>\n'
        f'    <label for="chk-burst{l.version}" style="color:{l.hex};"\n'
        f'      title="{l.title}">{l.label}</label>\n  </div>\n'
        for l in BURST_LAYER_SETS[name])


def burst_layers_js(name):
    return ",".join(f"'burst{l.version}'" for l in BURST_LAYER_SETS[name])




def load_burst_csv(path):
    """Load the notebook's burst table, keyed to match this viewer's loop variables.

    dtypes are forced because a bare read_csv turns FOV '015' into 15 and trench_id '1488'
    into 1488, which then match no card -- the same trap documented in
    mmtrack/cell_tables.py:load_cell_data.
    """
    if not path:
        return {}
    df = pd.read_csv(
        path, dtype={"experiment_name": str, "FOV": str, "trench_id": str, "gene": str}
    )
    df["FOV"] = df["FOV"].str.zfill(3)
    # Recorded BEFORE the default-fill below, or the error message at the end of this function
    # reports the columns it just invented as though the table had carried them.
    burst_cols_on_disk = sorted(c for c in df.columns if c.startswith("burst_"))
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
    # A single empty layer is legitimate -- an older table simply lacks that column, and the
    # default-to-False above is there on purpose. But EVERY layer empty on a non-empty table is
    # never intentional: it means the columns are spelled for a different generation, and the
    # build would otherwise emit a complete, fully-interactive ~170 MB viewer with no outlines
    # at all, announcing it only as "v1=0 | v2=0" in the middle of a long log.
    if len(df) and not any(n.values()):
        raise SystemExit(
            f"\n{path}: none of {list(BURST_VERSIONS)} matched a column. The table carries "
            f"{burst_cols_on_disk}.\nThis table predates the current definitions; "
            f"regenerate it with `python 07_build_bursts.py`.")
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
    # Bare dict index on purpose: a version added to BURST_VERSIONS without a width is a
    # loud KeyError here, not a silent layer that is invisible wherever it agrees.
    width = BURST_WIDTHS[version]
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
                f'stroke-width="{width}" stroke-opacity="0.95"/>'
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
        'A ctrl-group is probably missing its closing </div> (see the BURST_LAYER_SETS comment). '
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
every burst definition still detects on the unfiltered column.">Fluorescence (FIJI median r2)</label>
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
{burst_controls}</div>
{gene_panels}
<script>
(function() {{
  const tabs = document.querySelectorAll('#gene-tabs button');
  const panels = document.querySelectorAll('.gene-panel');
  const layers = ['phase','fluor','fluorfilt','fluornorm','mask','tracks','labels',
                  {burst_layers_js}];

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
    global BURST_VERSIONS, BURST_COLORS, BURST_WIDTHS
    layer_set = getattr(args, "burst_layers", DEFAULT_BURST_LAYERS)
    BURST_VERSIONS, BURST_COLORS, BURST_WIDTHS = _layer_globals(layer_set)
    print(f"  burst layers: {layer_set} {BURST_VERSIONS}")
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
            labels_svg = render_labels_svg(tracks, imgs.shape[2], kymo_shape, args.downscale, color_map,
                                           font_size=args.label_font_size)

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
        burst_controls=burst_controls_html(layer_set),
        burst_layers_js=burst_layers_js(layer_set),
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
        "--label-font-size",
        type=float,
        default=12.0,
        help="Font size (px) of the track_id on every cell-frame in the Labels layer. "
             "Default 12; 8 (the size through 100226) is the largest that keeps a 3-digit "
             "id inside one 20 px frame, so use it if adjacent ids collide.",
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
        default="data/bursts_viewer_v1v2v3manual_100226.csv",
        help="Burst table (default data/bursts_viewer_v1v2v3manual_100226.csv). Adds one "
             "toggleable outline layer per entry of the --burst-layers set; the default set is `manual` (the 837 "
             "hand-annotated bursts, cyan, widest), `v1` (the precise rule, 158 events, "
             "yellow), `v2` (the trained classifier, 1,041, magenta) and `v3` (the "
             "within-trench monotone model, 1,065, green) -- one outline per "
             "EVENT, drawn as a contour cut from ctc_masks around each bursting cell and "
             "captioned '<track_id> y<centroid_y_rel>'. Widths are NESTED so agreements are "
             "legible: a cyan halo with nothing inside it is a real burst both definitions "
             "missed. A column the CSV lacks defaults to False, so an older table renders an "
             "empty layer rather than raising, and extra burst_* columns are ignored; but if "
             "EVERY layer comes back empty the build fails, since that only ever means the "
             "table predates this generation.",
    )
    parser.add_argument(
        "--burst-layers",
        choices=sorted(BURST_LAYER_SETS),
        default=DEFAULT_BURST_LAYERS,
        help="Which set of outline layers to draw (BURST_LAYER_SETS). 'v1v2v3manual' "
             "(default): the 100226 hand labels, v1, v2, v3 -- pair with the default "
             "--burst-csv.",
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
