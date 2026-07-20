#!/usr/bin/env python
"""
Generate a self-contained HTML kymograph viewer with napari-like layer toggling.

Iterates over all trackastra-processed kymographs, renders phase/fluorescence/mask/track
layers, and embeds them into a single HTML file with interactive layer controls.

Usage:
    poetry run python generate_kymograph_viewer.py [--output data/kymograph_viewer.html]
"""

import argparse
import base64
import io
import json
import os
from collections import defaultdict

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Strain-experiment mapping (from notebooks/trackastra_figures.ipynb)
# ---------------------------------------------------------------------------

STRAIN_EXP_DICT_1 = {
    "chpS": {
        "DUMM_giTG62_Glucose_012925": ["005"],
        "DUMM_alkA_chpS_111325": ["003", "005", "006", "007", "009", "010", "011"],
    },
    "baeS": {"DUMM_giTG66_Glucose_012325": ["003"]},
    "lacZ": {
        "DUMM_giTG059_noKan_Glucose_031125": ["004"],
        "DUMM_giTG059_060_061125": ["008"],
    },
    "gfcE": {"DUMM_giTG064_Glucose_022625": ["001", "015"]},
    "gldA": {"DUMM_giTG69_Glucose_013025": ["007"]},
    "alkA": {
        "DUMM_giTG068_063_061725_v2": ["000"],
        "DUMM_giTG63_giTG67_Glucose_121724_1_v2": ["005"],
        "DUMM_alkA_chpS_111325": ["017", "018", "022", "023"],
    },
    "mazF": {"DUMM_giTG059_060_061125": ["015"]},
    "hupA": {
        "DUMM_giTG068_052925": ["005"],
        "DUMM_giTG068_063_061725": ["012"],
    },
    "araB": {
        "DUMM_CL008_giTG061_081225": ["015", "016", "017", "018", "019", "020"],
    },
}

STRAIN_EXP_DICT_2 = {
    "chpS": {"DUMM_giTG062_064_121325": ["005", "010", "012"]},
    "baeS": {
        "DUMM_gitg068_baeS_100225": ["013", "017", "018", "019", "020", "021", "022", "023"],
        "DUMM_baeS_gfcE_112025": ["005"],
        "DUMM_giTG066_063_120925": ["002", "007"],
    },
    "gfcE": {
        "DUMM_baeS_gfcE_112025": ["011", "012", "016"],
        "DUMM_giTG062_064_121325": ["015", "019"],
        "DUMM_giTG060_064_121425": ["013", "014", "020", "022"],
    },
    "gldA": {
        "DUMM_giTG069_063_121125": ["013", "016", "022", "025"],
        "DUMM_giTG069_063_121225": ["013", "018", "022"],
    },
    "alkA": {
        "DUMM_giTG066_063_120925": ["018", "019"],
        "DUMM_giTG069_063_121125": ["001", "005", "006", "008"],
        "DUMM_giTG069_063_121225": ["000"],
    },
    "mazF": {
        "DUMM_giTG060_064_121425": ["000", "005"],
        "DUMM_mazF_murQ_121625": ["005", "006", "012"],
    },
    "murQ": {"DUMM_mazF_murQ_121625": ["016", "017", "018"]},
}

STRAIN_EXP_DICT_3 = {
    "lacZ": {"DUMM_giTG059_068_SC_093025": ["015", "020"]},
    "araB": {"CL008_giTG068_072925": ["003", "007", "009"]},
}


def _merge_dicts(dict1, dict2):
    merged = defaultdict(dict)
    for key in dict1:
        if key in dict2:
            for subkey, values in dict1[key].items():
                if subkey in dict2[key]:
                    merged[key][subkey] = list(set(values + dict2[key][subkey]))
                else:
                    merged[key][subkey] = values
            for subkey, values in dict2[key].items():
                if subkey not in merged[key]:
                    merged[key][subkey] = values
        else:
            merged[key] = dict1[key]
    for key in dict2:
        if key not in merged:
            merged[key] = dict2[key]
    return dict(merged)


def build_combined_dict():
    combined = _merge_dicts(STRAIN_EXP_DICT_1, STRAIN_EXP_DICT_2)
    return _merge_dicts(combined, STRAIN_EXP_DICT_3)


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


def load_kymograph_data(exp, fov, peak, time_dict, base_dir, trackastra_dir):
    """Load all arrays for a single kymograph. Returns dict or None on failure."""
    imgs_path = os.path.join(trackastra_dir, f"{exp}_imgs_{fov}_{peak}.npy")
    ctc_path = os.path.join(trackastra_dir, f"{exp}_ctc_masks_{fov}_{peak}.npy")
    tracks_path = os.path.join(trackastra_dir, f"{exp}_napari_tracks_{fov}_{peak}.npy")
    graph_path = os.path.join(trackastra_dir, f"{exp}_napari_tracks_graph_{fov}_{peak}.json")

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
    if os.path.exists(graph_path):
        try:
            with open(graph_path) as f:
                graph = {int(k): int(v) for k, v in json.load(f).items()}
        except Exception:
            pass

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

    start_frame = time_dict[exp][fov][peak].get("start", 0)
    end_frame = start_frame + imgs.shape[0] - 1

    return {
        "imgs": imgs,
        "ctc_masks": ctc_masks,
        "tracks": tracks,
        "graph": graph,
        "fluor": fluor,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _normalize_to_uint8(arr):
    """Percentile-based contrast stretch to uint8."""
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, (1, 99))
    if p99 <= p1:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - p1) / (p99 - p1) * 255, 0, 255)
    return arr.astype(np.uint8)


def _to_kymograph(stack):
    """(t, y, x) -> (y, t*x)"""
    t, y, x = stack.shape
    return stack.transpose(1, 0, 2).reshape(y, -1)


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


def render_fluor(fluor, downscale, quality):
    kymo = _to_kymograph(fluor)
    kymo8 = _normalize_to_uint8(kymo)
    pil = Image.fromarray(kymo8, mode="L")
    if downscale != 1.0:
        new_size = (max(1, int(pil.width * downscale)), max(1, int(pil.height * downscale)))
        pil = pil.resize(new_size, Image.LANCZOS)
    return _img_to_base64(pil, "JPEG", quality)


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
    """Return an inline SVG overlay with a track_id text label at each track's start.

    Mirrors the napari 'track_labels' points layer: one label per lineage, placed at
    the cell's first appearance, colored to match its track/plot color.
    """
    if tracks is None or len(tracks) == 0:
        return ""

    h, w = kymo_shape
    svg_w = int(w * downscale)
    svg_h = int(h * downscale)

    track_ids = tracks[:, 0]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}" style="position:absolute;top:0;left:0;">'
    ]
    for tid in np.unique(track_ids):
        rows = tracks[track_ids == tid]
        first = rows[np.argmin(rows[:, 1])]  # earliest time_frame
        x = (first[1] * img_width + first[3]) * downscale
        y = first[2] * downscale
        color = color_map.get(tid, "#ffff00")
        lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="10" fill="{color}" '
            f'style="paint-order:stroke;stroke:#000;stroke-width:2px;stroke-linejoin:round;" '
            f'dominant-baseline="hanging">{int(tid)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


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

    from scipy import ndimage

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
                      width=460, height=200):
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


def render_fluor_charts(series, color_map, start_frame, end_frame, card_id):
    """Two stacked panels (raw time, elapsed-per-track) + a cell_id dropdown.

    Returns HTML for the right-hand fluorescence panel of a kymograph card.
    """
    if not series:
        return (
            f'<div class="fluor-panel" data-card="{card_id}">'
            f'<div class="fluor-none">No fluorescence data</div></div>'
        )

    all_vals = [v for pts in series.values() for _, v in pts]
    y_lo, y_hi = min(all_vals), max(all_vals)
    max_elapsed = max((pts[-1][0] - pts[0][0]) for pts in series.values())

    raw_panel = _render_one_panel(
        series, color_map, "Mean fluorescence — raw time",
        x_of=lambda fr, pts: fr, x_lo=start_frame, x_hi=end_frame,
        y_lo=y_lo, y_hi=y_hi,
    )
    elapsed_panel = _render_one_panel(
        series, color_map, "Mean fluorescence — frames since track start",
        x_of=lambda fr, pts: fr - pts[0][0], x_lo=0, x_hi=max_elapsed,
        y_lo=y_lo, y_hi=y_hi,
    )

    options = ['<option value="">All tracks</option>']
    options += [f'<option value="{tid}">cell {tid}</option>' for tid in sorted(series)]

    return f"""\
<div class="fluor-panel" data-card="{card_id}">
  <div class="fluor-select">
    <label>Highlight lineage:
      <select class="cell-select" data-card="{card_id}">{''.join(options)}</select>
    </label>
  </div>
  {raw_panel}
  {elapsed_panel}
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
</div>
{gene_panels}
<script>
(function() {{
  const tabs = document.querySelectorAll('#gene-tabs button');
  const panels = document.querySelectorAll('.gene-panel');
  const layers = ['phase','fluor','mask','tracks','labels'];

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    time_dict_path = os.path.join(script_dir, "data", "time_dict_070826.json")

    with open(time_dict_path) as f:
        time_dict = json.load(f)

    combined_dict = build_combined_dict()
    kymo_list = build_kymograph_list(combined_dict, time_dict, args.trackastra_dir)
    print(f"Found {len(kymo_list)} kymographs to render")

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
            data = load_kymograph_data(exp, fov, peak, time_dict, args.base_dir, args.trackastra_dir)
            if data is None:
                continue

            imgs = data["imgs"]
            ctc_masks = data["ctc_masks"]
            tracks = data["tracks"]
            graph = data["graph"]
            fluor = data["fluor"]
            start_frame = data["start_frame"]
            end_frame = data["end_frame"]

            color_map = build_track_color_map(tracks[:, 0]) if tracks is not None and len(tracks) else {}

            phase_b64, kymo_shape, label_bar_h = render_phase(imgs, args.downscale, args.jpeg_quality, start_frame)
            fluor_b64 = render_fluor(fluor, args.downscale, args.jpeg_quality) if fluor is not None else None
            mask_b64 = render_mask(ctc_masks, args.downscale)
            tracks_svg = render_tracks_svg(tracks, graph, imgs.shape[2], kymo_shape, args.downscale, color_map)
            labels_svg = render_labels_svg(tracks, imgs.shape[2], kymo_shape, args.downscale, color_map)

            fluor_series = compute_track_fluorescence(ctc_masks, fluor, start_frame)
            card_id = f"{exp}_{fov}_{peak}"
            fluor_charts = render_fluor_charts(
                fluor_series, color_map, start_frame, end_frame, card_id
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

            tracks_div = ""
            if tracks_svg:
                tracks_div = f'<div class="layer layer-tracks" style="height:{kymo_h}px;">{tracks_svg}</div>'

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
      <img class="layer layer-mask" src="data:image/png;base64,{mask_b64}" style="height:{kymo_h}px;" />
      {tracks_div}
      {labels_div}
    </div>
    {fluor_charts}
  </div>
</div>
"""
            total_rendered += 1

        gene_panels_html += f'<div class="gene-panel{active}" id="panel-{gene}">{cards_html}</div>\n'

    html = HTML_TEMPLATE.format(
        n_kymos=total_rendered,
        n_genes=len(genes),
        gene_tabs=gene_tabs_html,
        gene_panels=gene_panels_html,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
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
        help="JPEG quality for phase/fluor layers (1-100)",
    )
    args = parser.parse_args()
    generate_viewer(args)


if __name__ == "__main__":
    main()
