"""Figure style, palette and plots for the trackastra summary notebooks.

This is the presentation layer that used to live inline in
``notebooks/analysis_notebooks/trackastra_figures_082026.ipynb`` -- roughly 1,150 lines of it,
spread over 21 cells. It was hoisted here for three reasons:

* **The house style was already duplicated.** ``_style``/``_save`` existed in both that
  notebook and ``mmtrack/track_displacement.py``, and the notebook's two Delta histograms were
  explicit forks of the module's, taken because the module hardcodes a log y-axis and names
  ``intensity_mean_fluor`` in an axis label that is wrong once the table is in a.u./ms.
* **The gene palette must be pinned in one place.** ``GENE_COLOR`` maps gene to colour
  explicitly rather than positionally, because a positional map silently reassigns every
  colour when a gene is added, dropped, or the frame is scoped to a subset -- and then two
  figures in the same document disagree about what blue means.
* **Three functions used to default to a notebook global.** ``track_lifetimes``,
  ``division_events`` and ``bursts_by_gene`` all took ``cells=None`` and then reached for a
  module-level ``all_data``. Here every frame is passed in.

The notebook globals those functions closed over (``FIG_DIR``, ``SAVE_FIGS``, ``LABEL``,
``SUFFIX``) are now module defaults set once by :func:`configure`, with a per-call override on
every plot.

**Units.** The annotated exports carry four fluorescence columns and they are not
interchangeable:

===========================  ==========================================================
``intensity_mean_fluor``     raw a.u.; not comparable across experiments, which run at
                             150-500 ms exposure
``fluor_per_ms``             raw / ``fluor_exposure_ms``. The default here, and the only
                             one comparable across genes.
``fluor_filtered``           re-measured on median-filtered images. A few percent dimmer
                             than raw, and by a different factor per experiment (0.41 to
                             1.00), so it is on its own scale -- never plot it against raw.
``fluor_filtered_per_ms``    the filtered column, exposure-normalised
===========================  ==========================================================

Likewise on the pair table: ``delta_fluor`` is **raw a.u.**, and ``delta_fluor_per_ms`` is the
exposure-normalised one. Every function here defaults to the per-ms column, and takes the
column name as an argument so the choice is always visible at the call site.
"""

import os

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmtrack.experiment_metadata import FAILED_EXPERIMENTS
# Colours and the Delta-y axis contract are shared with stage 7 rather than redefined, so a
# Delta-y figure means the same thing whichever produced it.
from mmtrack.track_displacement import (
    C_ALL, C_GOOD, C_JUMP, C_MAIN, DY_DEFAULT_RANGE, DY_VALUE_COL, DY_XLABEL,
)

# ---------------------------------------------------------------------------------------
# Keys and column names -- the schema contract with the annotated exports
# ---------------------------------------------------------------------------------------

KYMO_KEY = ['experiment_name', 'FOV', 'trench_id']       # one kymograph / peak_id
CELL_KEY = KYMO_KEY + ['track_id']                       # one unique cell
KYMO_GROUP = ['experiment_name', 'gene', 'FOV', 'trench_id']   # as the exports group min/max

RAW_FLUOR_COL = 'intensity_mean_fluor'
EXPOSURE_COL = 'fluor_exposure_ms'
FLUOR_COL = 'fluor_per_ms'
DFLUOR_COL = 'delta_fluor_per_ms'

FLUOR_LABEL = 'mean fluorescence per ms exposure (a.u./ms)'
RAW_FLUOR_LABEL = 'mean fluorescence inside the cell mask (a.u.)'
DFLUOR_LABEL = ('Δfluor increase = fluor(t+1) − fluor(t)   '
                '[a.u./ms, bg-subtracted, > 0 only]')

# Axis label per fluorescence column, so a figure cannot mislabel its own units. The per-ms
# division only matters when POOLING ACROSS EXPERIMENTS: `fluor_exposure_ms` is one constant per
# experiment (asserted at load), so within a kymograph -- or within a (gene, experiment), the
# scope `compute_baselines` uses -- dividing by it is an exactly affine rescale and changes no
# verdict. Measured: 0 verdict disagreements over all 255 kymographs for both a median+k*MAD bar
# and a p95 bar. Across experiments it does matter, and not in one direction: every gene but murQ
# spans 2-3 exposure settings, and `figures.md` section 3 measures the replicate spread SHRINKING
# for 4 genes and GROWING for 5, because exposure is confounded with acquisition.
FLUOR_LABELS = {
    RAW_FLUOR_COL: RAW_FLUOR_LABEL,
    FLUOR_COL: FLUOR_LABEL,
    'fluor_filtered': 'mean fluorescence, median-filtered images (a.u.)',
    'fluor_filtered_per_ms': ('mean fluorescence, median-filtered, '
                              'per ms exposure (a.u./ms)'),
}


def fluor_label(col):
    """Axis label for a fluorescence column, falling back to the column name."""
    return FLUOR_LABELS.get(col, col)

# `predicted_lineage` holds '1', '1.1', '3.1.1'; pandas infers a numeric dtype on any subset
# without dots. FOV '003' and trench_id '1488' become integers without this and then match
# nothing -- the trap documented in track_displacement.load_cell_data.
READ_DTYPES = {'experiment_name': str, 'FOV': str, 'trench_id': str,
               'gene': str, 'predicted_lineage': str}

CELLS_CSV = 'data/all_data_annotated_082026.csv'
PAIRS_CSV = 'data/track_pairs_annotated_082026.csv'
BURSTS_CSV = 'data/bursts_082026.csv'

# ---------------------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------------------

_CFG = {'out_dir': 'notebooks/figures', 'save': False, 'label': '', 'suffix': ''}


def configure(out_dir=None, save=None, label=None, suffix=None):
    """Set the module-wide figure defaults. Call once per notebook.

    `label` goes into figure titles and `suffix` into filenames, so a re-run under a
    narrower scope neither mislabels a figure nor overwrites the previous one.
    """
    for k, v in (('out_dir', out_dir), ('save', save),
                 ('label', label), ('suffix', suffix)):
        if v is not None:
            _CFG[k] = v
    return dict(_CFG)


def _cfg(key, override=None):
    return _CFG[key] if override is None else override


def style(ax):
    """House style: drop the top and right spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_fig(fig, name, out_dir=None, save=None):
    """Write an SVG + 200 dpi PNG pair and return the SVG path, or None if saving is off.

    Deliberately does NOT close the figure -- unlike ``track_displacement._save`` -- so the
    same call renders inline in a notebook. Named `save_fig` rather than `save` because
    every plot function takes a `save=` flag, which would shadow it.
    """
    if not _cfg('save', save):
        return None
    out_dir = _cfg('out_dir', out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f'{out_dir}/{name}.svg', transparent=True, bbox_inches='tight')
    fig.savefig(f'{out_dir}/{name}.png', dpi=200, bbox_inches='tight')
    return f'{out_dir}/{name}.svg'


def _apply_yscale(ax, yscale, base):
    """Set the y-scale, label it, and return the filename suffix that keeps the log and
    linear variants from overwriting each other's SVG/PNG pair."""
    ax.set_yscale(yscale)
    ax.set_ylabel(f'{base} (log)' if yscale == 'log' else base)
    return '' if yscale == 'log' else '_linear'


# ---------------------------------------------------------------------------------------
# Gene palette
# ---------------------------------------------------------------------------------------

# Fixed gene -> tab10 colour. Pinned explicitly rather than derived from the data: a
# positional map silently reassigns every colour when a gene is added, dropped, or the frame
# is scoped to a subset. Ten genes, ten tab10 slots.
GENE_COLOR = {
    'alkA': '#1f77b4',   # blue
    'araB': '#ff7f0e',   # orange
    'baeS': '#2ca02c',   # green
    'chpS': '#d62728',   # red
    'gfcE': '#9467bd',   # purple
    'gldA': '#8c564b',   # brown
    'hupA': '#e377c2',   # pink
    'lacZ': '#7f7f7f',   # gray
    'mazF': '#bcbd22',   # olive
    'murQ': '#17becf',   # cyan
}

# Genes seen at runtime that are not in GENE_COLOR. Module state, mutated by gene_color().
_unmapped = {}


def gene_color(gene):
    """Colour for a gene.

    A new strain should be added to GENE_COLOR deliberately, so anything unmapped gets a
    tab20 fallback and says so once rather than silently borrowing another gene's colour.
    """
    if gene in GENE_COLOR:
        return GENE_COLOR[gene]
    if gene not in _unmapped:
        _unmapped[gene] = plt.get_cmap('tab20')(len(_unmapped) % 20)
        print(f'NOTE: {gene!r} is not in GENE_COLOR; using a tab20 fallback. '
              f'Add it to the map to pin its colour.')
    return _unmapped[gene]


def _shades(base, n):
    """n tints of `base`, dark -> light. Splits a gene's bar by experiment."""
    rgb = np.array(mcolors.to_rgb(base))
    factors = [0.0] if n == 1 else np.linspace(0.0, 0.6, n)
    return [tuple(rgb * (1 - f) + f) for f in factors]


def _label_color(rgb):
    """White on dark fills, dark grey on light ones."""
    r, g, b = mcolors.to_rgb(rgb)
    return 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else '0.2'


# Acquisition interval, not exposure: doubling time is quantized by minutes_per_frame.
# Green/purple deliberately avoid the bar colour (blue).
INTERVAL_COLOR = {5: '#2ca02c', 10: '#9467bd'}
_ERR = dict(ecolor='0.3', lw=0.9, capsize=2)

# One colour per exposure setting. Exposure is ordinal (150 < 250 < 450 < 500) and a
# categorical palette cannot show that, so the legend is sorted ascending to carry the
# ordering instead. Slots chosen to avoid the bar colour (blue).
_EXPOSURE_CYCLE = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']


def _interval_color(mins):
    return INTERVAL_COLOR.get(int(mins), '0.5')


def exposure_colors(df, col=EXPOSURE_COL):
    """Map each exposure present in `df` to a stable colour."""
    return {e: _EXPOSURE_CYCLE[i % len(_EXPOSURE_CYCLE)]
            for i, e in enumerate(sorted(df[col].unique()))}


# ---------------------------------------------------------------------------------------
# Loading the annotated exports
# ---------------------------------------------------------------------------------------

def _read_annotated(csv_path, name, drop_failed, data_root=''):
    """Read one annotated export: dtypes, the schema asserts, and the failed-run drop.

    Shared by :func:`load_cells` and :func:`load_annotated` so the two cannot drift apart on
    the dtype map or on what counts as an annotated file.
    """
    df = pd.read_csv(os.path.join(data_root, csv_path), dtype=READ_DTYPES)
    assert df['FOV'].str.len().eq(3).all(), f'{name}: FOV lost its zero padding'
    assert 'is_failed_experiment' in df.columns, (
        f'{name}: no is_failed_experiment column -- this is not an annotated export')
    if drop_failed:
        df = df[~df['is_failed_experiment']].copy()
    return df


def load_cells(cells_csv=CELLS_CSV, drop_failed=True, data_root=''):
    """Load ONLY the annotated cell table -- one row per cell per frame.

    The cells half of :func:`load_annotated`, and worth its own entry point rather than
    discarding a second return value: the pair CSV is 65 MB, and an analysis that is meant to
    be pair-free should not be able to read one by accident. The distribution route to bursts
    (`plot_fluor_histogram`, `plot_fluor_histogram_scoped`, `kymo_threshold_table`) is
    entirely per-cell-frame, as is all of :mod:`mmtrack.bursts`.

    See :func:`load_annotated` for what the annotated exports carry beyond the stage-8 table.
    """
    return _read_annotated(cells_csv, 'cells', drop_failed, data_root)


def load_annotated(cells_csv=CELLS_CSV, pairs_csv=PAIRS_CSV, drop_failed=True,
                   data_root=''):
    """Load the annotated cell and pair tables. Returns (cells, pairs).

    These are the outputs of section 11 of the 082026 notebook, and they already carry
    everything that notebook derived at runtime: `fluor_per_ms`, `fluor_filtered`, the
    per-kymograph min/max columns, `is_degenerate_mask`, `is_failed_experiment`, and the
    per-endpoint `from_track_id`/`to_track_id` keys. Nothing here re-derives anything.

    Use :func:`load_cells` when nothing downstream reads a pair -- it skips a 65 MB file.

    Args:
        drop_failed: drop the three failed acquisitions (default). The files carry all 23
            experiments; every published figure in the 082026 notebook was on the 20 kept
            ones, so True reproduces those numbers. Pass False to see the full set --
            but note `load_bursts` cannot be mixed with it, see there.
        data_root: prefix for both paths, for calling from outside the repo root.
    """
    cells = _read_annotated(cells_csv, 'cells', drop_failed, data_root)
    pairs = _read_annotated(pairs_csv, 'pairs', drop_failed, data_root)
    assert 'from_track_id' in pairs.columns, (
        'pairs: no from_track_id column -- this is not an annotated export')
    return cells, pairs


def load_bursts(bursts_csv=BURSTS_CSV, cells=None, data_root='', version='active'):
    """Load the detected burst peaks for ONE definition.

    Detection itself is not re-run here -- see :mod:`mmtrack.bursts`. This reads its output.

    **The CSV is the union of every definition**, one row per peak found by *any* of them,
    with a boolean column per definition. It is not a table of the active definition's
    events: of its 91 rows, 73 are v1 and 89 are v3, and 2 are v1-only. Not filtering was a
    real bug -- Table F reported 91 bursts and lacZ 35 where v3 gives 89 and lacZ 33.

    Args:
        version: which definition's events to keep, via the `burst_{version}` column.
            `'active'` (default) uses `BURST_DEF['version']`. Pass None for the raw union,
            which is what you want when comparing definitions against each other.
        cells: if given, checked against the burst table. This is not cosmetic: bursts were
            detected on the 20 kept experiments, while the annotated exports carry all 23.
            A burst *rate* computed against a 23-experiment denominator is silently wrong,
            so passing `cells` here is how you find that out.
    """
    bursts = pd.read_csv(os.path.join(data_root, bursts_csv), dtype=READ_DTYPES)
    bursts['FOV'] = bursts['FOV'].str.zfill(3)

    if version == 'active':
        version = _burst_def()['version']
    if version is not None:
        col = f'burst_{version}'
        if col not in bursts.columns:
            raise ValueError(
                f'{bursts_csv} has no {col!r} column; it carries '
                f'{sorted(c for c in bursts.columns if c.startswith("burst_"))}. '
                'Re-run detection to add it, or pass version=None for the union.')
        n_all = len(bursts)
        bursts = bursts[bursts[col]].reset_index(drop=True)
        print(f'{os.path.basename(bursts_csv)}: {len(bursts)} {version} bursts '
              f'({n_all - len(bursts)} rows belong to other definitions only)')

    if cells is not None:
        extra = set(bursts['experiment_name']) - set(cells['experiment_name'])
        assert not extra, (
            f'the burst table holds experiments absent from `cells`: {sorted(extra)}')
        failed = set(bursts['experiment_name']) & set(FAILED_EXPERIMENTS)
        assert not failed, f'the burst table holds failed runs: {sorted(failed)}'
        if cells['experiment_name'].isin(FAILED_EXPERIMENTS).any():
            raise ValueError(
                'bursts were detected with the failed runs excluded, but `cells` still '
                'contains them. Any per-cell or per-pair burst rate would use a denominator '
                'the numerator never saw. Load with drop_failed=True.')
    return bursts


# ---------------------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------------------

def track_frame_counts(df):
    """Frames observed per unique cell."""
    return df.groupby(CELL_KEY, observed=True)['time_frame'].nunique()


def scope(df, gene=None, experiments=None, fovs=None, min_frames=1):
    """Filter the cell table. `None` means no filter on that axis."""
    out = df
    if gene is not None:
        out = out[out['gene'].isin([gene] if isinstance(gene, str) else list(gene))]
    if experiments is not None:
        exps = [experiments] if isinstance(experiments, str) else list(experiments)
        out = out[out['experiment_name'].isin(exps)]
    if fovs is not None:
        f = [fovs] if isinstance(fovs, str) else list(fovs)
        out = out[out['FOV'].isin([str(x).zfill(3) for x in f])]
    out = out.copy()
    if min_frames and min_frames > 1:
        n = track_frame_counts(out)
        keep = n[n >= min_frames].index
        idx = out.set_index(CELL_KEY)
        out = idx[idx.index.isin(keep)].reset_index()
    return out


def scope_pairs(pairs, gene=None, experiments=None, fovs=None):
    """Same filters as `scope`, for the pair table.

    `min_frames` is deliberately absent: a pair already implies a track observed in at
    least two frames, and re-litigating a threshold the pair construction has passed would
    only drop pairs whose own endpoints are fine.
    """
    out = pairs
    if gene is not None:
        out = out[out['gene'].isin([gene] if isinstance(gene, str) else list(gene))]
    if experiments is not None:
        exps = [experiments] if isinstance(experiments, str) else list(experiments)
        out = out[out['experiment_name'].isin(exps)]
    if fovs is not None:
        f = [fovs] if isinstance(fovs, str) else list(fovs)
        out = out[out['FOV'].isin([str(x).zfill(3) for x in f])]
    return out.copy()


def scope_label(gene=None, experiments=None, fovs=None, min_frames=1, drop_failed=True):
    """Human-readable tag for figure titles.

    The failed-run exclusion is appended when it applies, so no figure can silently omit
    the fact that three acquisitions are missing.
    """
    bits = [gene if isinstance(gene, str)
            else ('all genes' if gene is None else '+'.join(gene))]
    if experiments is not None:
        exps = [experiments] if isinstance(experiments, str) else list(experiments)
        bits.append(exps[0] if len(exps) == 1 else f'{len(exps)} experiments')
    if fovs is not None:
        f = [fovs] if isinstance(fovs, str) else list(fovs)
        bits.append('FOV ' + ','.join(str(x).zfill(3) for x in f))
    if min_frames and min_frames > 1:
        bits.append(f'tracks ≥{min_frames} frames')
    if drop_failed:
        bits.append(f'{len(FAILED_EXPERIMENTS)} failed runs excluded')
    return ' | '.join(bits)


def scope_suffix(gene=None, experiments=None, fovs=None, min_frames=1):
    """Filename suffix, so a re-run under a different scope does not overwrite files."""
    bits = []
    if gene is not None:
        bits.append(gene if isinstance(gene, str) else '-'.join(gene))
    if experiments is not None:
        exps = [experiments] if isinstance(experiments, str) else list(experiments)
        bits.append(exps[0] if len(exps) == 1 else f'{len(exps)}exp')
    if fovs is not None:
        f = [fovs] if isinstance(fovs, str) else list(fovs)
        bits.append('fov' + '-'.join(str(x).zfill(3) for x in f))
    if min_frames and min_frames > 1:
        bits.append(f'min{min_frames}')
    return ('_' + '_'.join(bits)) if bits else ''


def describe_scope(df, label='loaded'):
    """One-line inventory: rows, experiments, genes, kymographs, unique cells."""
    n_kymo = len(df[KYMO_KEY].drop_duplicates())
    n_cell = len(df[CELL_KEY].drop_duplicates())
    print(f'{label}: {len(df):,} rows | {df.experiment_name.nunique()} experiments | '
          f'{df.gene.nunique()} genes | {n_kymo:,} kymographs | {n_cell:,} unique cells')


def select_trench(df, exp, fov, peak_id):
    """Rows for one trench. Tolerant of int-or-str FOV / peak_id.

    Both are strings in every table (FOV zero-padded to 3), but they read as integers, so
    accepting either is the difference between a plot and an empty frame.
    """
    return df[(df['experiment_name'] == exp)
              & (df['FOV'].astype(str).str.zfill(3) == str(fov).zfill(3))
              & (df['trench_id'].astype(str) == str(peak_id))]


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------

def summarize_by_gene(df):
    """Per-gene inventory: experiments, FOVs, kymographs, cells, singletons, cell-hours.

    `n_singleton` counts cells seen in exactly one frame -- the most likely segmentation
    noise, so treat the count and its percentage as a data-quality read.

    The TOTAL row is recomputed over the whole frame rather than summed: 10 of the 23
    experiments hold two genes in different FOVs, so summing the per-gene column would
    double-count them.
    """
    cells = df[CELL_KEY + ['gene']].drop_duplicates()
    kymos = df[KYMO_KEY + ['gene']].drop_duplicates()
    fovs = df[['gene', 'experiment_name', 'FOV']].drop_duplicates()

    n_frames = track_frame_counts(df).rename('n_frames').reset_index()
    n_frames = n_frames.merge(cells, on=CELL_KEY, how='left')

    cells_per_kymo = (cells.groupby(KYMO_KEY, observed=True).size().rename('n')
                      .reset_index().merge(kymos, on=KYMO_KEY, how='left'))

    out = pd.DataFrame({
        'n_experiments':         df.groupby('gene')['experiment_name'].nunique(),
        'n_FOVs':                fovs.groupby('gene').size(),
        'n_kymographs':          kymos.groupby('gene').size(),
        'n_unique_cells':        cells.groupby('gene').size(),
        'n_singleton':           n_frames[n_frames.n_frames == 1].groupby('gene').size(),
        'median_cells_per_kymo': cells_per_kymo.groupby('gene')['n'].median(),
        'median_track_frames':   n_frames.groupby('gene')['n_frames'].median(),
        'max_track_frames':      n_frames.groupby('gene')['n_frames'].max(),
        'n_rows':                df.groupby('gene').size(),
        'cell_hours':            (df['minutes_per_frame'] / 60).groupby(df['gene']).sum(),
    })
    out['n_singleton'] = out['n_singleton'].fillna(0).astype(int)
    out = out.sort_values('n_unique_cells', ascending=False)

    all_frames = track_frame_counts(df)
    out.loc['TOTAL'] = {
        'n_experiments':         df['experiment_name'].nunique(),
        'n_FOVs':                len(df[['experiment_name', 'FOV']].drop_duplicates()),
        'n_kymographs':          len(df[KYMO_KEY].drop_duplicates()),
        'n_unique_cells':        len(df[CELL_KEY].drop_duplicates()),
        'n_singleton':           int((all_frames == 1).sum()),
        'median_cells_per_kymo': cells_per_kymo['n'].median(),
        'median_track_frames':   all_frames.median(),
        'max_track_frames':      all_frames.max(),
        'n_rows':                len(df),
        'cell_hours':            (df['minutes_per_frame'] / 60).sum(),
    }

    counts = ['n_experiments', 'n_FOVs', 'n_kymographs', 'n_unique_cells',
              'n_singleton', 'max_track_frames', 'n_rows']
    out[counts] = out[counts].astype(int)
    out['pct_singleton'] = (100 * out['n_singleton'] / out['n_unique_cells']).round(1)
    out['cell_hours'] = out['cell_hours'].round(1)
    return out[['n_experiments', 'n_FOVs', 'n_kymographs', 'n_unique_cells',
                'n_singleton', 'pct_singleton', 'median_cells_per_kymo',
                'median_track_frames', 'max_track_frames', 'n_rows', 'cell_hours']]


def dy_by_experiment(pairs):
    """Per-experiment Delta-y diagnostic: median |Delta-y| in cell lengths, and flagged %.

    Grouped by acquisition interval as well as gene and experiment, because Delta-y is
    advective and its spread scales with the frame interval -- pooling 5- and 10-minute
    runs would hide exactly the effect the cell-length normalisation exists to remove.
    """
    rows = []
    for (mins, gene, exp), s in pairs.groupby(
            ['minutes_per_frame', 'gene', 'experiment_name'], sort=False):
        rows.append({
            'min_per_frame': mins, 'gene': gene, 'experiment_name': exp,
            'n_pairs': len(s),
            'median_cell_len': round(s['cell_length'].median(), 1),
            'median_abs_dy_cl': round(s[DY_VALUE_COL].abs().median(), 3),
            'flag_pct': round(100 * s['is_jump'].mean(), 2),
        })
    return pd.DataFrame(rows).sort_values('flag_pct').reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------

def fig_cells_per_gene(df, label=None, suffix=None, save=None, out_dir=None):
    """Unique cells per gene, each bar split into one segment per experiment."""
    label = _cfg('label', label)
    suffix = _cfg('suffix', suffix)
    cells = df[CELL_KEY + ['gene']].drop_duplicates()
    per_exp = cells.groupby(['gene', 'experiment_name']).size()
    # descending: with genes on x, largest reads left-to-right
    totals = per_exp.groupby('gene').sum().sort_values(ascending=False)
    genes = list(totals.index)

    fig, ax = plt.subplots(figsize=(0.95 * len(genes) + 3, 6))
    for x, gene in enumerate(genes):
        segs = per_exp[gene].sort_values(ascending=False)
        colors = _shades(gene_color(gene), len(segs))
        bottom = 0                      # the stacking offset; `left=` when horizontal
        for (exp, n), c in zip(segs.items(), colors):
            ax.bar(x, n, bottom=bottom, color=c, edgecolor='white', linewidth=0.6)
            if n > 0.06 * totals.max():
                ax.text(x, bottom + n / 2, f'{n:,}', ha='center', va='center',
                        fontsize=7.5, color=_label_color(c))
            bottom += n
        ax.text(x, bottom + 0.012 * totals.max(),
                f'{totals[gene]:,}\n({len(segs)} exp)',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(genes)), genes)
    ax.set_ylim(0, totals.max() * 1.18)
    ax.set_ylabel('unique cells  (distinct experiment × FOV × peak_id × track_id)')
    ax.set_title(f'Unique trackastra cells per gene, split by experiment\n{label}', loc='left')
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'cells_per_gene{suffix}', out_dir=out_dir, save=save)
    return fig


def plot_dy_histogram(pairs, bounds=None, label=None, bins=60, xrange='default',
                      yscale='log', save=None, out_dir=None):
    """Delta-y / cell length histogram. `yscale` picks the log or the linear variant.

    Parameterised by y-scale, unlike ``track_displacement.plot_dy_histogram`` which
    hardcodes log -- on a linear count axis the flagged tail is a flat line, and the tail is
    the thing being looked at.
    """
    label = _cfg('label', label)
    if bounds is None:
        # The threshold is per-row and dimensionless, so it is a constant rather than
        # something to recover from the data.
        k = float(pairs['dy_threshold_cell_lengths'].min())
        bounds = (-k, k)
    lower, upper = bounds
    if xrange == 'default':
        xrange = DY_DEFAULT_RANGE
    n_jump, n_total = int(pairs['is_jump'].sum()), len(pairs)

    values = pairs[DY_VALUE_COL].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, range=xrange, color=C_MAIN, alpha=0.85)
    ax.axvline(lower, color=C_JUMP, ls='--', lw=1.5,
               label=f'cutoff [{lower:.1f}, {upper:.1f}]')
    ax.axvline(upper, color=C_JUMP, ls='--', lw=1.5)
    if xrange is not None:
        n_off = int(((values < xrange[0]) | (values > xrange[1])).sum())
        if n_off:
            # Say so rather than letting a clipped axis imply the tail does not exist.
            ax.plot([], [], ' ', label=f'{n_off} pair(s) beyond ±{xrange[1]:g}, off-axis')
    ax.set_xlabel(DY_XLABEL)
    suffix = _apply_yscale(ax, yscale, 'pair count')
    ax.set_title(f'{label}: frame-to-frame displacement\n'
                 f'{n_jump:,}/{n_total:,} pairs flagged as jumps '
                 f'({100 * n_jump / max(n_total, 1):.1f}%)')
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'dy_delta_y_hist{_cfg("suffix")}{suffix}', out_dir=out_dir, save=save)
    return fig


def plot_dfluor_histogram(all_pairs, good_pairs, label=None, bins=60, yscale='log',
                          dfluor_col=DFLUOR_COL, xlabel=None, save=None, out_dir=None):
    """Delta-fluor increase histogram, positive changes only, all pairs vs good pairs.

    `dfluor_col` defaults to the exposure-normalised column. The raw `delta_fluor` is in
    a.u. and is NOT comparable across experiments, so plotting it under the a.u./ms axis
    label would be wrong in a way nothing else would catch.
    """
    label = _cfg('label', label)
    all_pos = all_pairs.loc[all_pairs[dfluor_col] > 0, dfluor_col].dropna()
    good_pos = good_pairs.loc[good_pairs[dfluor_col] > 0, dfluor_col].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    hi = np.nanpercentile(all_pos, 99.5) if len(all_pos) else 1.0
    rng = (0, hi)
    ax.hist(all_pos, bins=bins, range=rng, color=C_ALL, alpha=0.6,
            label=f'all pairs (n={len(all_pos):,})')
    ax.hist(good_pos, bins=bins, range=rng, color=C_GOOD, alpha=0.7,
            label=f'good pairs (n={len(good_pos):,})')
    ax.set_xlabel(DFLUOR_LABEL if xlabel is None else xlabel)
    suffix = _apply_yscale(ax, yscale, 'pair count')
    ax.set_title(f'{label}: frame-to-frame fluorescence change')
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'dy_delta_fluor_hist{_cfg("suffix")}{suffix}', out_dir=out_dir, save=save)
    return fig


def plot_doubling_by_experiment(events, title, name, label=None, save=None, out_dir=None):
    """Mean doubling time per experiment, one bar per (experiment, gene).

    Splitting the mixed-gene experiments into adjacent sub-bars avoids having to paint a
    single bar with two gene colours -- and it is the honest reading, since the two genes
    sit in different FOVs of the same acquisition. Error bars are SEM over divisions.
    """
    label = _cfg('label', label)
    per = (events.groupby(['experiment_name', 'gene'], observed=True)['doubling_min']
                 .agg(mean='mean', sem='sem', n='size').reset_index())
    order = (events.groupby('experiment_name', observed=True)['doubling_min']
                   .mean().sort_values(ascending=False).index)
    pooled = events['doubling_min'].mean()
    top = float(np.nanmax(per['mean'] + per['sem'].fillna(0)))

    fig, ax = plt.subplots(figsize=(0.66 * len(order) + 3.5, 5.6))
    seen = set()
    for x, exp in enumerate(order):
        sub = per[per['experiment_name'] == exp].sort_values('gene')
        w = 0.78 / len(sub)
        for j, (_, r) in enumerate(sub.iterrows()):
            xi = x - 0.39 + w * (j + 0.5)
            c = gene_color(r['gene'])
            ax.bar(xi, r['mean'], width=w * 0.9, color=c,
                   yerr=(r['sem'] if np.isfinite(r['sem']) else None), error_kw=_ERR,
                   label=r['gene'] if r['gene'] not in seen else None)
            seen.add(r['gene'])
            ax.text(xi, r['mean'] + (r['sem'] if np.isfinite(r['sem']) else 0) + 0.03 * top,
                    f"{int(r['n'])}", ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax.axhline(pooled, color='0.35', ls='--', lw=0.9, zorder=0,
               label=f'pooled mean {pooled:.1f} min')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=60, ha='right', fontsize=7.5)
    ax.set_xlim(-0.6, len(order) - 0.4)      # matplotlib pads bar axes; the pad reads as data
    ax.set_ylim(0, top * 1.32)
    ax.set_ylabel('mean doubling time (min)')
    ax.set_title(f'{title}\n{label}   (bar labels = number of divisions)', loc='left')
    ax.legend(frameon=False, fontsize=8, ncol=3, title='gene', title_fontsize=8)
    style(ax)
    fig.tight_layout()
    save_fig(fig, name, out_dir=out_dir, save=save)
    return fig


def plot_doubling_by_gene(summary, per_exp, title, name, seed=0, label=None,
                          save=None, out_dir=None):
    """Per gene: grand mean over divisions (SEM error bars), with one dot per experiment
    coloured by acquisition interval.

    One bar per gene, not two: the mean of the per-experiment means was dropped, and the
    dots carry the replicate spread it stood in for. The y-axis is LINEAR -- the range here
    is roughly 48-86 min, not the ~150x span the fluorescence figures have to compress.
    """
    label = _cfg('label', label)
    genes = [g for g in summary.index if g != 'TOTAL']
    s = summary.loc[genes]
    x = np.arange(len(genes))

    fig, ax = plt.subplots(figsize=(1.05 * len(genes) + 4, 5.6))
    ax.bar(x, s['mean_dt_min'], width=0.55, color=C_MAIN, alpha=0.85,
           yerr=s['sem_dt_min'], error_kw=_ERR, label='grand mean (per division)')

    rng = np.random.default_rng(seed)   # deterministic jitter -> reproducible figure
    seen = set()
    for xi, gene in enumerate(genes):
        sub = per_exp[per_exp['gene'] == gene]
        for (_, r), dx in zip(sub.iterrows(), rng.uniform(-0.20, 0.20, len(sub))):
            mins = int(r['min_per_frame'])
            ax.scatter(xi + dx, r['mean'], s=26, zorder=3,
                       color=_interval_color(mins), edgecolor='0.25', linewidth=0.5,
                       label=f'{mins} min/frame' if mins not in seen else None)
            seen.add(mins)

    top = float(np.nanmax([(s['mean_dt_min'] + s['sem_dt_min'].fillna(0)).max(),
                           per_exp['mean'].max()]))
    ax.set_xticks(x, genes)
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel('doubling time (min)')
    ax.set_title(f'{title}\n{label}', loc='left')
    ax.legend(frameon=False, fontsize=8, ncol=2, title='bar / interval', title_fontsize=8)
    style(ax)
    fig.tight_layout()
    save_fig(fig, name, out_dir=out_dir, save=save)
    return fig


def plot_cell_cycles_per_trench(per_trench, seed=0, label=None, suffix=None,
                                save=None, out_dir=None):
    """Two panels on one per-trench table from `lineage_stats.cell_cycles_per_trench`.

    **Panel A is the point.** x is the divisions the tracker actually closed in a trench; y is
    that trench's whole observation record converted into cell cycles
    (`n_cell_frames / frames_per_cycle`). Every point sits above `y = x`, and the gap is the
    imaging the division count throws away -- measured, a median 1.70x per trench and 1.75x
    pooled. Trenches whose cycle length was BORROWED from their gene's median (too few full
    cycles of their own) are drawn as open markers, because their y is an inference and not a
    measurement.

    **Panel B is in MINUTES, deliberately, and that is not cosmetic.** Frames per cycle is
    12.33 at 5 min/frame against 7.84 at 10 min/frame -- a 1.6x gap that is pure acquisition
    rate, so a frames axis would read as a biological difference between experiments. Minutes
    are comparable; the frames number stays in the table, where it is exactly right because
    `minutes_per_frame` cancels out of the effective-cycle arithmetic. Points are coloured by
    interval rather than by gene so the 5/10 min quantization stays visible.
    """
    label = _cfg('label', label)
    suffix = _cfg('suffix', suffix)
    t = per_trench.copy()
    measured = t['cycle_len_source'] == 'trench'
    genes = sorted(t['gene'].dropna().unique())

    # Panel A is narrower on purpose: it is drawn at equal aspect so `y = x` really is 45
    # degrees, and y reaches ~1.8x what x does, so its box ends up tall.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.0, 5.8),
                                   gridspec_kw={'width_ratios': [1, 1.5]})

    # ---- Panel A -----------------------------------------------------------------------
    xhi = float(t['n_divisions'].max()) * 1.06
    yhi = float(t['eff_cycles'].max()) * 1.06
    hi = max(xhi, yhi)
    ratio = float((t['eff_cycles'].sum() / t['n_divisions'].sum()))
    axA.plot([0, hi], [0, hi], color='0.35', ls='--', lw=0.9, zorder=0,
             label='y = x  (no loss)')
    axA.plot([0, hi], [0, ratio * hi], color='0.55', ls=':', lw=1.2, zorder=0,
             label=f'y = {ratio:.2f}x  (pooled)')
    for gene in genes:
        for sel, face, ec, lw in ((measured, None, '0.25', 0.4),
                                  (~measured, 'none', None, 1.1)):
            sub = t[(t['gene'] == gene) & sel]
            if sub.empty:
                continue
            axA.scatter(sub['n_divisions'], sub['eff_cycles'], s=30, zorder=3,
                        color=face if face == 'none' else gene_color(gene),
                        edgecolor=ec or gene_color(gene), linewidth=lw,
                        label=gene if face != 'none' else None)
    axA.set_xlim(0, xhi)
    axA.set_ylim(0, yhi)
    axA.set_aspect('equal', adjustable='box')
    axA.set_xlabel('divisions the tracker closed')
    axA.set_ylabel('effective cell cycles  (cell-frames / frames per cycle)')
    # The open-marker key goes in the legend, not the title: at equal aspect this panel's box
    # is narrow, and a title long enough to explain it runs into the panel beside it.
    n_b = int((~measured).sum())
    handles, labels = axA.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls='none', marker='o', markerfacecolor='none',
                              markeredgecolor='0.35', markersize=6))
    labels.append(f'borrowed cycle length ({n_b})')
    axA.set_title(
        f'Effective cycles vs observed divisions — {len(t)} trenches\n'
        f'median {t["eff_per_division"].median():.2f}× per trench, '
        f'{ratio:.2f}× pooled', loc='left', fontsize=9.5)
    axA.legend(handles, labels, frameon=False, fontsize=7, ncol=2, loc='upper left')
    style(axA)

    # ---- Panel B -----------------------------------------------------------------------
    rng = np.random.default_rng(seed)       # deterministic jitter -> reproducible figure
    seen = set()
    for xi, gene in enumerate(genes):
        sub = t[t['gene'] == gene]
        for (_, r), dx in zip(sub.iterrows(), rng.uniform(-0.22, 0.22, len(sub))):
            mins = int(r['minutes_per_frame'])
            axB.scatter(xi + dx, r['min_per_cycle'], s=24, zorder=3,
                        color=_interval_color(mins), edgecolor='0.25', linewidth=0.4,
                        label=f'{mins} min/frame' if mins not in seen else None)
            seen.add(mins)
        med = float(sub['min_per_cycle'].median())
        axB.plot([xi - 0.34, xi + 0.34], [med, med], color='0.15', lw=1.6, zorder=4,
                 label='gene median' if xi == 0 else None)
        # Clear of the jittered points, which span +/-0.22.
        axB.text(xi + 0.37, med, f'{med:.0f}', ha='left', va='center', fontsize=7.5,
                 color='0.35')

    axB.set_xticks(range(len(genes)), genes)
    axB.set_xlim(-0.6, len(genes) - 0.25)
    axB.set_ylim(0, float(t['min_per_cycle'].max()) * 1.14)
    axB.set_ylabel('cell cycle length (min)')
    axB.set_title(
        f'Cell cycle length per trench — median '
        f'{t["min_per_cycle"].median():.0f} min\n'
        f'in minutes, not frames: frames per cycle is 1.6× apart between the '
        f'two intervals for no biological reason', loc='left', fontsize=9.5)
    axB.legend(frameon=False, fontsize=8, title='interval', title_fontsize=8)
    style(axB)

    fig.suptitle(label, x=0.005, ha='left', fontsize=9, color='0.35')
    fig.tight_layout()
    save_fig(fig, f'cell_cycles_per_trench{suffix}', out_dir=out_dir, save=save)
    return fig


def fluor_by_gene(df, value_col=FLUOR_COL):
    """Per gene: grand mean over cell-frames, mean of per-experiment means, and the
    per-experiment values those are built from."""
    per_exp = (df.groupby(['gene', 'experiment_name'])
                 .agg(mean=(value_col, 'mean'), n_rows=(value_col, 'size'),
                      exposure_ms=(EXPOSURE_COL, 'first'))
                 .reset_index())
    summary = pd.DataFrame({
        'n_experiments':     per_exp.groupby('gene').size(),
        'n_rows':            df.groupby('gene')[value_col].size(),
        'grand_mean':        df.groupby('gene')[value_col].mean(),
        'mean_of_exp_means': per_exp.groupby('gene')['mean'].mean(),
        'min_exp_mean':      per_exp.groupby('gene')['mean'].min(),
        'max_exp_mean':      per_exp.groupby('gene')['mean'].max(),
    })
    summary['fold_range'] = (summary['max_exp_mean'] / summary['min_exp_mean'])
    summary = summary.sort_values('grand_mean', ascending=False)   # == the plotted bar
    return summary, per_exp


def plot_fluor_bars(summary, per_exp, vlabel, title, name, seed=0, label=None,
                    save=None, out_dir=None):
    """Per-gene fluorescence, log y, with one dot per experiment coloured by exposure."""
    label = _cfg('label', label)
    expo_color = {e: _EXPOSURE_CYCLE[i % len(_EXPOSURE_CYCLE)]
                  for i, e in enumerate(sorted(per_exp['exposure_ms'].unique()))}
    genes = list(summary.index)
    x = np.arange(len(genes))
    fig, ax = plt.subplots(figsize=(1.05 * len(genes) + 4, 5.5))

    ax.bar(x, summary['grand_mean'], width=0.55, color=C_MAIN,
           alpha=0.85, label='grand mean (per cell-frame)')

    # One dot per experiment. Deterministic jitter so the figure is reproducible.
    rng = np.random.default_rng(seed)
    seen = set()
    for xi, gene in enumerate(genes):
        sub = per_exp[per_exp['gene'] == gene]
        jitter = rng.uniform(-0.20, 0.20, len(sub))
        for (_, row), dx in zip(sub.iterrows(), jitter):
            expo = row['exposure_ms']
            ax.scatter(xi + dx, row['mean'], s=26, zorder=3,
                       color=expo_color[expo], edgecolor='0.25', linewidth=0.5,
                       label=f'{expo:g} ms' if expo not in seen else None)
            seen.add(expo)

    ax.set_yscale('log')          # the log scale follows the VALUE axis
    ax.set_xticks(x, genes)       # genes on x
    ax.set_ylabel(vlabel)
    ax.set_title(f'{title}\n{label}', loc='left')
    ax.legend(frameon=False, fontsize=8, ncol=2, title='bar / exposure', title_fontsize=8)
    style(ax)
    fig.tight_layout()
    save_fig(fig, name, out_dir=out_dir, save=save)
    return fig


# ---------------------------------------------------------------------------------------
# Single-trench inspection
# ---------------------------------------------------------------------------------------

def plot_track_fluorescence(df, exp, fov, peak_id, cell_id=None, fluor_col=FLUOR_COL,
                            save=None, out_dir=None):
    """Fluorescence vs time for every track in one trench, raw and re-zeroed.

    Left panel:  raw `time_frame` on the x-axis.
    Right panel: time re-zeroed per track (frames since each track's first appearance).

    Args:
        df: per-cell-per-frame table with KYMO_KEY + track_id, time_frame and `fluor_col`.
        fov: FOV id; accepts a zero-padded str like '015' or an int.
        peak_id: trench / peak id; accepts str or int.
        cell_id: optional track_id to highlight. That lineage is drawn boldly and the rest
            dimmed to grey. The id matches the labels in the kymograph viewer.
        fluor_col: fluorescence column to plot. Defaults to a.u./ms.
    """
    sub = select_trench(df, exp, fov, peak_id)
    if sub.empty:
        print(f'WARNING: no rows for experiment={exp!r}, FOV={fov}, trench/peak={peak_id}')
        return None

    track_ids = sorted(sub['track_id'].unique())
    # tab20 by position in the sorted track list -- deliberately NOT the pinned tab10 gene
    # palette. This matches build_track_color_map() in generate_kymograph_viewer.py, so a
    # line here is the same colour as that track in the HTML viewer.
    cmap = plt.get_cmap('tab20', len(track_ids))
    track_colors = {tid: cmap(i) for i, tid in enumerate(track_ids)}

    if cell_id is not None:
        cell_id = int(cell_id)
        if cell_id not in track_ids:
            print(f'WARNING: cell_id={cell_id} not found in FOV {fov}, trench {peak_id}. '
                  f'Available track_ids: {track_ids}')
            cell_id = None

    fig, (ax_raw, ax_elapsed) = plt.subplots(1, 2, figsize=(14, 5))
    for tid in track_ids:
        g = sub[sub['track_id'] == tid].sort_values('time_frame')
        elapsed = g['time_frame'] - g['time_frame'].min()
        if cell_id is None:
            kw = dict(color=track_colors[tid], label=tid)
        elif tid == cell_id:
            kw = dict(color=track_colors[tid], label=tid, linewidth=2.5, alpha=1.0, zorder=3)
        else:
            kw = dict(color='lightgray', linewidth=1.0, alpha=0.4, zorder=1)
        ax_raw.plot(g['time_frame'], g[fluor_col], **kw)
        ax_elapsed.plot(elapsed, g[fluor_col], **kw)

    note = f' (cell {cell_id})' if cell_id is not None else ''
    ax_raw.set_xlabel('Time frame')
    ax_raw.set_title(f'{exp}\nFOV {fov}, trench {peak_id} — raw time{note}')
    ax_elapsed.set_xlabel('Frames since track start')
    ax_elapsed.set_title(f'{exp}\nFOV {fov}, trench {peak_id} — '
                         f'time elapsed per track{note}')
    for ax in (ax_raw, ax_elapsed):
        ax.set_ylabel(FLUOR_LABEL)
        style(ax)

    if cell_id is not None:
        ax_raw.legend(title='track_id', fontsize=8)
    elif len(track_ids) <= 20:
        ax_raw.legend(title='track_id', fontsize=8, ncol=2)

    fig.tight_layout()
    save_fig(fig, f'track_fluor_{exp}_{str(fov).zfill(3)}_{int(peak_id)}',
             out_dir=out_dir, save=save)
    return fig


def plot_trench_deltas(pairs, exp, fov, peak_id, panels=('dy', 'dfluor'), bins=40,
                       cell_id=None, top_n=10, yscale='log', dfluor_col=DFLUOR_COL,
                       save=None, out_dir=None):
    """Delta-y and/or Delta-fluor for one trench, drawn over its whole experiment.

    Companion to `plot_track_fluorescence` -- same (exp, fov, peak_id) arguments, but
    reading the pair table rather than the cell table.

    The grey histogram is every pair in the experiment and the trench is drawn on top, so
    you can see at a glance whether this trench moves or brightens unusually *for its own
    acquisition*. A trench is a median ~5-9 % of its experiment's pairs and its rarest bins
    hold a single pair, hence `yscale='log'` by default: on a linear axis the trench peak is
    10-14 % of the axis height and the tails are invisible. Pass `yscale='linear'` for the
    view where the bulk is legible instead of the tail.

    `panels` selects which halves to draw -- this replaces the two near-duplicate functions
    the notebook grew (a 2-panel log version and a 1-panel linear Delta-fluor one).

    UNITS. `dfluor_col` defaults to a.u./ms **per frame**: exposure-normalised, but not
    per-minute. The backdrop is a single experiment, so the 5-vs-10 min/frame difference
    never enters this figure.

    Args:
        top_n: print this many pairs with the largest fluorescence INCREASE (0 = skip).
    """
    backdrop = pairs[pairs['experiment_name'] == exp]
    if backdrop.empty:
        print(f'WARNING: no pairs for experiment={exp!r}')
        return None

    sub = select_trench(pairs, exp, fov, peak_id)
    if sub.empty:
        print(f'WARNING: no pairs for experiment={exp!r}, FOV={fov}, trench/peak={peak_id}')
        return None

    if cell_id is not None:
        cell_id = int(cell_id)
        if cell_id not in set(sub['track_id']):
            print(f'WARNING: cell_id={cell_id} has no pairs in FOV {fov}, '
                  f'trench {peak_id}. Available: {sorted(sub["track_id"].unique())}')
            cell_id = None
        else:
            sub = sub[sub['track_id'] == cell_id]

    mins = int(sub['minutes_per_frame'].iloc[0])
    note = f' | cell {cell_id}' if cell_id is not None else ''
    print(f'{exp} FOV {str(fov).zfill(3)} trench {int(peak_id)}{note}: '
          f'{len(sub)} pairs | {sub.track_id.nunique()} tracks | '
          f'median cell length {sub.cell_length.median():.1f} px  '
          f'(experiment backdrop: {len(backdrop):,} pairs)')

    spec = {
        'dy': ('delta_y', 'Δy = centroid_y(t+1) − centroid_y(t)  [px]', 'Δy'),
        'dfluor': (dfluor_col,
                   'Δfluor = fluor(t+1) − fluor(t)  '
                   '[a.u./ms per frame, bg-subtracted]', 'Δfluor'),
    }
    chosen = [spec[p] for p in panels]
    fig, axes = plt.subplots(1, len(chosen), figsize=(7 * len(chosen), 4.5), squeeze=False)

    for ax, (col, xlabel, short) in zip(axes[0], chosen):
        # Bin edges from the EXPERIMENT's robust range: a handful of extreme pairs would
        # otherwise flatten every bin that matters.
        lo, hi = np.nanpercentile(backdrop[col].dropna(), [0.5, 99.5])
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(backdrop[col].dropna(), bins=edges, color=C_ALL, alpha=0.55,
                label=f'experiment (n={len(backdrop):,})')
        ax.hist(sub[col].dropna(), bins=edges, color=C_MAIN, alpha=0.95,
                label=f'this trench (n={len(sub):,})')
        off = int(((sub[col] < lo) | (sub[col] > hi)).sum())
        if off:
            # Say so rather than letting the clipped range imply the trench has no tail.
            ax.plot([], [], ' ', label=f'{off} trench pair(s) off-axis')
        _apply_yscale(ax, yscale, 'pair count')
        ax.set_xlabel(xlabel)
        ax.set_title(f'{exp}\nFOV {str(fov).zfill(3)}, '
                     f'trench {int(peak_id)}{note} — {short}', fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        style(ax)

    fig.tight_layout()
    save_fig(fig, f'trench_deltas_{exp}_{str(fov).zfill(3)}_{int(peak_id)}'
                  f'{"" if yscale == "log" else "_linear"}', out_dir=out_dir, save=save)

    if top_n and 'dfluor' in panels:
        cols = ['track_id', 'frame_from', 'frame_to', 'time_min_from',
                'fluor_per_ms_from', 'fluor_per_ms_to', dfluor_col,
                'delta_y', DY_VALUE_COL, 'predicted_lineage']
        cols = [c for c in cols if c in sub.columns]
        top = (sub[sub[dfluor_col] > 0]
               .sort_values(dfluor_col, ascending=False).head(top_n)[cols])
        print(f'\ntop {len(top)} pairs by fluorescence INCREASE  '
              f'(a.u./ms per frame, {mins} min/frame)')
        # delta_y is kept alongside on purpose: a large step next to a large displacement
        # is usually the track moving, not the cell brightening.
        print(top.round(3).to_string(index=False))
    return fig


# ---------------------------------------------------------------------------------------
# Burst display support
#
# Detection itself is NOT here -- it lives in :mod:`mmtrack.bursts`, along with the versioned
# definition dicts. What follows is only what is needed to DRAW a burst over the trace it was
# found on, plus the baseline that sets the threshold line. The primitives below
# (`compute_baselines`, `trench_max_envelope`, `build_lineage_traces`,
# `lineage_family_roots`) are shared: `mmtrack.bursts` imports them from here rather than
# keeping its own copies.
#
# Import direction is one-way -- `bursts` imports `figures`, never the reverse -- so the
# definition dicts are reached through the module `__getattr__` at the bottom of this file.
# `figures.BURST_DEF` therefore still resolves, but there is only one dict.
# ---------------------------------------------------------------------------------------


def _burst_def(bdef=None):
    """Resolve a burst definition, defaulting to the shipped one.

    Deferred import: :mod:`mmtrack.bursts` imports this module at module level, so importing
    it back at the top here would be a cycle.
    """
    if bdef is not None:
        return bdef
    from mmtrack.bursts import BURST_DEF
    return BURST_DEF


def compute_baselines(df, fluor_col=FLUOR_COL):
    """Robust per-(gene, experiment) baseline and MAD scale for a modified z-score.

    baseline = median fluorescence; scale = 1.4826 * MAD (median absolute deviation).
    Robust because rare bursts don't inflate the median/MAD (unlike mean/std). A small
    floor keeps the scale positive for near-constant (e.g. constitutive) signals so the
    threshold stays finite.

    Grouping by (gene, experiment) is what makes the threshold invariant under the exposure
    rescale: exposure is one value per experiment, so median, MAD, height and prominence all
    scale by the same 1/E.

    Returns a DataFrame indexed by (gene, experiment_name) with columns median, mad_scale.
    """
    rows = []
    for (gene, exp), g in df.groupby(['gene', 'experiment_name']):
        x = g[fluor_col].to_numpy()
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        rows.append({'gene': gene, 'experiment_name': exp,
                     'median': med, 'mad_scale': 1.4826 * mad})
    base = pd.DataFrame(rows).set_index(['gene', 'experiment_name'])
    base['mad_scale'] = base['mad_scale'].clip(lower=1e-6)
    return base


def trench_max_envelope(df_trench, fluor_col=FLUOR_COL):
    """The brightest cell at each frame -> one continuous signal per trench.

    A burst is inherited by all descendants after a division, so at every frame the
    *maximum* fluorescence across the cells present captures the burst's envelope while
    smoothing the per-cell cell-cycle sawtooth. On this envelope a single burst that decays
    over many frames is one smooth hump, not many little peaks. Returns the picked rows
    (one per time_frame), sorted by time_frame.
    """
    idx = df_trench.groupby('time_frame')[fluor_col].idxmax()
    return df_trench.loc[idx].sort_values('time_frame').reset_index(drop=True)


def build_lineage_traces(df_trench, fluor_col=FLUOR_COL):
    """Stitch each cell onto its ancestors into continuous root->leaf fluorescence traces.

    A track_id only lasts until the cell divides (then daughters get new track_ids). Walking
    up parent_track_id and concatenating gives each root->leaf path as one continuous signal,
    so a daughter's *inherited* fluorescence appears as a declining slope rather than a fresh
    peak.

    Args:
        df_trench: rows for ONE (experiment_name, FOV, trench_id) group.

    Returns:
        list of DataFrames (one per root->leaf path), each sorted by time_frame with columns
        time_frame, fluor, node_id, track_id, predicted_lineage.
    """
    track_rows, parent_of, children = {}, {}, {}
    for tid, g in df_trench.groupby('track_id'):
        g = g.sort_values('time_frame')
        track_rows[tid] = g
        par = g['parent_track_id'].iloc[0]
        par = int(par) if pd.notna(par) else None
        parent_of[tid] = par
        if par is not None:
            children.setdefault(par, []).append(tid)

    leaves = [t for t in track_rows if t not in children]
    if not leaves:                       # degenerate (e.g. a cycle) -> treat all as leaves
        leaves = list(track_rows)

    cols = ['time_frame', fluor_col, 'node_id', 'track_id', 'predicted_lineage']
    traces = []
    for leaf in leaves:
        chain, cur, seen = [], leaf, set()
        while cur is not None and cur in track_rows and cur not in seen:
            seen.add(cur)                # guard against cyclic parent references
            chain.append(cur)
            par = parent_of.get(cur)
            cur = par if (par in track_rows) else None
        chain.reverse()                  # root -> leaf order
        trace = (pd.concat([track_rows[t][cols] for t in chain])
                   .sort_values('time_frame').reset_index(drop=True)
                   .rename(columns={fluor_col: 'fluor'}))
        traces.append(trace)
    return traces


def lineage_family_roots(df_trench):
    """{track_id: root track_id} for one trench -- connected components of the parent forest.

    A burst is inherited by every descendant of the cell that fired it, so the whole
    root->leaf tree is one candidate signal. Grouping by family instead of by trench is what
    lets two bursts running at the same time in one trench be seen separately.

    The `seen` guard mirrors build_lineage_traces: a cyclic parent reference would otherwise
    loop forever.
    """
    par = df_trench.groupby('track_id')['parent_track_id'].first().to_dict()
    out = {}
    for t in par:
        cur, seen = int(t), set()
        while True:
            p = par.get(cur)
            if pd.isna(p) or int(p) not in par or int(p) in seen:
                break
            seen.add(cur)
            cur = int(p)
        out[int(t)] = cur
    return out


def plot_bursts_for_trench(df, bursts, exp, fov, peak_id, bdef=None, by_family=True,
                           save=None, out_dir=None):
    """Overlay detected bursts on one trench -- the validation figure.

    Grey lines = per-cell stitched lineage traces. Crimson dots = the detected bursts, read
    from the burst table rather than re-detected. Dashed blue = the (gene x experiment)
    threshold. A single decaying burst gets ONE dot at its hump; a constitutive gene's
    steady-high envelope gets none.

    `by_family=True` draws one coloured envelope per lineage family, which is the signal
    detection actually runs on. Use it for concurrent bursts: with a single trench envelope
    the second burst's marker appears to float BELOW the line, because at that frame the
    trench max is following the other, brighter burst. `by_family=False` reproduces the
    older one-envelope-per-trench view.
    """
    bdef = _burst_def(bdef)
    fluor_col = bdef['fluor_col']
    df_trench = select_trench(df, exp, fov, peak_id)
    if df_trench.empty:
        print(f'WARNING: no rows for {exp} FOV {fov} trench {peak_id}')
        return None
    gene = df_trench['gene'].iloc[0]

    # Recomputed from the cell table rather than read off the burst rows: the burst table
    # only carries a threshold where a peak was found, so a quiet trench would get no line.
    baselines = compute_baselines(df, fluor_col=fluor_col)
    height = None
    if (gene, exp) in baselines.index:
        b = baselines.loc[(gene, exp)]
        height = b['median'] + bdef['k'] * b['mad_scale']

    b_tr = select_trench(bursts, exp, fov, peak_id)

    fig, ax = plt.subplots(figsize=(11, 5))
    for trace in build_lineage_traces(df_trench, fluor_col=fluor_col):
        ax.plot(trace['time_frame'], trace['fluor'],
                color='lightgray', lw=1, alpha=0.6, zorder=1)

    if by_family:
        fams = lineage_family_roots(df_trench)
        sub = df_trench.assign(_fam=[fams[int(t)] for t in df_trench['track_id']])
        # Only families that ever cross the threshold get a coloured line; the rest stay
        # grey, otherwise a busy trench is 20 indistinguishable lines.
        hot = [f for f, g in sub.groupby('_fam')
               if height is None or g[fluor_col].max() > height]
        cmap = plt.get_cmap('tab10', max(len(hot), 1))
        for i, f in enumerate(sorted(hot)):
            env = trench_max_envelope(sub[sub['_fam'] == f], fluor_col=fluor_col)
            ax.plot(env['time_frame'], env[fluor_col], lw=1.8, zorder=2,
                    color=cmap(i), label=f'family {f}')
    else:
        env = trench_max_envelope(df_trench, fluor_col=fluor_col)
        ax.plot(env['time_frame'], env[fluor_col], color='darkorange', lw=1.5, zorder=2,
                label='trench max')

    if not b_tr.empty:
        ax.scatter(b_tr['time_frame'], b_tr['fluor'], color='crimson', s=70, zorder=4,
                   label=f'burst (n={len(b_tr)})')
    if height is not None:
        ax.axhline(height, color='steelblue', ls='--', lw=1,
                   label=f'threshold = {height:.4g}')

    ax.set_xlabel('Time frame')
    ax.set_ylabel(FLUOR_LABEL)
    view = 'per-family' if by_family else 'trench max'
    ax.set_title(f'{gene}  |  {exp}\nFOV {str(fov).zfill(3)}, trench {int(peak_id)} — '
                 f'{view} & bursts  [{bdef["version"]}]')
    ax.legend(fontsize=8)
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'bursts_{bdef["version"]}_{exp}_{str(fov).zfill(3)}_{int(peak_id)}',
             out_dir=out_dir, save=save)
    return fig


# ---------------------------------------------------------------------------------------
# Reading a frame number off the viewer
# ---------------------------------------------------------------------------------------

TIME_DICT_JSON = 'data/time_dict_082026.json'


def trench_start_frame(exp, fov, trench, time_dict=None, data_root=''):
    """The `start` offset for one trench, i.e. the absolute frame its kymograph column 0 is."""
    if time_dict is None:
        import json
        with open(os.path.join(data_root, TIME_DICT_JSON)) as fh:
            time_dict = json.load(fh)
    entry = (time_dict.get(exp, {})
                      .get(str(fov).zfill(3), {})
                      .get(str(trench), {}))
    return int(entry.get('start', 0) or 0)


def viewer_frame_to_time_frame(exp, fov, trench, viewer_frame, time_dict=None, data_root=''):
    """Translate a frame number read off the kymograph viewer into a table `time_frame`.

    **These are not the same number.** The viewer labels its x-axis and its fluorescence
    charts with ABSOLUTE frames, `start + column`
    (`generate_kymograph_viewer.py` lines 260 and 523). Every table in this repo -- the cell
    table, the pair table, `bursts_082026.csv` -- stores the 0-based kymograph COLUMN in
    `time_frame`, deliberately un-offset, because the burst outlines are drawn in column
    space.

    So for a trench with `start = 37`, "frame 73" in the viewer is `time_frame` 36, and
    looking up 73 finds nothing -- or worse, silently finds the wrong frame. Trenches with
    `start = 0` are unaffected, which is what makes this easy to miss.

    Returns the `time_frame`, or raises if it lands outside the trench's recorded range.
    """
    start = trench_start_frame(exp, fov, trench, time_dict=time_dict, data_root=data_root)
    tf = int(viewer_frame) - start
    if tf < 0:
        raise ValueError(
            f'viewer frame {viewer_frame} is before this trench starts (start={start})')
    return tf


def plot_prominence_bases(df, exp, fov, peak_id, bdef=None, save=None, out_dir=None):
    """Draw WHY a candidate peak passed or failed prominence, for one trench.

    `plot_bursts_for_trench` shows the detected bursts; this shows the arithmetic behind the
    verdict, which is what you need when a trench that visibly bursts catches nothing.

    Drawn: the per-lineage-family envelopes, the height bar, every height-clearing local
    maximum, and for the strongest candidate its LEFT and RIGHT base levels. Prominence is
    the peak minus the **higher** of those two, so the higher line is the one that decides --
    and it is usually the right one (53 of 84 rejections), because a burst whose envelope has
    not returned to baseline by the end of the recording has its right base set by a plateau
    rather than by a baseline.

    The two annotated arrows are the delivered prominence and the required
    `k_prom * mad_scale`; when the first is shorter than the second, that is the whole reason
    for the rejection.
    """
    from mmtrack.bursts import burst_candidates, prom_multiplier
    bdef = _burst_def(bdef)
    fluor_col = bdef['fluor_col']
    df_trench = select_trench(df, exp, fov, peak_id)
    if df_trench.empty:
        print(f'WARNING: no rows for {exp} FOV {fov} trench {peak_id}')
        return None
    gene = df_trench['gene'].iloc[0]

    baselines = compute_baselines(df, fluor_col=fluor_col)
    b = baselines.loc[(gene, exp)]
    height = b['median'] + bdef['k'] * b['mad_scale']
    req = prom_multiplier(bdef) * b['mad_scale']

    cand = select_trench(burst_candidates(df, bdef=bdef), exp, fov, peak_id)

    fig, ax = plt.subplots(figsize=(11, 5))
    fams = lineage_family_roots(df_trench)
    sub = df_trench.assign(_fam=[fams[int(t)] for t in df_trench['track_id']])
    cmap = plt.get_cmap('tab10', max(sub['_fam'].nunique(), 1))
    for i, (f, g) in enumerate(sub.groupby('_fam')):
        env = trench_max_envelope(g, fluor_col=fluor_col)
        ax.plot(env['time_frame'], env[fluor_col], lw=1.6, color=cmap(i), zorder=2,
                label=f'family {f}')

    ax.axhline(height, color='steelblue', ls='--', lw=1,
               label=f'height bar = {height:.4g}')
    if not cand.empty:
        ax.scatter(cand['time_frame'], cand['fluor'], s=55, facecolors='none',
                   edgecolors='0.3', zorder=4, label=f'candidates (n={len(cand)})')
        best = cand.loc[cand['prom_ratio'].idxmax()]
        ax.axhline(best['left_base'], color='seagreen', ls=':', lw=1.2,
                   label=f"left base = {best['left_base']:.4g}")
        ax.axhline(best['right_base'], color='darkorange', ls=':', lw=1.2,
                   label=f"right base = {best['right_base']:.4g}")
        base = max(best['left_base'], best['right_base'])
        t0 = best['time_frame']
        ax.annotate('', xy=(t0, best['fluor']), xytext=(t0, base),
                    arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.4))
        ax.text(t0 + 0.6, (best['fluor'] + base) / 2,
                f"prominence {best['prom_s']:.2f} s", color='crimson', fontsize=8, va='center')
        ax.annotate('', xy=(t0 - 2, base + req), xytext=(t0 - 2, base),
                    arrowprops=dict(arrowstyle='<->', color='0.45', lw=1.4))
        ax.text(t0 - 2.6, base + req / 2, f'required {prom_multiplier(bdef):.2f} s',
                color='0.35', fontsize=8, va='center', ha='right')
        ax.set_title(f"{gene}  |  {exp}\nFOV {str(fov).zfill(3)}, trench {int(peak_id)} — "
                     f"strongest candidate at frame {int(t0)}: {best['verdict']} "
                     f"({best['prom_ratio']:.0%} of the bar)  [{bdef['version']}]")
    else:
        ax.set_title(f'{gene}  |  {exp}\nFOV {str(fov).zfill(3)}, trench {int(peak_id)} — '
                     f'no frame clears the height bar  [{bdef["version"]}]')

    ax.set_xlabel('Time frame')
    ax.set_ylabel(FLUOR_LABEL)
    ax.legend(fontsize=8)
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'prominence_{exp}_{str(fov).zfill(3)}_{int(peak_id)}',
             out_dir=out_dir, save=save)
    return fig


def plot_burst_coverage(cov, save=None, out_dir=None):
    """Per gene, what fraction of its trenches could ever register a burst.

    Read this next to any burst-rate bar chart. A trench whose envelope never crosses its
    (gene x experiment) height bar contributes only to the DENOMINATOR, so a low rate mixes
    "does not burst" with "could not have been seen to burst" -- and the mix is very uneven:
    87% of lacZ trenches are eligible against 25% of baeS and 0% of mazF.

    Takes the output of `mmtrack.bursts.burst_coverage`. Bars use the pinned gene colour, with
    the ineligible remainder in grey so the denominator is visible rather than implied.
    """
    t = (cov.groupby('gene')
            .agg(trenches=('eligible', 'size'), eligible=('eligible', 'sum'))
            .assign(pct=lambda d: 100 * d.eligible / d.trenches)
            .sort_values('pct', ascending=False))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(t))
    ax.bar(x, t['pct'], color=[gene_color(g) for g in t.index], zorder=3)
    ax.bar(x, 100 - t['pct'], bottom=t['pct'], color='0.88', zorder=2,
           label='cannot reach its own height bar')
    for i, (g, r) in enumerate(t.iterrows()):
        ax.text(i, r['pct'] + 2, f"{int(r['eligible'])}/{int(r['trenches'])}",
                ha='center', va='bottom', fontsize=8, color='0.25')
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, rotation=45, ha='right')
    ax.set_ylim(0, 112)
    ax.set_ylabel('% of trenches eligible')
    n_ok, n = int(cov.eligible.sum()), len(cov)
    ax.set_title(f'Trenches that can ever register a burst — {n_ok}/{n} '
                 f'({100 * n_ok / n:.0f}%) overall')
    ax.legend(fontsize=8, loc='lower left')
    style(ax)
    fig.tight_layout()
    save_fig(fig, 'burst_coverage', out_dir=out_dir, save=save)
    return fig


# ---------------------------------------------------------------------------------------
# Burst tables and rate rankings
# ---------------------------------------------------------------------------------------

def _poisson_ci(n, denom, scale):
    """Exact (Garwood) Poisson 95% CI for a rate of `n` events per `denom`, times `scale`.

    Counts here run 0-31 per gene, so the point estimates alone are badly over-readable --
    murQ's single burst gives a rate whose interval spans almost every other gene. Exact
    rather than normal-approximate because the approximation is meaningless at n=0.
    """
    from scipy.stats import chi2
    lo = np.where(n > 0, chi2.ppf(0.025, 2 * np.maximum(n, 1)) / 2, 0.0)
    hi = chi2.ppf(0.975, 2 * (n + 1)) / 2
    return [f'{scale * a / d:.2f}-{scale * b / d:.2f}' for a, b, d in zip(lo, hi, denom)]


def _poisson_bounds(n, denom, scale):
    """Numeric exact (Garwood) Poisson 95% bounds for a rate of `n` per `denom`, times `scale`.

    Numeric sibling of `_poisson_ci`, which formats to a display string and so cannot feed a
    matplotlib `yerr`. Same interval: the lower limit is 0 when n == 0 (you cannot bound a
    zero count away from zero), the upper limit uses n+1 so it stays finite there.
    """
    from scipy.stats import chi2
    n = np.asarray(n, dtype=float)
    denom = np.asarray(denom, dtype=float)
    lo = np.where(n > 0, chi2.ppf(0.025, 2 * np.maximum(n, 1)) / 2, 0.0)
    hi = chi2.ppf(0.975, 2 * (n + 1)) / 2
    return scale * lo / denom, scale * hi / denom


def _label_bars(ax, xs, vals, top, notes=None, fmt='{:.3f}', rotation=0, fontsize=7.5):
    """Write each bar's own value above it, with an optional second line beneath.

    The counterpart to the whiskers. Under `show_ci=False` the point estimate is the only
    thing left on the figure, so it has to be readable as a number rather than eyeballed off
    the axis -- and `notes` (the n/N counts) is what stops a rate built on 2 bursts from
    reading like one built on 267.

    A non-finite value is skipped rather than labelled 'nan': a gene with bursts but no
    divisions is already NaN-ed and named by `bursts_per_cell_cycle`, and printing 'nan' on
    the axis would imply the bar is there when it is not.
    """
    notes = list(notes) if notes is not None else None
    for i, (xi, v) in enumerate(zip(xs, vals)):
        if not np.isfinite(v):
            continue
        txt = fmt.format(v) if notes is None else f'{fmt.format(v)}\n{notes[i]}'
        ax.text(xi, v + 0.03 * top, txt, ha='center', va='bottom', fontsize=fontsize,
                color='0.35', rotation=rotation, linespacing=1.2)


def bursts_by_gene(bursts, cells, bdef=None):
    """Table: one row per gene, plus a recomputed TOTAL. Also returns the per-experiment
    counts the TOTAL and any per-gene dot plot are built from.

    Denominators. `per_100_cells` normalises by sampling effort in cells; `per_1k_cell_hours`
    normalises by observed TIME, using each experiment's own minutes_per_frame. The set mixes
    5 and 10 min intervals, so the time-based rate is the one that is comparable across
    experiments -- a burst has to clear threshold within one frame interval either way.

    `cells` is required, and must be scoped the same way the bursts were (see load_bursts):
    a denominator the numerator never saw gives a silently wrong rate.
    """
    bdef = _burst_def(bdef)
    uniq = cells.drop_duplicates(CELL_KEY)

    n_bursts = bursts.groupby('gene', observed=True).size()
    out = pd.DataFrame({
        'n_bursts':   n_bursts,
        'n_cells':    uniq.groupby('gene', observed=True).size(),
        'cell_hours': (cells['minutes_per_frame'] / 60).groupby(cells['gene']).sum(),
    })
    out['n_bursts'] = out['n_bursts'].fillna(0).astype(int)

    per_exp = (bursts.groupby(['gene', 'experiment_name'], observed=True).size()
                     .rename('n_bursts').reset_index())

    # TOTAL is recomputed over the whole frame rather than summed. For the count columns the
    # two agree (every cell carries exactly one gene, and no row has a null gene), but the
    # RATES must not be: TOTAL's per_100_cells is total bursts / total cells, which is not
    # the mean of the per-gene rates whenever genes differ in size -- and they differ 10x.
    out.loc['TOTAL'] = {
        'n_bursts':   len(bursts),
        'n_cells':    len(uniq),
        'cell_hours': (cells['minutes_per_frame'] / 60).sum(),
    }
    out['n_bursts'] = out['n_bursts'].astype(int)
    out['n_cells'] = out['n_cells'].astype(int)

    out['per_100_cells'] = (100 * out['n_bursts'] / out['n_cells']).round(2)
    out['per_1k_cell_hours'] = (1000 * out['n_bursts'] / out['cell_hours']).round(2)
    out['ci95_per_100_cells'] = _poisson_ci(out['n_bursts'].to_numpy(),
                                            out['n_cells'].to_numpy(), 100)
    out['constitutive'] = out.index.isin(bdef['constitutive_genes'])
    out['def_version'] = bdef['version']
    out['cell_hours'] = out['cell_hours'].round(1)

    # Sort genes by rate, TOTAL pinned last.
    body = out.drop(index='TOTAL').sort_values('per_100_cells', ascending=False)
    return pd.concat([body, out.loc[['TOTAL']]]), per_exp


def burst_trench_ranking(bursts, gene=None, top=15):
    """Biggest burst per trench, ranked -- the tool for finding more examples like one you have.

    One row per trench that bursts, keeping only that trench's strongest episode. `rise` is
    peak height above the (gene x experiment) threshold, which is the principled version of
    "largest delta fluorescence for this trench": it is already scaled by that gene's own
    noise in that experiment, so it is comparable across genes in a way raw a.u. is not.
    """
    b = bursts if gene is None else bursts[bursts['gene'] == gene]
    if b.empty:
        print(f'WARNING: no bursts for gene={gene!r}')
        return pd.DataFrame()
    b = b.assign(rise=b['fluor'] - b['threshold'])
    best = (b.sort_values('rise', ascending=False)
             .drop_duplicates(KYMO_KEY)
             .sort_values('rise', ascending=False))
    cols = ['gene', 'experiment_name', 'FOV', 'trench_id', 'track_id', 'time_frame',
            'fluor', 'threshold', 'rise']
    # 4 dp, not 2: in a.u./ms a threshold is ~0.03 and 2 dp rounds the table to mush.
    return best[cols].head(top).round(4).reset_index(drop=True)


def burst_rate_table(bursts, cells, pairs):
    """Burst counts and both normalised rates per gene, with numeric Poisson 95% bounds.

    The pair denominator is `within_track` pairs with `is_jump` False -- one consecutive-frame
    observation per cell per frame transition, tracking errors removed. Spelled out here
    rather than reusing a general "good pairs" frame, which also carries division pairs:
    quietly swapping one for the other moves every rate by ~13%.
    """
    good_wt = pairs[(~pairs['is_jump']) & (pairs['pair_type'] == 'within_track')]
    uniq = cells.drop_duplicates(CELL_KEY)

    t = pd.DataFrame({
        'n_bursts':     bursts.groupby('gene', observed=True).size(),
        'n_good_pairs': good_wt.groupby('gene', observed=True).size(),
        'n_cells':      uniq.groupby('gene', observed=True).size(),
    }).fillna(0).astype(int)

    t['per_1k_good_pairs'] = 1000 * t['n_bursts'] / t['n_good_pairs']
    t['gp_lo'], t['gp_hi'] = _poisson_bounds(t['n_bursts'], t['n_good_pairs'], 1000)
    t['per_100_cells'] = 100 * t['n_bursts'] / t['n_cells']
    t['uc_lo'], t['uc_hi'] = _poisson_bounds(t['n_bursts'], t['n_cells'], 100)
    return t


def plot_burst_rank_bars(t, value_col, lo_col, hi_col, ylabel, title, name,
                         denom_col=None, show_ci=False, label=None, save=None, out_dir=None):
    """Genes ranked by a burst rate, its value written on each bar.

    One colour for the single series -- bar length already carries the value, so hue would be
    redundant and no legend is needed. Linear y-axis: these rates span only ~13x, so a log
    axis would obscure rather than reveal.

    Sorted descending by `value_col`, so each figure is a ranking in its own right. The two
    denominators do NOT produce the same order in the middle of the field; sorting each by its
    own metric makes that visible instead of hiding it behind a shared order.

    `show_ci=True` restores the exact Poisson 95% whiskers this used to draw by default. They
    are not gone from the analysis either way -- `lo_col`/`hi_col` are still columns of `t`,
    which is where a quoted interval should come from. The value is printed at 2 dp, not the
    3 dp the per-cell-cycle plots use: these rates are order 1-35, not order 0.01.
    """
    label = _cfg('label', label)
    t = t.sort_values(value_col, ascending=False)
    genes = list(t.index)
    x = np.arange(len(genes))
    val = t[value_col].to_numpy()
    # n/N is load-bearing on either path -- a rate of 0.67 from 1/1,490 is a far weaker claim
    # than 3.40 from 35/10,305.
    notes = [f'{n}/{d:,}' for n, d in zip(t['n_bursts'], t[denom_col or 'n_good_pairs'])]

    fig, ax = plt.subplots(figsize=(1.05 * len(genes) + 4, 5.5))
    ax.bar(x, val, width=0.55, color=C_MAIN, alpha=0.85, zorder=2)

    if show_ci:
        # yerr is expressed as distance from the bar top, not as absolute bounds.
        err = np.vstack([val - t[lo_col].to_numpy(), t[hi_col].to_numpy() - val])
        # A zero-height bar still gets a whisker: 0 observed bursts is not evidence of a zero
        # rate.
        ax.errorbar(x, val, yerr=err, fmt='none', ecolor='0.35', elinewidth=1.1,
                    capsize=3, capthick=1.1, zorder=3)
        top = float(t[hi_col].max())
        for xi, (note, hi) in enumerate(zip(notes, t[hi_col])):
            ax.text(xi, hi + 0.03 * top, note, ha='center', va='bottom', fontsize=7.5,
                    color='0.35')
        ax.set_ylim(0, top * 1.16)      # headroom so the n/N labels are never clipped
        note_txt = 'bars = point estimate, labels = bursts/denominator'
    else:
        # `or 1.0`: an all-zero scope would otherwise give set_ylim(0, 0).
        top = float(np.nanmax(val)) or 1.0
        _label_bars(ax, x, val, top, notes=notes, fmt='{:.2f}')
        ax.set_ylim(0, top * 1.22)      # two-line labels need more headroom than one
        note_txt = 'value and bursts/denominator above each bar'

    ax.set_xticks(x, genes)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\n{label}  |  {note_txt}', loc='left', fontsize=10)
    style(ax)
    fig.tight_layout()
    save_fig(fig, name, out_dir=out_dir, save=save)
    return fig


# ---------------------------------------------------------------------------------------
# Comparing definitions: counts and bursts per cell cycle
# ---------------------------------------------------------------------------------------

def select_definition(bursts, version):
    """Filter a union burst table down to one definition's events.

    `data/bursts_090326.csv` holds the UNION of every definition, one row per peak found by
    any of them, with a boolean `burst_v*` column each. Nothing downstream should count its
    rows directly -- of its 196, only 118 are v5 and 181 are v5local.

    `version=None` returns the frame untouched, for a table that is already scoped.
    """
    if version is None:
        return bursts
    col = f'burst_{version}'
    if col not in bursts.columns:
        avail = sorted(c[6:] for c in bursts.columns if c.startswith('burst_'))
        raise ValueError(f'no {col!r} column; this table carries {avail}')
    return bursts[bursts[col]].copy()


def bursts_per_cell_cycle(bursts, divisions, version='v5', exclude_jumps=False, denom=None):
    """Bursts per observed cell cycle, per gene, with an exact Poisson interval.

    One division = one cell cycle that was watched from start to finish, so
    `n_bursts / n_divisions` reads as **the chance of seeing a burst per generation** -- a
    dimensionless per-cell-cycle probability, which is the form worth comparing across genes
    that differ 10x in how long they were imaged and 5-10 min in frame interval.

    Two things this rate is NOT, both worth stating before quoting it:

    * **The numerator is per EPISODE, the denominator is per DIVISION.** A burst inherited
      through a division counts once (that is the point of the max envelope), while both
      daughters' subsequent divisions each count in the denominator. So a gene whose bursts
      are long-lived relative to its cell cycle is scored lower than one whose bursts are
      brief, at equal biology.
    * **Both terms are observation-limited.** A division only enters the denominator if the
      tracker followed the cell through it, so a trench with poor tracking contributes
      bursts while contributing fewer divisions, which inflates the rate. `exclude_jumps`
      tightens the denominator to divisions whose parent had no |dy| > 1 cell-length hop;
      that is stricter but not obviously less biased, since it removes denominator without
      removing numerator. Default False, and the table reports both counts so the ratio can
      always be re-derived.

    **`denom` is the answer to that second bullet.** Pass a per-gene count of EFFECTIVE cell
    cycles -- `lineage_stats.effective_cycles_by_gene(...)['eff_cycles']` -- and the rate is
    taken against every cell-frame that was imaged, converted into cycle units, instead of
    only the divisions the tracker happened to close. Two things to know before quoting it:

    * **It is an estimate, not a count.** The exact-Poisson interval treats the denominator as
      known, so the reported bounds are too narrow by however much `frames_per_cycle` is
      uncertain. The interval is still returned, with that caveat attached.
    * **Measured, it rescales without reordering.** Per gene the effective count is
      1.68-1.86x the division count (total 1.75x), a 1.11x spread across ten genes, so every
      rate falls by about the same factor and the ranking barely moves. Do not present a
      changed absolute rate as a new biological result.

    A gene with bursts but zero divisions would divide by zero; it is reported with a NaN
    rate and named, rather than silently dropped.

    Args:
        bursts: the union burst table, or an already-scoped one with `version=None`.
        divisions: one row per division event (`data/divisions_082026.csv`, or
            `mmtrack.lineage_stats.division_events`).
        version: which definition to count -- 'v4', 'v5', 'v6', 'v5local', ... Each is a
            `burst_{version}` boolean column on the union export; `None` means the frame is
            already scoped. Counts are NOT comparable across versions as evidence of
            anything except the definition change: v6 is a strict superset of v5, so its
            higher rate is entirely the 38 events the left-base fold route admits.
        denom: optional per-gene Series of cycle counts replacing the division count.
            `exclude_jumps` is then ignored for the denominator (it is already applied inside
            the cycle-length estimate) and saying both raises.

    Returns a DataFrame indexed by gene with n_bursts, n_divisions, per_cell_cycle and the
    95% bounds, sorted descending, plus a recomputed TOTAL row. `denom_kind` records which
    denominator was used; the column keeps the name `n_divisions` either way, because five
    call sites read it and renaming it would break all of them -- read `denom_kind`, not the
    column name, to know what the number is.
    """
    b = select_definition(bursts, version)
    d = divisions[~divisions['has_jump']] if exclude_jumps else divisions

    n_b = b.groupby('gene', observed=True).size()
    if denom is None:
        n_d = d.groupby('gene', observed=True).size()
        total_d, kind = len(d), 'divisions'
    else:
        assert not exclude_jumps, (
            'exclude_jumps has no meaning alongside `denom`: the jump filter is already '
            'applied inside the cycle-length estimate. Pass one or the other.')
        n_d = pd.Series(denom).drop(index='TOTAL', errors='ignore').astype(float)
        total_d, kind = float(n_d.sum()), 'effective_cycles'

    genes = sorted(set(n_b.index) | set(n_d.index))
    out = pd.DataFrame({
        'n_bursts': n_b.reindex(genes).fillna(0).astype(int),
        'n_divisions': n_d.reindex(genes).fillna(0),
    })
    if denom is None:
        out['n_divisions'] = out['n_divisions'].astype(int)

    missing = out.index[(out.n_divisions == 0) & (out.n_bursts > 0)].tolist()
    if missing:
        print(f'WARNING: {missing} have bursts but no divisions; rate is undefined for them')

    # TOTAL recomputed, never summed -- 10 of 23 experiments hold two genes, and a ratio of
    # sums is not the sum of ratios. (The denominator itself is extensive over disjoint
    # trenches, so summing THAT is correct; it is the rate that must not be averaged.)
    out.loc['TOTAL'] = {'n_bursts': len(b), 'n_divisions': total_d}

    denom_s = out['n_divisions'].where(out['n_divisions'] > 0)
    out['per_cell_cycle'] = (out['n_bursts'] / denom_s).round(4)
    lo, hi = _poisson_bounds(out['n_bursts'].to_numpy(), denom_s.to_numpy(), 1.0)
    out['ci_lo'], out['ci_hi'] = np.round(lo, 4), np.round(hi, 4)
    out['constitutive'] = out.index.isin(_burst_def()['constitutive_genes'])
    out['def_version'] = version
    out['denom_kind'] = kind

    body = out.drop(index='TOTAL').sort_values('per_cell_cycle', ascending=False)
    return pd.concat([body, out.loc[['TOTAL']]])


def plot_burst_counts(bursts, version='v5', by_experiment=True, label=None, suffix=None,
                      save=None, out_dir=None):
    """Bar plot: how many bursts each gene has under one definition.

    Raw counts, deliberately unnormalised -- this is the "what did the detector find" figure,
    and it is the one to read beside a rate rather than instead of one, because gene sample
    sizes differ ~10x. Use `bursts_per_cell_cycle` / `burst_rate_table` for anything
    comparative.

    Constitutive genes are hatched. They are the acceptance test: a definition that scores
    hupA or mazF above zero is wrong regardless of what it does elsewhere, so the control has
    to be visible on the figure and not just in a footnote.

    `by_experiment` splits each bar into per-experiment segments, matching
    `fig_cells_per_gene`, so a gene's whole count resting on one acquisition is obvious.
    """
    label = _cfg('label', label)
    suffix = _cfg('suffix', suffix)
    b = select_definition(bursts, version)
    if b.empty:
        print(f'WARNING: no bursts for version={version!r}')
        return None

    per_exp = b.groupby(['gene', 'experiment_name'], observed=True).size()
    totals = per_exp.groupby('gene').sum().sort_values(ascending=False)
    genes = list(totals.index)
    constitutive = set(_burst_def()['constitutive_genes'])

    fig, ax = plt.subplots(figsize=(0.95 * len(genes) + 3, 5.5))
    for x, gene in enumerate(genes):
        hatch = '///' if gene in constitutive else None
        if by_experiment:
            segs = per_exp[gene].sort_values(ascending=False)
            colors = _shades(gene_color(gene), len(segs))
            bottom = 0
            for (_exp, n), c in zip(segs.items(), colors):
                ax.bar(x, n, bottom=bottom, color=c, edgecolor='white', linewidth=0.6,
                       hatch=hatch, zorder=2)
                if n > 0.08 * totals.max():
                    ax.text(x, bottom + n / 2, str(n), ha='center', va='center',
                            fontsize=7.5, color=_label_color(c))
                bottom += n
            note = f'{totals[gene]}\n({len(segs)} exp)'
        else:
            ax.bar(x, totals[gene], color=gene_color(gene), hatch=hatch, zorder=2)
            note = str(totals[gene])
        ax.text(x, totals[gene] + 0.02 * totals.max(), note,
                ha='center', va='bottom', fontsize=8)

    # Genes with zero bursts are absent from `per_exp` entirely -- name them on the axis, or
    # the figure silently implies they were never looked at.
    zero = sorted(set(GENE_COLOR) - set(genes))
    if zero:
        ax.text(0.99, 0.97, 'no bursts: ' + ', '.join(zero), transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='0.35')

    ax.set_xticks(range(len(genes)), genes)
    ax.set_ylim(0, totals.max() * 1.2)
    ax.set_ylabel('bursts detected (episodes)')
    ax.set_title(f'Bursts per gene [{version}]  \u2014  hatched = constitutive control, '
                 f'expected ~0\n{label}', loc='left')
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'burst_counts_{version}{suffix}', out_dir=out_dir, save=save)
    return fig


def plot_bursts_per_cell_cycle(bursts, divisions, version='v5', exclude_jumps=False,
                               denom=None, show_ci=False, label=None, suffix=None, save=None,
                               out_dir=None):
    """Bar plot of bursts per observed cell cycle, its value written on each bar.

    See `bursts_per_cell_cycle` for what the rate does and does not mean -- in particular
    that the numerator counts episodes and the denominator counts divisions.

    **By default each bar carries its own rate at 3 dp with `n/N` beneath it, and there are
    no whiskers.** The rate is the quantity the figure is about, and it used to appear nowhere
    on the figure -- only the counts did, so the number had to be read back off the printed
    table. 3 dp rather than 2 because the rule definitions run 0.000-0.070, where 2 dp
    collapses half the field onto 0.01.

    `show_ci=True` restores the exact Poisson (Garwood) 95% whiskers, which is the form this
    plot shipped in. They are worth looking at, because counts run 0-36 per gene and a
    zero-height bar still carries a finite upper bound -- hupA's 0 over 88 divisions admits a
    rate up to 0.042. That has not been lost by turning them off: `ci_lo`/`ci_hi` are columns
    of the table `bursts_per_cell_cycle` returns, and that is where a quoted interval should
    come from in either case.

    A zero-rate gene now labels at y=0 instead of above its whisker, which is where a zero
    belongs; `n/N` is what still marks it as 0-of-88 rather than 0-of-1,255.
    """
    label = _cfg('label', label)
    suffix = _cfg('suffix', suffix)
    t = bursts_per_cell_cycle(bursts, divisions, version=version,
                              exclude_jumps=exclude_jumps, denom=denom).drop(index='TOTAL')
    t = t[t['per_cell_cycle'].notna()]
    if t.empty:
        print(f'WARNING: nothing to plot for version={version!r}')
        return None

    # An effective-cycle denominator is fractional, so n/N is printed to 0 dp and the axis
    # says which denominator it is. Without that the figure is indistinguishable from the
    # division-count one while being ~1.75x lower.
    eff = t['denom_kind'].iloc[0] == 'effective_cycles'
    genes = list(t.index)
    x = np.arange(len(genes))
    val = t['per_cell_cycle'].to_numpy()
    notes = [f'{n}/{d:,.0f}' for n, d in zip(t['n_bursts'], t['n_divisions'])]
    constitutive = set(_burst_def()['constitutive_genes'])

    fig, ax = plt.subplots(figsize=(1.05 * len(genes) + 4, 5.5))
    ax.bar(x, val, width=0.6, zorder=2,
           color=[gene_color(g) for g in genes],
           hatch=['///' if g in constitutive else None for g in genes])

    if show_ci:
        err = np.vstack([val - t['ci_lo'].to_numpy(), t['ci_hi'].to_numpy() - val])
        ax.errorbar(x, val, yerr=err, fmt='none', ecolor='0.3', elinewidth=1.1,
                    capsize=3, capthick=1.1, zorder=3)
        top = float(t['ci_hi'].max())
        for xi, (note, hi) in enumerate(zip(notes, t['ci_hi'])):
            ax.text(xi, hi + 0.03 * top, note, ha='center', va='bottom',
                    fontsize=7.5, color='0.35')
        ax.set_ylim(0, top * 1.18)
        note_txt = 'n/N above each bar, exact Poisson 95%'
    else:
        # `or 1.0`: a scope where every gene is at zero would give set_ylim(0, 0).
        top = float(np.nanmax(val)) or 1.0
        _label_bars(ax, x, val, top, notes=notes)
        ax.set_ylim(0, top * 1.22)      # two-line labels need more headroom than one
        note_txt = 'rate and n/N above each bar'

    ax.set_xticks(x, genes)
    ax.set_ylabel('bursts per cell cycle  (episodes / '
                  + ('effective cell cycle)' if eff else 'observed division)'))
    _jump = ', no-jump divisions only' if exclude_jumps else ''
    _den = ', effective cycles' if eff else ''
    ax.set_title(f'Bursts per cell cycle [{version}{_jump}{_den}]  \u2014  {note_txt}'
                 f'\nhatched = constitutive control{label and chr(10) + label}',
                 loc='left')
    style(ax)
    _name = f'bursts_per_effective_cycle_{version}' if eff \
        else f'bursts_per_cell_cycle_{version}'
    fig.tight_layout()
    save_fig(fig, f'{_name}{suffix}', out_dir=out_dir, save=save)
    return fig


# ---------------------------------------------------------------------------------------
# Comparing definitions
# ---------------------------------------------------------------------------------------

# One colour per burst definition, for the grouped comparison plot. In that figure GENE is on
# the x-axis, so colour is free to encode the definition -- which is why these are not
# GENE_COLOR and why a tab10 collision does not matter here.
#
# Validated all-pairs with the dataviz skill's checks (ported; see BAR_LINE_COLOR): every subset
# containing v7 clears worst CVD ΔE 13.4 (target 8) and worst normal-vision ΔE 22.8 (floor 15),
# including all five at once. All-pairs rather than adjacent-pairs is the right gate here,
# because grouped bars put every definition next to every other inside one gene.
VERSION_COLOR = {
    'v4': '#2a78d6',        # blue      4.3:1 on white
    'v5': '#eb6834',        # orange    3.1:1
    'v6': '#a3003f',        # crimson   7.8:1
    'v7': '#1a237e',        # indigo   12.9:1
    'v5local': '#0b0b0b',   # ink      19.2:1
    # v9 measures a different IMAGE (the FIJI transform), not another rule on fluor_per_ms,
    # so it is deliberately far from the v4-v7 ramp: those are comparable to each other and
    # v9 is not comparable to any of them.
    'v9': '#00694e',        # deep teal 6.6:1 on white
    'v10': '#7a3b00',       # burnt umber 7.1:1 -- also off the v4-v7 ramp,
                            # and distinct from v9's teal since the two are
                            # rival rules on the SAME values.
    'v11': '#5b2d8e',       # violet    8.4:1 -- third rival on the same values,
                            # so it is kept clear of both v9's teal and v10's umber.
    'v12': '#00695f',       # deep green 6.9:1 -- NOT a rule but a trained classifier, so it
                            # is kept off the v4-v7 ramp entirely. See bursts.md section 14.
}


def version_color(version):
    """Colour for a burst definition, with a stated fallback for an unpinned one."""
    if version in VERSION_COLOR:
        return VERSION_COLOR[version]
    if version not in _unmapped:
        _unmapped[version] = plt.get_cmap('tab20')(len(_unmapped) % 20)
        print(f'NOTE: {version!r} is not in VERSION_COLOR; using a tab20 fallback. Add it to '
              'the map and re-run the palette check to pin it.')
    return _unmapped[version]


def burst_rate_comparison(bursts, divisions, versions=('v6', 'v7'), exclude_jumps=False):
    """Per-cell-cycle rate for several definitions side by side. One row per gene.

    The table behind `plot_burst_rate_comparison`, and the honest way to read a definition
    change: `bursts_per_cell_cycle` on its own gives one number per gene per definition, and
    the interesting quantity is the DIFFERENCE, which a single-version bar chart cannot show.

    Columns per version: `{v}`, `{v}_n`, `{v}_lo`, `{v}_hi`. Plus `n_divisions` (shared -- the
    denominator does not depend on the definition) and, for the last two versions given,
    `delta` and `delta_n`.

    **A higher rate under a later definition is not a better estimate of anything.** v5 ⊃ v4,
    v6 ⊃ v5 and v7 ⊃ v6 by construction, so each increment is exactly the events that
    definition's new route admits, and nothing else. The comparison is a statement about the
    rules, not about the biology.

    **The intervals overlap almost everywhere.** Counts run 0-43 per gene, so treat a
    reordering between two definitions as unresolved unless the whiskers separate -- which,
    measured on this dataset, they never do between adjacent definitions.
    """
    versions = tuple(versions)
    parts = {}
    for v in versions:
        parts[v] = bursts_per_cell_cycle(bursts, divisions, version=v,
                                         exclude_jumps=exclude_jumps).drop(index='TOTAL')
    genes = list(parts[versions[-1]].index)
    out = pd.DataFrame(index=genes)
    out.index.name = 'gene'
    for v in versions:
        t = parts[v].reindex(genes)
        out[v] = t['per_cell_cycle']
        out[f'{v}_n'] = t['n_bursts']
        out[f'{v}_lo'], out[f'{v}_hi'] = t['ci_lo'], t['ci_hi']
    out['n_divisions'] = parts[versions[-1]].reindex(genes)['n_divisions']
    if len(versions) >= 2:
        a, b = versions[-2], versions[-1]
        out['delta'] = (out[b] - out[a]).round(4)
        out['delta_n'] = out[f'{b}_n'] - out[f'{a}_n']
    out['constitutive'] = out.index.isin(_burst_def()['constitutive_genes'])
    return out.sort_values(versions[-1], ascending=False)


def plot_burst_rate_comparison(bursts, divisions, versions=('v6', 'v7'),
                               exclude_jumps=False, show_ci=False, label=None, suffix=None,
                               save=None, out_dir=None):
    """Grouped bar plot: bursts per cell cycle, one bar per definition within each gene.

    Colour encodes the DEFINITION here (`VERSION_COLOR`), not the gene, because gene is on the
    x-axis. Constitutive genes keep the hatch they carry in every other burst figure, so the
    control is identifiable without reading the tick labels.

    **By default each bar carries its own rate**, and `n/N` for the LAST version given sits
    above each group, since the denominator is shared and only the numerator moves. From three
    versions up the per-bar labels are rotated 90 degrees -- five definitions across ten genes
    will not take horizontal text, and the same trick is used for the same reason in
    `plot_doubling_by_experiment`.

    `show_ci=True` restores the exact Poisson 95% whiskers, which is how this shipped and which
    is the honest way to read a definition change: each definition is a superset of the last,
    so the bars can only go up, and whether that rise means anything is entirely a question of
    whether the intervals separate. Measured on this dataset, between adjacent definitions they
    do not -- so with the whiskers off, **do not read a taller bar as a better estimate**. The
    bounds are still `{v}_lo`/`{v}_hi` on the `burst_rate_comparison` table.
    """
    label = _cfg('label', label)
    suffix = _cfg('suffix', suffix)
    versions = tuple(versions)
    t = burst_rate_comparison(bursts, divisions, versions=versions,
                              exclude_jumps=exclude_jumps)
    t = t[t[versions[-1]].notna()]
    if t.empty:
        print(f'WARNING: nothing to plot for versions={versions!r}')
        return None

    genes = list(t.index)
    x = np.arange(len(genes))
    n_v = len(versions)
    w = 0.8 / n_v
    constitutive = set(_burst_def()['constitutive_genes'])

    fig, ax = plt.subplots(figsize=(1.35 * len(genes) + 4, 5.8))
    # Rotate the per-bar labels from three definitions up; see the docstring.
    rot = 0 if show_ci else (90 if n_v >= 3 else 0)
    if show_ci:
        top = float(max(t[f'{v}_hi'].max() for v in versions))
    else:
        # `or 1.0`: every definition at zero everywhere would give set_ylim(0, 0).
        top = float(max(np.nanmax(t[v].to_numpy(dtype=float)) for v in versions)) or 1.0

    for i, v in enumerate(versions):
        off = (i - (n_v - 1) / 2) * w
        val = t[v].to_numpy(dtype=float)
        ax.bar(x + off, val, width=w * 0.92, zorder=2, label=v,
               color=version_color(v),
               hatch=['///' if g in constitutive else None for g in genes])
        if show_ci:
            err = np.vstack([val - t[f'{v}_lo'].to_numpy(dtype=float),
                             t[f'{v}_hi'].to_numpy(dtype=float) - val])
            ax.errorbar(x + off, val, yerr=err, fmt='none', ecolor='0.3', elinewidth=1.0,
                        capsize=2, capthick=1.0, zorder=3)
        else:
            _label_bars(ax, x + off, val, top, rotation=rot,
                        fontsize=6.5 if rot else 7.5)

    # The group-level n/N goes above whatever is tallest in the group, so it clears both the
    # whiskers and the per-bar labels.
    last = versions[-1]
    head = 0.11 * top if not show_ci else 0.03 * top
    for xi, (n, d) in enumerate(zip(t[f'{last}_n'], t['n_divisions'])):
        if show_ci:
            band = max(float(t.iloc[xi][f'{v}_hi']) for v in versions)
        else:
            band = max(float(t.iloc[xi][v]) for v in versions)
        ax.text(xi, band + head, f'{int(n)}/{int(d):,}', ha='center', va='bottom',
                fontsize=7.5, color='0.35')

    ax.set_ylim(0, top * (1.20 if show_ci else (1.42 if rot else 1.30)))
    ax.set_xticks(x, genes)
    ax.set_ylabel('bursts per cell cycle  (episodes / observed division)')
    _jump = ', no-jump divisions only' if exclude_jumps else ''
    _note = ('exact Poisson 95%' if show_ci else 'rate on each bar')
    ax.set_title(
        f'Bursts per cell cycle — {" vs ".join(versions)}{_jump}  —  '
        f'n/N is {last}, {_note}\n'
        f'each definition is a superset of the last, so bars can only rise; '
        f'hatched = constitutive control{label and chr(10) + label}', loc='left')
    ax.legend(frameon=False, fontsize=8, title='definition', title_fontsize=8)
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'bursts_per_cell_cycle_cmp_{"-".join(versions)}{suffix}',
             out_dir=out_dir, save=save)
    return fig

# ---------------------------------------------------------------------------------------
# Fluorescence distributions
# ---------------------------------------------------------------------------------------

_NORM_XLABEL = {
    'none': None,        # -> FLUOR_LABEL, absolute a.u./ms
    'minmax': ('fluorescence, min–max scaled within (gene, experiment)\n'
               '[0 = dimmest, 1 = brightest cell-frame of that acquisition]'),
    'robust': ('fluorescence, p1–p99 scaled within (gene, experiment)\n'
               '[0 = 1st percentile, 1 = 99th percentile of that acquisition]'),
}


def rescale_within_group(s, how):
    """Affine rescale of ONE group onto [0, 1].

    'minmax' uses the two extreme observations, so every value lands in [0, 1] by
    construction. 'robust' uses the 1st and 99th percentiles instead, so a single bad mask
    cannot set the axis and the burst tail runs past 1.0 rather than being pinned to it.

    A group whose values are all identical has no scale; it returns NaN rather than dividing
    by zero, and the caller reports it.

    NOTE this is NOT the same quantity as the exported `*_kymo_minmax` columns. Those are
    scaled within one KYMOGRAPH (292 groups); `plot_fluor_histogram` uses this to scale
    within one (gene, experiment) (~30 groups), which is what makes a gene's replicates
    poolable. Different partition, different answer -- do not substitute one for the other.
    """
    x = s.to_numpy(dtype=float)
    if how == 'none':
        return x
    if how == 'minmax':
        lo, hi = np.nanmin(x), np.nanmax(x)
    elif how == 'robust':
        lo, hi = np.nanpercentile(x, [1, 99])
    else:
        raise ValueError(f"normalize must be 'none', 'minmax' or 'robust', got {how!r}")
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.full_like(x, np.nan)
    return (x - lo) / span


def plot_fluor_histogram(df, gene, bins=60, value_col=FLUOR_COL, xrange=None,
                         count_scale='linear', normalize='minmax', suffix=None,
                         save=None, out_dir=None):
    """Distribution of per-cell-frame fluorescence for one gene.

    One observation = one segmented cell at one frame, i.e. one row of `df`. A cell tracked
    across 40 frames contributes 40 values; there is deliberately NO per-cell deduplication.
    A burst lasts a handful of frames, so collapsing each cell to a single mean would average
    out the thing being looked for.

    `normalize` rescales each cell-frame against its own (gene, experiment) group before the
    gene's experiments are pooled -- without it, a batch offset between one gene's replicates
    reads as a bright subpopulation:

        'none'    absolute a.u./ms, no rescaling
        'minmax'  (f - min) / (max - min)    every value in [0, 1]
        'robust'  (f - p1)  / (p99 - p1)     immune to one bad mask; tail runs past 1.0

    Pass `df[~df.is_degenerate_mask]`: on a 1-pixel mask `intensity_mean_fluor` is a single
    raw pixel with no averaging, and such a mask can set the group MAXIMUM -- i.e. the whole
    x-axis under min-max scaling.

    Args:
        count_scale: 'linear' or 'log'. On linear counts a 20-cell-frame tail beside a
            3,000-cell-frame mode is a flat line -- and that is where the bursts are.
    """
    suffix = _cfg('suffix', suffix)
    sub = df[df['gene'] == gene]
    if sub.empty:
        print(f'WARNING: no rows for gene={gene!r}')
        return None

    # The gene is selected first, so grouping by experiment_name here IS grouping by
    # (gene, experiment).
    scaled = sub.groupby('experiment_name')[value_col].transform(
        lambda s: rescale_within_group(s, normalize))
    # A group with no spread scales to NaN. Name it rather than quietly plotting fewer points.
    lost = int(scaled.isna().sum() - sub[value_col].isna().sum())
    if lost:
        bad = sorted(sub.loc[scaled.isna(), 'experiment_name'].unique())
        print(f'WARNING: {gene}: {lost:,} cell-frames dropped, no spread to scale in {bad}')

    values = scaled.dropna()
    n_cells = len(sub.drop_duplicates(CELL_KEY))
    n_exp = sub['experiment_name'].nunique()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, range=xrange, color=gene_color(gene), alpha=0.85)

    if xrange is not None:
        # Say so rather than letting a clipped axis imply the tail does not exist.
        n_off = int(((values < xrange[0]) | (values > xrange[1])).sum())
        if n_off:
            ax.plot([], [], ' ', label=f'{n_off:,} cell-frame(s) off-axis')
            ax.legend(frameon=False)

    ax.set_xlabel(_NORM_XLABEL[normalize] or FLUOR_LABEL)
    ax.set_ylabel('cell-frame count' + (' (log)' if count_scale == 'log' else ''))
    ax.set_yscale(count_scale)
    # Median and max are of the PLOTTED values: on a scaled axis an a.u./ms median is
    # meaningless.
    unit = ' a.u./ms' if normalize == 'none' else ''
    scheme = 'absolute' if normalize == 'none' else f'{normalize}-scaled per (gene, experiment)'
    ax.set_title(f'{gene} — per-cell-frame fluorescence, {scheme}\n'
                 f'{len(values):,} cell-frames from {n_cells:,} unique cells, '
                 f'{n_exp} experiment(s)  |  median {values.median():.4g}, '
                 f'max {values.max():.4g}{unit}', loc='left')
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'fluor_hist_{gene}_{normalize}{suffix}', out_dir=out_dir, save=save)
    return fig


# ---------------------------------------------------------------------------------------
# The self-referenced route: one kymograph against its own distribution
# ---------------------------------------------------------------------------------------
#
# Everything below is per-cell-frame and reads no pair table. `plot_fluor_histogram` above
# asks "what does this GENE's distribution look like"; these ask "what does this TRENCH's
# distribution look like, and against which bar".
#
# Read `bursts.md` section 8 before drawing a conclusion from them. The aggregate form of the
# per-trench baseline has been measured four times: v5local (181 events vs v5's 118, coverage
# 43% -> 59%, controls intact) is evaluated and NOT shipped; unfloored per-kymograph breaks the
# mazF control at 0.19; `kymograph_global_scale` is a measured no-op; and the per-TIME-FRAME
# baseline cannot separate the target event from the controls under any window or smoothing.
# What was left open is judging the 68 genuinely local-only events by eye, which is what these
# are for.

# Reference-line colours. One per (rule, scope), all six mutually distinct -- NOT one colour
# per scope with linestyle carrying the rule, which is what this used to be and which drew
# `bars=('p95','tukey')` as four lines in two visual styles.
#
# The previous values were `C_JUMP` ('#d62728') and '#9467bd'. Measured in OKLab, those are
# ΔE **0.0** from chpS's and gfcE's histogram fills -- the same hex, not merely close. GENE_COLOR
# occupies all ten tab10 slots, so no tab10 colour is safe here; these six are off-palette.
#
# Validated with the dataviz skill's six checks, ported to Python (there is no `node` in this
# environment). Every drawable set -- each rule alone, each pair, and all three together --
# clears the hard gates: worst CVD ΔE 8.4 (target 8), worst normal-vision ΔE 18.3 (floor 15),
# and a worst ΔE of 12.0 from any of the ten gene fills.
#
# Two deliberate deviations, both because these are thin annotation lines and not categorical
# fills: ink sits below the skill's OKLCH lightness band and chroma floor by construction.
# Those checks exist so a FILL does not read as grey; for a line over a coloured fill the
# governing property is contrast, and ink is the most legible mark on the figure at 19.2:1.
BAR_LINE_COLOR = {
    ('p95', 'own'):    '#0b0b0b',   # ink      19.2:1 on white
    ('p95', 'gene'):   '#a3003f',   # crimson   7.8:1
    ('tukey', 'own'):  '#00626e',   # teal      6.9:1
    ('tukey', 'gene'): '#b06f00',   # amber     4.0:1
    ('mad', 'own'):    '#7b1fa2',   # purple    8.0:1
    ('mad', 'gene'):   '#1a237e',   # indigo   12.9:1
}

# Scope also gets a linestyle, so identity is never colour-alone (the legend names both).
BAR_LINE_STYLE = {'own': '-', 'gene': '--'}

# A white ring under every reference line, so a 1.5pt line stays legible wherever it crosses a
# saturated fill. The skill's "surface ring on overlapping marks", and what makes the hue
# distance to the fill underneath a non-issue.
_BAR_HALO = [pe.withStroke(linewidth=3.0, foreground='white')]
# Which `kymo_threshold_table` columns each `bars=` rule reads, at the (gene, experiment) and
# trench scope, and how to name it in a legend. `p95` is a quantile and flags ~5% of any trench
# by construction; `tukey` is the rule that actually discriminates; `mad` is the shipped
# detector's bar. See `kymo_threshold_table` for the measurements behind that ordering.
#
# THREE rules, not five. `p99` and `tukey_far` are columns on the table for anyone who wants
# them but are deliberately not drawable: six mutually-distinguishable line colours is the most
# OKLab space allows under the contrast and CVD gates, and a fourth rule would have to fall back
# to distinguishing lines by dash pattern alone -- which is the defect this palette exists to
# fix. Draw them by passing the column to `axvline` yourself if you need them.
_BAR_RULES = {
    'p95': ('gene_p95', 'p95', 'p95'),
    'tukey': ('gene_tukey_fence', 'tukey_fence', 'Tukey Q3+1.5·IQR'),
    'mad': ('gene_bar', 'local_bar', 'median+4·s'),
}

_SCOPED_BACKDROP_LABEL = {
    'gene_experiment': 'its (gene, experiment) — the scope the detector uses',
    'gene': 'its gene, all experiments pooled',
    'experiment': 'its experiment, all genes',
}


def select_scope(df, experiment=None, gene=None, fov=None, trench_id=None):
    """Rows for any (experiment, gene, FOV, trench_id) selection. `None` = no filter.

    The four fields of `KYMO_GROUP`, each optional, so one call covers everything from a whole
    gene down to a single kymograph. `select_trench` is the all-four case with a positional
    signature; this is the partial one.

    FOV and `trench_id` get `select_trench`'s coercions -- FOV zero-padded to 3, `trench_id`
    compared as a string -- because both read as integers but are stored as strings, and
    passing `3` for `'003'` is the difference between a plot and an empty frame.
    """
    out = df
    if experiment is not None:
        out = out[out['experiment_name'] == experiment]
    if gene is not None:
        out = out[out['gene'] == gene]
    if fov is not None:
        out = out[out['FOV'].astype(str).str.zfill(3) == str(fov).zfill(3)]
    if trench_id is not None:
        out = out[out['trench_id'].astype(str) == str(trench_id)]
    return out


def scoped_label(experiment=None, gene=None, fov=None, trench_id=None):
    """Compact tag for a `select_scope` selection, e.g. 'gfcE | 013/153'."""
    bits = []
    if gene is not None:
        bits.append(str(gene))
    if experiment is not None:
        bits.append(str(experiment))
    if fov is not None or trench_id is not None:
        f = str(fov).zfill(3) if fov is not None else '*'
        bits.append(f'{f}/{trench_id if trench_id is not None else "*"}')
    return ' | '.join(bits) if bits else 'everything'


def _quantile_stats(x):
    """Q1, Q3, IQR, p95, p99 and the two Tukey fences for one group of values.

    `np.percentile`'s default linear interpolation, so p95 generally lands BETWEEN two order
    statistics. That is why "count strictly above p95" is near but not exactly 0.05*n at small
    n -- measured over the 255 kymographs the fraction runs 0.0488 to 0.0909, median 0.0513.
    """
    q1, q3, p95, p99 = np.percentile(x, [25, 75, 95, 99])
    iqr = q3 - q1
    return {'q1': q1, 'q3': q3, 'iqr': iqr, 'p95': p95, 'p99': p99,
            'tukey_fence': q3 + 1.5 * iqr, 'tukey_far': q3 + 3.0 * iqr}


def kymo_threshold_table(df, k=4.0, scale_floor_frac=0.25, fluor_col=RAW_FLUOR_COL):
    """Every per-kymograph threshold, side by side. One row per trench.

    The table behind the self-referenced route, and the single source of truth for the lines
    `plot_fluor_histogram_scoped` draws -- so a figure and this table cannot disagree.

    Three families of threshold, all computed at BOTH the kymograph and the (gene, experiment)
    scope, and they answer different questions:

    **1. p95 / p99 -- a quantile, NOT an outlier test.** `n_above_own_p95` is ~0.05*n for every
    trench by construction, bursting or flat alike. So it cannot say "this trench has no bright
    cells" (there is always a top 5%), it cannot pass the constitutive control, and ranking by
    it ranks by trench SIZE. It is the right thing to *draw* -- "here is the top 5% of this
    trench" is what tells you which cells to open in the viewer -- and the wrong thing to count.
    The informative companion is `n_above_gene_p95`: how many of this trench's cell-frames are
    in its own (gene, experiment)'s top 5%, which for a dim trench is legitimately 0. Every
    `gene_*` column here is keyed (gene, experiment) -- 28 groups for 10 genes -- never pooled
    across a gene's replicates.

    **2. Tukey's fence, `q3 + 1.5*iqr` -- a real test.** The flagged fraction depends on the
    shape of the distribution rather than being pinned, so it discriminates. Measured per gene
    on raw fluorescence, median `frac_above_own_tukey`: lacZ 0.0569, gldA 0.0364, gfcE 0.0345,
    alkA 0.0286, araB 0.0245, **chpS 0.0236, mazF 0.0200, hupA 0.0191**, murQ 0.0181,
    baeS 0.0118 -- the three constitutive controls land 6th, 7th and 8th of 10, and lacZ top /
    baeS bottom reproduces the shipped ranking's shape (`bursts.md` section 10) with no peak
    detection, no prominence and no gene-scoped baseline.

    It also reaches both documented near-misses that NO height bar does, at either scope:
    gfcE 013/153 flags 2 frames of 300 and one of them is frame 31, the v6 peak; alkA 001/1192
    flags 4 of 275 including frame 9, the v5 peak. Not magic -- for 013/153 the fence sits at
    14.89 against a MAD bar of 16.63, i.e. simply lower on a trench whose bulk is tight -- but
    lower in a shape-aware way that keeps the control ordering.

    **3. `median + k*s` -- the shipped detector's bar.** Kept because it is the tie to
    `bursts.md`: it reproduces the documented 43% gene-bar / 59% local-bar trench coverage, and
    `local_bar` is taken from `bursts.attach_kymograph_baselines(scope='kymograph')` rather than
    re-derived, so it is the detector's own number. Its floor is not cosmetic -- unfloored, a
    trench that never does anything gets a tiny MAD and its own noise clears its bar, which
    takes mazF from 0.00 to 0.19 and breaks the control.

    **Three things a flagged count is not.** (1) It is not an event count: araB 003/1598 flags
    **44 cell-frames over 33 distinct time_frames** -- 37-66, then a 15-frame gap, then 81, 86
    and 87 (one real cell, track 50, areas 261-344 px). So a count mixes magnitude, duration AND
    the number of co-bright cells, and the flagged set is not one contiguous episode even in the
    canonical clean-burst trench. (2) `n_above_own_tukey` counts cell-frames, not frames: they
    differ in **112 of 255** trenches, by up to 24 (araB 009/551, 58 cell-frames over 34 frames).
    (3) Per *trench* the controls are not at zero (hupA 005/648 flags 4, mazF 006/327 flags 6) --
    the control holds in the per-gene ranking, not per trench.

    Comparisons are strict `>`, matching `bursts.burst_candidates`. `detect_bursts` reaches the
    same MAD bar through `scipy.find_peaks(height=...)`, which is inclusive -- a hairline
    difference that bites only on an exact tie.

    Args:
        df: per-cell-per-frame table. Pass the FULL frame, not a selection: a gene-scoped
            statistic computed from one trench's rows is not that gene's statistic. Pass
            `cells[~cells.is_degenerate_mask]`, since one 1-px mask can set a kymograph's max.
        fluor_col: defaults to RAW `intensity_mean_fluor`. Exposure normalisation is an exactly
            affine rescale within every group here, so it changes no verdict -- see
            `FLUOR_LABELS`.
        k: MAD-scale multiplier for the `median + k*s` bars. 4.0 is the shipped height bar.
        scale_floor_frac: floor the local MAD scale at this fraction of the gene's. 0 disables.

    Returns:
        One row per `KYMO_GROUP`, sorted by `frac_above_own_tukey` descending -- the worklist
        order. Deliberately NOT sorted by a p95 count, which would sort by trench size.
    """
    # Deferred: bursts imports this module at its top, so a module-level import would cycle.
    import mmtrack.bursts as _b

    glob = compute_baselines(df, fluor_col=fluor_col)
    local = _b.attach_kymograph_baselines(df, fluor_col=fluor_col, scope='kymograph',
                                          scale_floor_frac=scale_floor_frac)

    gkey = list(zip(local['gene'], local['experiment_name']))
    local['_gene_med'] = [glob['median'].get(kk, np.nan) for kk in gkey]
    local['_gene_scale'] = [glob['mad_scale'].get(kk, np.nan) for kk in gkey]
    local['_gene_bar'] = local['_gene_med'] + k * local['_gene_scale']
    local['_local_bar'] = local['_base_med'] + k * local['_base_scale']

    # Quantile statistics at both scopes. Attached per row so the counts below are one pass.
    gq = {key: _quantile_stats(sub.to_numpy(float)) for key, sub
          in df.groupby(['gene', 'experiment_name'], observed=True)[fluor_col]}
    kq = {key: _quantile_stats(sub.to_numpy(float)) for key, sub
          in df.groupby(KYMO_GROUP, observed=True)[fluor_col]}
    kkey = list(zip(*[local[c] for c in KYMO_GROUP]))
    for stat in ('q1', 'q3', 'iqr', 'p95', 'p99', 'tukey_fence', 'tukey_far'):
        local[f'_{stat}'] = [kq[kk][stat] for kk in kkey]
        local[f'_gene_{stat}'] = [gq[kk][stat] for kk in gkey]

    f = local[fluor_col]
    local['_over_gene'] = f > local['_gene_bar']
    local['_over_local'] = f > local['_local_bar']
    local['_local_only'] = local['_over_local'] & ~local['_over_gene']
    local['_over_own_p95'] = f > local['_p95']
    local['_over_gene_p95'] = f > local['_gene_p95']
    local['_over_own_tukey'] = f > local['_tukey_fence']
    local['_over_own_tukey_far'] = f > local['_tukey_far']
    local['_over_gene_tukey'] = f > local['_gene_tukey_fence']

    g = local.groupby(KYMO_GROUP, observed=True)
    agg = {
        'n_cell_frames': g.size(),
        'n_cells': g['track_id'].nunique(),
        'fluor_med': g[fluor_col].median(),
        'fluor_max': g[fluor_col].max(),
        'n_above_gene_bar': g['_over_gene'].sum(),
        'n_above_local_bar': g['_over_local'].sum(),
        'n_local_only': g['_local_only'].sum(),
        'n_above_own_p95': g['_over_own_p95'].sum(),
        'n_above_gene_p95': g['_over_gene_p95'].sum(),
        'n_above_own_tukey': g['_over_own_tukey'].sum(),
        'n_above_own_tukey_far': g['_over_own_tukey_far'].sum(),
        'n_above_gene_tukey': g['_over_gene_tukey'].sum(),
    }
    for stat in ('q1', 'q3', 'iqr', 'p95', 'p99', 'tukey_fence', 'tukey_far'):
        agg[stat] = g[f'_{stat}'].first()
        agg[f'gene_{stat}'] = g[f'_gene_{stat}'].first()
    for src_col, out_col in (('_gene_med', 'gene_med'), ('_gene_scale', 'gene_scale'),
                             ('_gene_bar', 'gene_bar'), ('_base_med', 'local_med'),
                             ('_base_scale', 'local_scale'), ('_local_bar', 'local_bar')):
        agg[out_col] = g[src_col].first()
    out = pd.DataFrame(agg).reset_index()

    out['frac_above_own_tukey'] = out['n_above_own_tukey'] / out['n_cell_frames']
    out['frac_above_own_p95'] = out['n_above_own_p95'] / out['n_cell_frames']
    # Unit-free tail scores. Safe without a denominator floor here because
    # intensity_mean_fluor is strictly positive (min 0.3462; every per-kymograph median >= 1.50)
    # -- unlike v6's fold_left, which needed a 0.5*m floor because an unfloored left_base of
    # 0.006 scored chpS 010/46 at 30x. Per gene, the MAX p99/p50 separates cleanly: all five
    # bursty genes >= 3.51, all three controls <= 2.85.
    out['p99_over_p50'] = out['p99'] / out['fluor_med']
    out['p95_over_p50'] = out['p95'] / out['fluor_med']
    out['scale_ratio'] = out['local_scale'] / out['gene_scale']
    # Matches burst_coverage's headroom_s: how far the brightest cell-frame sits above the bar,
    # in that bar's own MAD-scales. Negative = the trench can never reach it.
    out['headroom_gene_s'] = (out['fluor_max'] - out['gene_bar']) / out['gene_scale']
    out['headroom_local_s'] = (out['fluor_max'] - out['local_bar']) / out['local_scale']
    out['eligible_gene'] = out['fluor_max'] > out['gene_bar']
    out['eligible_local'] = out['fluor_max'] > out['local_bar']
    out = (out.sort_values('frac_above_own_tukey', ascending=False)
              .reset_index(drop=True))
    # Provenance, so `plot_fluor_histogram_scoped` can refuse a table built on a different
    # column or with a different k. Without it a per-ms table drawn on a raw axis renders a
    # fully plausible figure claiming 300 of 300 cell-frames above p95.
    out.attrs['thr_meta'] = {'fluor_col': fluor_col, 'k': k,
                             'scale_floor_frac': scale_floor_frac}
    return out


def _scoped_values(sel, backdrop_df, value_col, normalize, full_df=None):
    """(backdrop values, selection values, x-axis label) under one `normalize` mode.

    'none' and 'kymo_minmax' are lookups -- the second reads a column the export already
    carries.

    'minmax' and 'robust' rescale within (gene, experiment) via `rescale_within_group`, and the
    groups come from `full_df`, NOT from the backdrop. That distinction is the whole
    correctness of the axis label: grouping the backdrop meant that with the default
    `backdrop=None` the backdrop IS the selection, so a single-trench call rescaled within the
    TRENCH while the label said '(gene, experiment)'. Measured, that made it bit-identical
    (1e-16) to the exported `*_kymo_minmax` column -- the exact substitution
    `rescale_within_group`'s docstring says must never happen.
    """
    if normalize == 'none':
        return backdrop_df[value_col], sel[value_col], fluor_label(value_col)
    if normalize == 'kymo_minmax':
        col = f'{value_col}_kymo_minmax'
        if col not in sel.columns:
            raise ValueError(
                f'{col!r} is not in this frame. The annotated export carries it for '
                f'intensity_mean_fluor, fluor_per_ms, fluor_filtered and '
                f'fluor_filtered_per_ms; pass one of those as value_col.')
        return (backdrop_df[col], sel[col],
                f'{value_col}, min–max scaled within each KYMOGRAPH\n'
                '[0 = dimmest, 1 = brightest cell-frame of that trench]')
    base = backdrop_df if full_df is None else full_df
    scaled = base.groupby(['gene', 'experiment_name'], observed=True)[value_col].transform(
        lambda s: rescale_within_group(s, normalize))
    bd_s, sel_s = scaled.reindex(backdrop_df.index), scaled.reindex(sel.index)
    # A group with no spread scales to NaN. Name it rather than quietly plotting fewer points,
    # matching what plot_fluor_histogram already does.
    lost = int(sel_s.isna().sum() - sel[value_col].isna().sum())
    if lost:
        bad = sorted(sel.loc[sel_s.isna(), 'experiment_name'].unique())
        print(f'WARNING: {lost:,} of {len(sel):,} selected cell-frames dropped, no spread to '
              f'scale in {bad}')
    return bd_s, sel_s, _NORM_XLABEL[normalize]


def plot_fluor_histogram_scoped(
        df, experiment=None, gene=None, fov=None, trench_id=None,
        value_col=RAW_FLUOR_COL, bins=40, count_scale='linear', normalize='none',
        backdrop=None, layout='auto', twin_y=False,
        bars='p95', show_thresholds=True, thresholds=None,
        k=4.0, scale_floor_frac=0.25, xrange=None,
        suffix=None, save=None, out_dir=None):
    """Per-cell-frame fluorescence for one selection, over the population it is judged against.

    The distribution route (`plot_fluor_histogram`) applied at the level of one **kymograph**
    instead of one gene, which is what makes a trench comparable to *itself*.

    By default this is **one histogram of the selection alone**, on a linear count axis. The
    population view is a separate figure -- `plot_fluor_histogram` per gene -- so there is no
    reason to repeat it here.

    Both reference lines are still drawn without it, and that is the point: the
    **(gene, experiment)** p95 is a vertical mark showing where that acquisition's top-5% cut
    falls across this trench's own distribution, which is the informative comparison. A dim
    trench legitimately has 0 cell-frames past it. It is per acquisition, never pooled over a
    gene's replicates -- `compute_baselines` and the quantile groups are both keyed
    (gene, experiment), 28 groups for 10 genes -- so the legend says `gene×exp`.

    Pass `backdrop='gene_experiment'` (or `'gene'` / `'experiment'`) to draw the population
    too. Measured over all 255 trenches it is **1.0x to 227x** the size of one trench, median
    **15x** (gfcE 013/153 is 300 cell-frames against its gene-experiment's 3,663 = 12x), so on
    one count axis the selection would be a flat line beneath it. `layout` therefore defaults to
    `'stacked'` whenever a backdrop is requested: two panels, two honest count axes. `twin_y=True`
    overlays them on a second y-axis instead, which is compact but is a **dual-axis chart** -- the
    most misread chart form there is -- so it is off by default.

    **The vertical lines.** By default the 95th percentile at both scopes: the trench's own
    p95, and its (gene, experiment)'s. Every line comes from `kymo_threshold_table`, so a
    figure and that table cannot disagree.

    **A trench's own p95 is a quantile, not an outlier test.** ~5% of any trench's cell-frames
    sit above it by construction -- measured over the 255 kymographs the fraction runs 0.0488
    to 0.0909, median 0.0513, and the spread is interpolation and ties at small `n`, not signal.
    So it cannot say "this trench has no bright cells", it cannot pass the constitutive control,
    and counting above it counts `0.05*n`. What it IS good for is reading a histogram and
    picking which cells to open in the viewer, which is why it is the default line. The
    informative number in the legend is the (gene, experiment)-scoped count: how many of this
    trench's cell-frames are in its own acquisition's top 5%, which for a dim trench is
    legitimately 0.

    Pass `bars='tukey'` for the rule that actually discriminates (`Q3 + 1.5*IQR`, whose flagged
    fraction depends on shape), `bars='mad'` for the shipped detector's `median + 4*s`, or a
    tuple for several. `bars=None` draws none. See `kymo_threshold_table` for what each one
    measures and the per-gene control ranking behind the recommendation.

    **What a distribution cannot tell you at all.** Throwing time away throws away the *rise*,
    and prominence -- not height -- is what separates a burst from a constitutive gene
    (`bursts.md` section 4). Use these figures to judge candidates, never to count events; a
    contiguous flagged run is one burst plus its dilution decay, not many bursts.

    One observation is one **cell-frame**, matching `plot_fluor_histogram` -- no per-cell
    deduplication, because a burst is a handful of frames and collapsing each cell to its
    lifetime mean averages out the thing being looked for. Titles carry both counts.

    Args:
        experiment, gene, fov, trench_id: the selection; `None` means no filter on that axis.
            All four set gives one kymograph; `gene=` alone gives a pooled per-gene view.
        value_col: fluorescence column. Defaults to RAW `intensity_mean_fluor` (a.u.), because
            within a kymograph -- and within a (gene, experiment) -- the exposure division is
            an exactly affine rescale that changes no verdict. It matters only when pooling
            across experiments; see `FLUOR_LABELS` and the `backdrop` note below.
        count_scale: 'linear' by default. Note the trade-off: on linear counts a
            20-cell-frame tail beside a 3,000-cell-frame mode is nearly flat, and the tail is
            where the bursts are -- pass 'log' to read the tail, 'linear' to read the bulk.
        layout: 'overlay' (default, one panel) or 'stacked' (two panels sharing the x-axis).
        twin_y: in overlay, give the selection its own y-axis so both series stay in true
            counts and both stay visible. False shares one axis.
        normalize: 'none' (the raw absolute axis, the default here because the reference
            lines live on it), 'kymo_minmax' (the export's pre-computed per-kymograph scale),
            or 'minmax'/'robust' (rescaled within (gene, experiment), as
            `plot_fluor_histogram` does). Thresholds are drawn only for 'none'.
        backdrop: population to draw behind the selection -- None (default: the selection
            alone), 'gene_experiment' (the scope the shipped detector uses), 'gene', or
            'experiment'.
            'gene' pools a gene's experiments, which is the one combination where a RAW column
            is confounded: every gene but murQ spans 2-3 exposure settings. It warns rather
            than refusing, since the shape read is often still what you want.
        thresholds: a precomputed `kymo_threshold_table(df)`. Pass it when looping over
            trenches; otherwise it is recomputed per call.
        k, scale_floor_frac: forwarded to `kymo_threshold_table` when it computes them.

    Returns:
        The Figure, or None when the selection is empty.
    """
    suffix = _cfg('suffix', suffix)
    tag = scoped_label(experiment, gene, fov, trench_id)

    sel = select_scope(df, experiment, gene, fov, trench_id)
    if sel.empty:
        # Say which field emptied it rather than leaving the caller to bisect four arguments.
        for name, val, part in (('experiment_name', experiment, dict(experiment=experiment)),
                                ('gene', gene, dict(gene=gene)),
                                ('FOV', fov, dict(fov=fov)),
                                ('trench_id', trench_id, dict(trench_id=trench_id))):
            if val is not None and select_scope(df, **part).empty:
                print(f'WARNING: no rows anywhere with {name}={val!r} -- '
                      f'check the type and the zero padding')
                return None
        print(f'WARNING: no rows for [{tag}] -- each field exists, but not in combination')
        return None

    # The backdrop is built from the SELECTION's groups, so it is always the population the
    # selection actually sits in even when the selection spans more than one of them.
    if backdrop is None:
        bd = sel
    elif backdrop == 'gene_experiment':
        keys = set(zip(sel['gene'], sel['experiment_name']))
        bd = df[[kk in keys for kk in zip(df['gene'], df['experiment_name'])]]
    elif backdrop == 'gene':
        # The one combination where raw units are confounded: pooling a gene's experiments
        # mixes 2-3 exposure settings for every gene but murQ. Warn, do not refuse.
        if value_col in (RAW_FLUOR_COL, 'fluor_filtered') and normalize == 'none':
            n_exp = df.loc[df['gene'].isin(sel['gene'].unique()), EXPOSURE_COL].nunique()
            if n_exp > 1:
                print(f'NOTE: backdrop=\'gene\' pools {n_exp} exposure settings and '
                      f'{value_col!r} is not exposure-normalised, so the backdrop mixes '
                      f'units. Use value_col={FLUOR_COL!r} for a cross-experiment read.')
        bd = df[df['gene'].isin(sel['gene'].unique())]
    elif backdrop == 'experiment':
        bd = df[df['experiment_name'].isin(sel['experiment_name'].unique())]
    else:
        raise ValueError(f"backdrop must be 'gene_experiment', 'gene', 'experiment' or None, "
                         f'got {backdrop!r}')

    bd_vals, sel_vals, xlabel = _scoped_values(sel, bd, value_col, normalize, full_df=df)
    bd_vals, sel_vals = bd_vals.dropna(), sel_vals.dropna()
    if sel_vals.empty:
        print(f'WARNING: [{tag}] has no finite {value_col} under normalize={normalize!r}')
        return None

    # One shared bin edge array, so the two panels are actually comparable under sharex.
    lo, hi = (xrange if xrange is not None
              else (min(bd_vals.min(), sel_vals.min()), max(bd_vals.max(), sel_vals.max())))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        lo, hi = float(sel_vals.min()), float(sel_vals.min()) + 1.0
    edges = np.linspace(lo, hi, bins + 1)

    # Reference lines. Only defined on the absolute axis; the trench-scoped line only when
    # the selection resolves to exactly one kymograph, since "its own" is otherwise undefined.
    rules = (bars,) if isinstance(bars, str) else tuple(bars or ())
    for r in rules:
        if r not in _BAR_RULES:
            raise ValueError(f'bars must be drawn from {sorted(_BAR_RULES)}, got {r!r}')
    n_kymo = len(sel[KYMO_GROUP].drop_duplicates())
    lines = []          # (value, colour, linestyle, legend text)
    if show_thresholds and rules and normalize == 'none':
        if thresholds is None:
            thresholds = kymo_threshold_table(df, k=k, scale_floor_frac=scale_floor_frac,
                                              fluor_col=value_col)
        else:
            # A table built on a different column or a different k draws a wholly wrong
            # figure with no visible symptom -- a per-ms table on a raw axis reports 300 of
            # 300 cell-frames above p95. Refuse instead.
            meta = thresholds.attrs.get('thr_meta')
            if meta is None:
                print('NOTE: `thresholds` carries no provenance; cannot check it was built '
                      'on the same column and k. Rebuild it with kymo_threshold_table.')
            else:
                want = {'fluor_col': value_col, 'k': k,
                        'scale_floor_frac': scale_floor_frac}
                off = {kk: (meta[kk], vv) for kk, vv in want.items() if meta[kk] != vv}
                if off:
                    raise ValueError(
                        '`thresholds` was built with different parameters than this call: '
                        + ', '.join(f'{kk} table={a!r} call={b!r}' for kk, (a, b)
                                    in off.items())
                        + '. The bars would be in the wrong units. Rebuild the table, or '
                          'pass matching arguments.')
        rows = select_scope(thresholds, experiment, gene, fov, trench_id)
        for r in rules:
            gcol, lcol, label = _BAR_RULES[r]
            # 'gene×exp', not 'gene': `gq` in kymo_threshold_table is keyed
            # (gene, experiment_name) -- 28 groups for 10 genes -- so these bars are per
            # ACQUISITION, never pooled over a gene's replicates. Calling them the gene's
            # would contradict this module's own insistence that pooling a gene across
            # experiments mixes 2-3 exposure settings.
            if gcol in rows and len(rows[gcol].unique()) == 1:
                v = float(rows[gcol].iloc[0])
                n = int((sel_vals > v).sum())
                lines.append((v, BAR_LINE_COLOR[(r, 'gene')], BAR_LINE_STYLE['gene'],
                              f'gene×exp {label} {v:.4g} — {n} of {len(sel_vals):,} above it'))
            elif gcol in rows:
                print(f'NOTE: [{tag}] spans {len(rows[gcol].unique())} (gene, experiment) '
                      f'groups, so there is no single {label} bar for it. Add experiment= '
                      f'to draw one.')
            if n_kymo == 1 and len(rows) == 1 and lcol in rows:
                v = float(rows[lcol].iloc[0])
                n = int((sel_vals > v).sum())
                lines.append((v, BAR_LINE_COLOR[(r, 'own')], BAR_LINE_STYLE['own'],
                              f'own {label} {v:.4g} — {n} of {len(sel_vals):,} '
                              f'({100 * n / len(sel_vals):.1f}%)'))
            elif n_kymo > 1:
                print(f'NOTE: [{tag}] covers {n_kymo} kymographs, so "its own {label}" is '
                      f'undefined and only the gene×exp bar is drawn.')
    elif show_thresholds and rules:
        print(f"NOTE: reference lines are drawn only for normalize='none' -- they are in "
              f'{fluor_label(value_col)} and {normalize!r} rescales the axis away from them.')

    if layout not in ('overlay', 'stacked', 'auto'):
        raise ValueError(f"layout must be 'auto', 'overlay' or 'stacked', got {layout!r}")
    single = backdrop is None
    if layout == 'auto':
        # A backdrop is 3-227x the size of one trench (median 15x), so an overlay in true
        # counts needs two y-scales -- and a dual-axis chart is the single most misread chart
        # form there is. Two panels say the same thing with two honest axes, so that is the
        # default whenever a backdrop is asked for.
        layout = 'overlay' if single else 'stacked'
    if layout == 'stacked' and single:
        print("NOTE: layout='stacked' needs a backdrop to stack against; drawing one panel. "
              "Pass backdrop='gene_experiment' for two.")
        layout = 'overlay'
    sel_gene = sel['gene'].iloc[0] if sel['gene'].nunique() == 1 else None
    sel_colour = gene_color(sel_gene) if sel_gene else C_MAIN
    ylab = 'cell-frame count' + (' (log)' if count_scale == 'log' else '')
    bd_kymo = None if single else len(bd[KYMO_GROUP].drop_duplicates())
    bd_title = (None if single else
                f'{_SCOPED_BACKDROP_LABEL[backdrop]} — {len(bd_vals):,} cell-frames, '
                f'{bd_kymo:,} kymographs  |  median {bd_vals.median():.4g}, '
                f'max {bd_vals.max():.4g}')
    bd_legend = (None if single else
                 f'{_SCOPED_BACKDROP_LABEL[backdrop]}  ({len(bd_vals):,} cell-frames, '
                 f'median {bd_vals.median():.4g})')
    sel_legend = f'this selection  ({len(sel_vals):,} cell-frames)'

    if layout == 'stacked' and not single:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
        axes = [ax_top, ax_bot]
        ax_top.hist(bd_vals, bins=edges, color=C_ALL, alpha=0.85)
        ax_top.set_yscale(count_scale)
        ax_top.set_ylabel(ylab)
        ax_top.set_title(bd_title, loc='left', fontsize=10)
        style(ax_top)
        ax_bot.hist(sel_vals, bins=edges, color=sel_colour, alpha=0.85)
        ax_bot.set_yscale(count_scale)
        ax_bot.set_ylabel(ylab)
        legend_ax, extra = ax_bot, []
    else:
        # Overlay. The selection is 1x-227x smaller than its backdrop (median 15x), so in
        # TRUE counts on one axis it is usually a flat line under it -- worse on a linear count
        # scale, since the mode dominates. `twin_y=True` gives the selection its own
        # colour-matched y-axis; it is OFF by default because a dual-axis chart is the most
        # misread chart form there is, and `layout='stacked'` says the same thing with two
        # honest axes.
        fig, ax_bot = plt.subplots(figsize=(9, 5.5))
        axes = [ax_bot]
        extra = []
        if single:
            ax_bot.hist(sel_vals, bins=edges, color=sel_colour, alpha=0.85,
                        label=sel_legend)
            ax_bot.set_ylabel(ylab)
        else:
            ax_bot.hist(bd_vals, bins=edges, color=C_ALL, alpha=0.55, label=bd_legend)
            ax_bot.set_ylabel(f'{ylab} — backdrop')
            ax_sel = ax_bot.twinx() if twin_y else ax_bot
            # Step outline plus a light fill, so neither series hides the other.
            ax_sel.hist(sel_vals, bins=edges, color=sel_colour, alpha=0.25)
            ax_sel.hist(sel_vals, bins=edges, color=sel_colour, histtype='step', lw=2.0,
                        label=sel_legend)
            ax_sel.set_yscale(count_scale)
            if twin_y:
                ax_sel.set_ylabel(f'{ylab} — selection', color=sel_colour)
                ax_sel.tick_params(axis='y', colors=sel_colour)
                ax_bot.tick_params(axis='y', colors='0.35')
                ax_bot.yaxis.label.set_color('0.35')
                ax_sel.spines['top'].set_visible(False)
                extra = [ax_sel]
            else:
                ax_sel.set_ylabel(ylab)
        legend_ax = extra[0] if extra else ax_bot
    ax_bot.set_yscale(count_scale)
    ax_bot.set_xlabel(xlabel)

    for ax in axes:
        for v, colour, ls, _ in lines:
            ax.axvline(v, color=colour, ls=ls, lw=1.6, path_effects=_BAR_HALO)

    # The counts the whole figure exists to show. Note the own-p95 count is ~5% of n by
    # construction -- the informative one is how many land in the GENE's top slice, which for
    # a dim trench is legitimately 0.
    for _, colour, ls, text in lines:
        legend_ax.plot([], [], color=colour, ls=ls, lw=1.6, label=text)
    handles, labels = [], []
    for ax in (ax_bot, *extra):
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    if handles:
        legend_ax.legend(handles, labels, frameon=False, fontsize=8,
                         loc='upper center', bbox_to_anchor=(0.5, -0.17),
                         ncol=2 if len(handles) > 3 else 1)

    n_cells = len(sel.drop_duplicates(CELL_KEY))
    ax_bot.set_title(f'{tag} — {len(sel_vals):,} cell-frames from {n_cells:,} unique cells'
                     + (f', {n_kymo:,} kymographs' if n_kymo > 1 else '')
                     + f'  |  median {sel_vals.median():.4g}, max {sel_vals.max():.4g}',
                     loc='left', fontsize=10)
    style(ax_bot)
    fig.tight_layout()
    name = 'fluor_hist_scoped_' + (tag.replace(' | ', '_').replace('/', '-') or 'all')
    # backdrop and twin_y are in the name too: without them a `backdrop=None` figure and a
    # `backdrop='gene_experiment'` one silently overwrite each other's SVG/PNG pair.
    save_fig(fig, f'{name}_{normalize}_{"-".join(rules) or "nobars"}_{layout}'
                  f'_{count_scale}_bd-{backdrop or "none"}'
                  f'{"" if twin_y or single or layout == "stacked" else "_sharedy"}{suffix}',
             out_dir=out_dir, save=save)
    return fig


# ---------------------------------------------------------------------------------------
# Back-compatible access to the burst definitions
# ---------------------------------------------------------------------------------------

_BURST_NAMES = {'BURST_DEF', 'BURST_DEF_V1', 'BURST_DEF_V2', 'BURST_DEF_V3', 'BURST_DEF_V4',
                'BURST_DEF_V5', 'BURST_DEF_V6', 'BURST_DEF_V7', 'BURST_DEF_V8',
                'BURST_DEF_V9', 'BURST_DEF_V10', 'BURST_DEF_V11', 'BURST_DEFS'}


def __getattr__(name):
    """Forward the burst definition dicts to :mod:`mmtrack.bursts` (PEP 562).

    They used to be duplicated here. Keeping `figures.BURST_DEF` working matters because it
    is what the summary notebooks reference, but a second copy is exactly how the two drifted
    apart last time. Resolved lazily, so `bursts` can still import this module at its top.
    """
    if name in _BURST_NAMES:
        import mmtrack.bursts as _b
        return getattr(_b, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
