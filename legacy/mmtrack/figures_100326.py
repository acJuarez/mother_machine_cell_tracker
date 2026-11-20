"""Figure functions that only `legacy/notebooks/trackastra_figures_100326.ipynb` calls.

Per-gene fluorescence bars, the rate-vs-reference table and the constitutive-brightness test.
Needs the repo root on `sys.path`; `mmtrack.figures` supplies the shared style helpers.
"""
import numpy as np
import pandas as pd

from mmtrack.bursts import FLUOR_COL
from mmtrack.figures import (
    CONSTITUTIVE_GENES, bursts_per_cell_cycle, gene_color, save_fig, style, _cfg,
)

RAW_FLUOR_COL = 'intensity_mean_fluor'
EXPOSURE_COL = 'fluor_exposure_ms'
PIXEL_SCALE_OUTLIER = 'DUMM_giTG66_Glucose_012325'

FLUOR_LABELS = {
    RAW_FLUOR_COL: 'mean fluorescence inside the cell mask (a.u.)',
    FLUOR_COL: 'mean fluorescence per ms exposure (a.u./ms)',
    'fluor_filtered': 'mean fluorescence, median-filtered images (a.u.)',
    'fluor_filtered_per_ms': 'mean fluorescence, median-filtered, per ms exposure (a.u./ms)',
}

_EXPOSURE_CYCLE = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
C_MAIN = '#1f77b4'


def fluor_label(col):
    return FLUOR_LABELS.get(col, col)


def _apply_yscale(ax, yscale, base):
    """Set the y-scale and return the filename suffix that keeps log and linear apart."""
    ax.set_yscale(yscale)
    ax.set_ylabel(f'{base} (log)' if yscale == 'log' else base)
    return '' if yscale == 'log' else '_linear'


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


def plot_fluor_bars(summary, per_exp, vlabel, title, name, seed=0, hatch_genes=None,
                    yscale='log', gene_colors=False, label=None, save=None, out_dir=None):
    label = _cfg('label', label)
    expo_color = {e: _EXPOSURE_CYCLE[i % len(_EXPOSURE_CYCLE)]
                  for i, e in enumerate(sorted(per_exp['exposure_ms'].unique()))}
    genes = list(summary.index)
    x = np.arange(len(genes))
    fig, ax = plt.subplots(figsize=(1.05 * len(genes) + 4, 5.5))

    _hatch = set(hatch_genes or ())
    ax.bar(x, summary['grand_mean'], width=0.55,
           color=[gene_color(g) for g in genes] if gene_colors else C_MAIN,
           alpha=0.45 if gene_colors else 0.85, label='grand mean (per cell-frame)',
           hatch=['///' if g in _hatch else None for g in genes],
           edgecolor=['black' if g in _hatch else 'none' for g in genes], linewidth=0.8)

    # One dot per experiment. Deterministic jitter so the figure is reproducible.
    rng = np.random.default_rng(seed)
    seen = set()
    for xi, gene in enumerate(genes):
        sub = per_exp[per_exp['gene'] == gene]
        jitter = rng.uniform(-0.20, 0.20, len(sub))
        for (_, row), dx in zip(sub.iterrows(), jitter):
            expo = row['exposure_ms']
            ax.scatter(xi + dx, row['mean'], s=30 if gene_colors else 26, zorder=3,
                       color=expo_color[expo],
                       edgecolor='white' if gene_colors else '0.25',
                       linewidth=0.9 if gene_colors else 0.5,
                       label=f'{expo:g} ms' if expo not in seen else None)
            seen.add(expo)

    # the scale follows the VALUE axis, and names itself in the filename
    _ys = _apply_yscale(ax, yscale, vlabel)
    ax.set_xticks(x, genes)       # genes on x
    ax.set_title(f'{title}\n{label}', loc='left')
    ax.legend(frameon=False, fontsize=8, ncol=2, title='bar / exposure', title_fontsize=8)
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'{name}{_ys}', out_dir=out_dir, save=save)
    return fig


def burst_rate_vs_reference(bursts, divisions, versions=('v1', 'v2', 'v3'),
                            reference='manual', denom=None, unreviewed=()):
    unreviewed = set(unreviewed)
    parts = {v: bursts_per_cell_cycle(bursts, divisions, version=v, denom=denom)
                 .drop(index='TOTAL', errors='ignore')
             for v in (*versions, reference)}
    genes = list(parts[reference].index)
    out = pd.DataFrame(index=pd.Index(genes, name='gene'))
    for v in (*versions, reference):
        p = parts[v]
        out[v] = p['per_cell_cycle']
        out[f'{v}_n'] = p['n_bursts']
        out[f'{v}_lo'], out[f'{v}_hi'] = p['ci_lo'], p['ci_hi']
    out['n_divisions'] = parts[reference]['n_divisions']
    out['constitutive'] = parts[reference]['constitutive']
    out['reviewed'] = ~out.index.isin(unreviewed)
    # A zero reference in an unreviewed gene is an ABSENCE of measurement, not a measurement
    # of zero. Blank it so nothing downstream can divide by it or plot it as a bar.
    for c in (reference, f'{reference}_lo', f'{reference}_hi'):
        out.loc[~out['reviewed'], c] = np.nan
    out.loc[~out['reviewed'], f'{reference}_n'] = pd.NA
    for v in versions:
        out[f'{v}_x'] = (out[v] / out[reference]).round(3)
    out.attrs['denom_kind'] = parts[reference]['denom_kind'].iloc[0]
    out.attrs['reference'] = reference
    return out


def constitutive_brightness_test(df, value_col=FLUOR_COL,
                                 constitutive=CONSTITUTIVE_GENES):
    from scipy.stats import mannwhitneyu
    _, per_exp = fluor_by_gene(df, value_col=value_col)
    rows = []
    for name, gset in (('all', tuple(constitutive)),
                       ('without chpS', tuple(g for g in constitutive if g != 'chpS'))):
        a = per_exp.loc[per_exp['gene'].isin(gset), 'mean']
        b = per_exp.loc[~per_exp['gene'].isin(gset), 'mean']
        u, pv = mannwhitneyu(a, b, alternative='greater')
        rows.append({'value_col': value_col, 'set': name, 'genes': '+'.join(gset),
                     'n_const': len(a), 'n_other': len(b),
                     'median_const': round(float(a.median()), 4),
                     'median_other': round(float(b.median()), 4),
                     'fold': round(float(a.median() / b.median()), 2),
                     'U': float(u), 'p_one_sided': float(pv)})
    return pd.DataFrame(rows)
