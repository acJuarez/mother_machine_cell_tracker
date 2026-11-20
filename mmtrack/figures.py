"""Shared style and the burst figures the analysis notebook draws.

Call `configure()` once per notebook to set the save target, the title label and the
filename suffix.
"""
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmtrack.bursts import READ_DTYPES   # noqa: F401  (re-exported for notebooks)

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
}

CONSTITUTIVE_GENES = ('hupA', 'mazF', 'chpS')

_CFG = {'out_dir': 'notebooks/figures', 'save': False, 'label': '', 'suffix': ''}


def configure(out_dir=None, save=None, label=None, suffix=None):
    """Set the module-wide figure defaults.

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
    if not _cfg('save', save):
        return None
    out_dir = _cfg('out_dir', out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f'{out_dir}/{name}.svg', transparent=True, bbox_inches='tight')
    fig.savefig(f'{out_dir}/{name}.png', dpi=200, bbox_inches='tight')
    return f'{out_dir}/{name}.svg'


_unmapped = {}


def gene_color(gene):
    """A gene's pinned colour, or a tab20 fallback that stays stable within a session."""
    if gene in GENE_COLOR:
        return GENE_COLOR[gene]
    if gene not in _unmapped:
        _unmapped[gene] = plt.get_cmap('tab20')(len(_unmapped) % 20)
        print(f'NOTE: {gene!r} is not in GENE_COLOR; using a tab20 fallback. '
              f'Add it to the map to pin its colour.')
    return _unmapped[gene]


def _shades(base, n):
    rgb = np.array(mcolors.to_rgb(base))
    factors = [0.0] if n == 1 else np.linspace(0.0, 0.6, n)
    return [tuple(rgb * (1 - f) + f) for f in factors]


def _label_color(rgb):
    r, g, b = mcolors.to_rgb(rgb)
    return 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else '0.2'


def _label_bars(ax, xs, vals, top, notes=None, fmt='{:.3f}', rotation=0, fontsize=7.5):
    notes = list(notes) if notes is not None else None
    for i, (xi, v) in enumerate(zip(xs, vals)):
        if not np.isfinite(v):
            continue
        txt = fmt.format(v) if notes is None else f'{fmt.format(v)}\n{notes[i]}'
        ax.text(xi, v + 0.03 * top, txt, ha='center', va='bottom', fontsize=fontsize,
                color='0.35', rotation=rotation, linespacing=1.2)


def _poisson_bounds(n, denom, scale):
    """Exact Poisson 95% interval for a rate of n per denom."""
    from scipy.stats import chi2
    n = np.asarray(n, dtype=float)
    denom = np.asarray(denom, dtype=float)
    lo = np.where(n > 0, chi2.ppf(0.025, 2 * np.maximum(n, 1)) / 2, 0.0)
    hi = chi2.ppf(0.975, 2 * (n + 1)) / 2
    return scale * lo / denom, scale * hi / denom


def scope_label(gene=None, experiments=None, fovs=None, min_frames=1, drop_failed=True,
                drop_excluded=True):
    """One line naming the scope a figure was drawn under, for its title."""
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
    return ' | '.join(bits)


def select_definition(bursts, version):
    """Rows of the union table flagged by one definition; `version=None` passes it through."""
    if version is None:
        return bursts
    col = f'burst_{version}'
    if col not in bursts.columns:
        raise ValueError(f'no {col!r} column; the table carries '
                         f'{sorted(c for c in bursts.columns if c.startswith("burst_"))}')
    return bursts[bursts[col].astype(bool)].copy()


def bursts_per_cell_cycle(bursts, divisions, version='v1', exclude_jumps=False, denom=None):
    """Burst rate per gene, with an exact Poisson 95% interval.

    One cycle is one observed division, so the denominator counts the divisions the tracker
    closed. Pass `denom` (a per-gene Series of cycle counts) to rate against all the imaging
    instead; `exclude_jumps` is then meaningless and saying both raises.

    Args:
        bursts: the union burst table, or an already-scoped one with `version=None`.
        divisions: one row per division event.
        version: 'v1', 'v2', 'v3' or 'manual' -- a `burst_{version}` column on the union.

    Returns a DataFrame indexed by gene, sorted descending, with a recomputed TOTAL row.
    The count column is named `n_divisions` under either denominator: read `denom_kind` to
    know which one it holds.
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

    # TOTAL is recomputed, never summed: 10 of 23 experiments hold two genes, and a ratio of
    # sums is not the sum of ratios. The denominator is extensive over disjoint trenches, so
    # summing that is correct; it is the rate that must not be averaged.
    out.loc['TOTAL'] = {'n_bursts': len(b), 'n_divisions': total_d}

    denom_s = out['n_divisions'].where(out['n_divisions'] > 0)
    out['per_cell_cycle'] = (out['n_bursts'] / denom_s).round(4)
    lo, hi = _poisson_bounds(out['n_bursts'].to_numpy(), denom_s.to_numpy(), 1.0)
    out['ci_lo'], out['ci_hi'] = np.round(lo, 4), np.round(hi, 4)
    out['constitutive'] = out.index.isin(CONSTITUTIVE_GENES)
    out['def_version'] = version
    out['denom_kind'] = kind

    body = out.drop(index='TOTAL').sort_values('per_cell_cycle', ascending=False)
    return pd.concat([body, out.loc[['TOTAL']]])


def plot_burst_counts(bursts, version='v1', by_experiment=True, unmeasured=(), label=None,
                      suffix=None, save=None, out_dir=None):
    """Bursts per gene under one definition. Constitutive genes are hatched.

    `unmeasured` names genes with no labels at all; they are dropped from the bars and called
    out on the figure, so an absence of measurement never reads as a measured zero.
    """
    label = _cfg('label', label)
    suffix = _cfg('suffix', suffix)
    unmeasured = sorted(unmeasured)
    b = select_definition(bursts, version)
    if unmeasured:
        b = b[~b['gene'].isin(unmeasured)]
    if b.empty:
        print(f'WARNING: no bursts for version={version!r}')
        return None

    per_exp = b.groupby(['gene', 'experiment_name'], observed=True).size()
    totals = per_exp.groupby('gene').sum().sort_values(ascending=False)
    genes = list(totals.index)
    constitutive = set(CONSTITUTIVE_GENES)

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

    # A gene with zero bursts is absent from `per_exp`, so name it on the figure or it reads
    # as never having been looked at.
    zero = sorted(set(GENE_COLOR) - set(genes) - set(unmeasured))
    _notes = []
    if zero:
        _notes.append('no bursts: ' + ', '.join(zero))
    if unmeasured:
        _notes.append('NOT MEASURED (no labels exist): ' + ', '.join(unmeasured))
    if _notes:
        ax.text(0.99, 0.97, '\n'.join(_notes), transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='0.35')

    ax.set_xticks(range(len(genes)), genes)
    ax.set_ylim(0, totals.max() * 1.2)
    ax.set_ylabel('bursts detected (episodes)')
    ax.set_title(f'Bursts per gene [{version}]  —  hatched = constitutive control'
                 f'\n{label}', loc='left')
    style(ax)
    fig.tight_layout()
    save_fig(fig, f'burst_counts_{version}{suffix}', out_dir=out_dir, save=save)
    return fig


def plot_bursts_per_cell_cycle(bursts, divisions, version='v1', exclude_jumps=False,
                               denom=None, show_ci=False, unmeasured=(), label=None,
                               suffix=None, save=None, out_dir=None):
    """The rate from `bursts_per_cell_cycle`, as bars labelled n/N."""
    label = _cfg('label', label)
    suffix = _cfg('suffix', suffix)
    unmeasured = sorted(unmeasured)
    t = bursts_per_cell_cycle(bursts, divisions, version=version,
                              exclude_jumps=exclude_jumps, denom=denom).drop(index='TOTAL')
    t = t[t['per_cell_cycle'].notna() & ~t.index.isin(unmeasured)]
    if t.empty:
        print(f'WARNING: nothing to plot for version={version!r}')
        return None

    eff = t['denom_kind'].iloc[0] == 'effective_cycles'
    genes = list(t.index)
    x = np.arange(len(genes))
    val = t['per_cell_cycle'].to_numpy()
    notes = [f'{n}/{d:,.0f}' for n, d in zip(t['n_bursts'], t['n_divisions'])]
    constitutive = set(CONSTITUTIVE_GENES)

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
        top = float(np.nanmax(val)) or 1.0   # `or 1.0`: an all-zero scope would set_ylim(0, 0)
        _label_bars(ax, x, val, top, notes=notes)
        ax.set_ylim(0, top * 1.22)
        note_txt = ''

    if unmeasured:
        ax.text(0.99, 0.97, 'NOT MEASURED (no labels exist): ' + ', '.join(unmeasured),
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color='0.35')

    ax.set_xticks(x, genes)
    ax.set_ylabel('bursts per cell cycle')
    _jump = ', no-jump divisions only' if exclude_jumps else ''
    _den = ', effective cycles' if eff else ''
    ax.set_title(f'Bursts per cell cycle [{version}{_jump}{_den}]  —  {note_txt}',
                 loc='left')
    style(ax)
    _name = f'bursts_per_effective_cycle_{version}' if eff \
        else f'bursts_per_cell_cycle_{version}'
    fig.tight_layout()
    save_fig(fig, f'{_name}{suffix}', out_dir=out_dir, save=save)
    return fig
