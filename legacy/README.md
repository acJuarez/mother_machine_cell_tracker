# legacy/

Code that is no longer on the path from raw images to the figures in
`notebooks/analysis_notebooks/trackastra_figures_102026.ipynb`.

Kept for provenance. **Nothing here is imported by live code.** The modules under
`legacy/mmtrack/` are not part of the `mmtrack` package any more, so run everything from the
repo root.

All four scripts here still resolve their imports, because `legacy/mmtrack/` is not a package
and each one loads what it needs by path. Run them from the repo root.
`build_v4.py` and `compare_raters.py` graft their archived half (`mmtrack/bursts_v4.py`,
`mmtrack/second_rater.py`) back onto `mmtrack.bursts`, so the rest of each script reads as it
did when it was live.

Archived on 2026-11-20, off branch `extraction`.

## The GNN tracking route

Superseded by Trackastra (stages 4 and 5).

| file | what it was |
|---|---|
| `05_lineage_tracking.py` | predicted parent–daughter links with a trained GNN |
| `mmtrack/bh_track.py` | the `LineageLinkPredictionGNN` model, training and evaluation |
| `mmtrack/lineage_detection.py` | turned edge predictions into a node-level lineage table |
| `download_assets.py` | fetched the GNN checkpoint from HuggingFace |
| `notebooks/older_notebooks/20250815_GNN_train_cell_lineage.ipynb` | the training notebook |

## Superseded pipeline stages

| file | superseded by |
|---|---|
| `04_feature_extraction.py` | `05_feature_extraction.py` (the Trackastra homolog) |
| `07_track_displacement_analysis.py` | the compute half now lives in `mmtrack/cell_tables.py`; only the Δy figures are here |
| `08_build_combined_data.py` | folded verbatim into `mmtrack/cell_tables.py` as step 6a |
| `build_time_dict.py` | a one-off that built `data/time_dict_082026.json` from a curated sheet |
| `mmtrack/cell_segmentation.py` | `mmtrack/mm_cell_segmentation.py`, which added the opening-morphology options |
| `mmtrack/segmentation_correction.py` | `mmtrack/napari_annotator.py`, which added the `run_annotator` entry point |
| `mmtrack/run_trackastra_broken.py` | `mmtrack/run_trackastra.py`. This draft never ran: it passes `required=True` to positional `add_argument` calls, which raises before parsing, and it has `import pandas as np`. It also never wrote `ctc_tracks.csv`, which stage 5 requires. |
| `mmtrack/trackastra_extraction.py` | an exploratory script with hardcoded single-trench paths; no importers |
| `mmtrack/verify_mask.py`, `mmtrack/growth_curves.py` | no importers in the live pipeline |

## Analysis the figures notebook does not use

| file | what it was |
|---|---|
| `mmtrack/track_displacement.py` | the Δy / Δfluorescence histograms and their driver. The pair builders, `flag_jumps`, `compute_filtered_fluor` and `compute_fiji_fluor` moved to `mmtrack/cell_tables.py` |
| `mmtrack/lineage_stats.py` | doubling times, and the effective-cycle denominator (`cell_cycles_per_trench`, `effective_cycles_by_gene`). `division_events` and `track_lifetimes` moved to `mmtrack/cell_tables.py` |
| `mmtrack/figures_100326.py` | `fluor_by_gene`, `plot_fluor_bars`, `burst_rate_vs_reference`, `constitutive_brightness_test` — the figures only the 100326 notebook drew |
| `mmtrack/bursts_v4.py` + `build_v4.py` | **v4**: v2's 43 features refitted on the 110926 labels with v3's split discipline. Built `data/bursts_v4_110926.csv` and `models/burst_v4_110926.joblib`. Not a viewer layer any more and not in `bursts.EXPECTED` |
| `compare_raters.py` + `mmtrack/second_rater.py` | the inter-rater study against the second labeller (`data/subgen_kymograph_heena.csv`). The counts agree; the timing does not, because one marks the peak and the other the rise. `second_rater.md` holds the findings, moved out of `bursts.md` |

## Notebooks

`notebooks/trackastra_figures_080926.ipynb`, `_082026.ipynb` and `_100326.ipynb` are the
earlier generations of the figures notebook. Section 11 of the 082026 one is where
`all_data_annotated_082026.csv` and `track_pairs_annotated_082026.csv` were first built; that
code is now step 6d of `mmtrack/cell_tables.py` and reproduces both files exactly.

`notebooks/older_notebooks/` and the loose notebooks beside it are earlier exploratory work.
