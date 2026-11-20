# DuMM Bacteria Tracker

Cell segmentation, lineage tracking and transcription-burst analysis for Mother Machine
time-lapse microscopy. Segmentation is an mm3-style watershed, tracking is
[Trackastra](https://github.com/weigertlab/trackastra), and the analysis ends in one figures
notebook.

Package name: `mmtrack`. Python 3.11.8.

---

## 1. Setup

On the lab machine everything runs in the **`mmct` conda env**, which holds the pinned
versions. `poetry` is not installed there.

```bash
conda run -n mmct --no-capture-output python <script>
```

Run scripts from the repo root, so `import mmtrack` resolves. Stages 1–4, the FIJI step of
stage 6, and the kymograph viewer all read the raw data on `/Volumes/mcovert`.

To build the environment elsewhere, `pyproject.toml` and `poetry.lock` pin every dependency:

```bash
poetry install
```

Two interactive napari plugins are needed for stage 1: `napari-correct-drift` and
`napari-fast4dreg`.

---

## 2. The pipeline

```
raw TIFFs → 01 → 02 → 03 → 03b → 04 → 05 → 06 → 07 → figures notebook
```

Each stage is a script at the repo root. Two are interactive and two need the cluster or the
share; the rest run anywhere.

### Stage 1 — image correction

Drift-corrects the raw stacks, rotates them so the trenches are vertical, and crops out one
stack per trench.

```bash
conda run -n mmct --no-capture-output python 01_image_correction.py \
    --input-dir /path/to/experiment \
    --trench-ends-orientation down \
    --growth-channel-length 400
```

Options: `--phase-channel-idx` (0), `--fast-drift-correction` (True), `--pos-list`
(a JSON list such as `['Pos22','Pos23']`).

**Then look at the output.** Open the trench masks and the time-lapse and pick, per FOV, an
`empty_stack_id` (a trench with no cells) and the `ana_peak_ids` you want to analyse. Stage 2
needs both, and nothing can infer them.

### Stage 2 — background subtraction

Subtracts the empty trench from each analysis trench, once for phase and once for
fluorescence.

```bash
conda run -n mmct --no-capture-output python 02_background_subtraction.py \
    --input-path <stage 1 mm_channels dir> \
    --fov 007 --empty-stack-id 765 \
    --ana-peak-ids 992 1219 1749
```

Options: `--phase-index` (0), `--fluor-index` (1).

Output clips at 0, which is why the viewer's black point is a fraction of the range rather
than a percentile.

### Stage 3 — cell segmentation

Otsu threshold → binary opening → distance transform → random walker → shape and area
filtering → small-region merging. Pops a matplotlib overlay per trench for visual QC.

```bash
conda run -n mmct --no-capture-output python 03_cell_segmentation.py \
    --base-dir /path/to/experiments \
    --exp-dict '{"my_experiment": {"007": ["992", "1219"]}}'
```

Tuning: `--otsu-threshold` (0.5), `--first-opening` (4), `--distance-threshold` (3.0),
`--second-opening-size` (3), `--min-cell-area` (100), `--max-cell-area` (1000),
`--small-merge-area-threshold` (100), `--first-opening-morph` / `--second-opening-morph`
(0 or 1). Also `--start-frame`, `--end-frame`, `--phase-channel`.

Writes `mm3_segmented_*.tif` next to the subtracted phase stack. `seg_notebooks/` holds one
notebook per experiment recording the parameters that worked for it.

### Stage 3b — napari correction

Opens napari per trench with the phase stack and the auto mask drawn as a kymograph. Paint
and erase labels, close the window, and the corrected stack is written to
`napari_corrections/{FOV}_{peak}_corrected.tif`. Originals are never touched.

```bash
conda run -n mmct --no-capture-output python 03b_napari_correction.py \
    --base-dir /path/to/experiments \
    --exp-dict '{"my_experiment": {"007": ["992", "1219"]}}'
```

**This stage is mandatory.** Stage 4 reads the corrected mask, not `mm3_segmented_*.tif`, and
skips a trench that has none.

### Stage 4 — Trackastra tracking

```bash
conda run -n mmct --no-capture-output python 04_trackastra_tracking.py \
    --base-path /path/to/experiments \
    --time-range-dict '{"my_experiment":{"007":{"992":{"start":0,"end":null}}}}'
```

Writes five files per trench to `--output-dir`: `ctc_tracks.csv` (the lineage),
`ctc_masks.npy`, `imgs.npy`, and two napari display files.

On Sherlock this runs one sbatch job per trench:

```bash
python Slurm/submit_trackastra_jobs.py \
    --base-path <path> --time-dict '<json>' --batch-script Slurm/track_astra.batch
```

`Slurm/track_astra.batch` holds an absolute path to `04_trackastra_tracking.py`, so pull the
cluster checkout before submitting.

The stack is trimmed to `[start:end]` before tracking, and `end: null` means the last frame is
dropped. **Keep the same time-range dict for stage 5.**

### Stage 5 — feature extraction

Per cell-frame morphology and phase/fluorescence intensity via `regionprops_table`, plus
hierarchical lineage ids from `ctc_tracks.csv`. One `trackastra_cell_data_{exp}.pkl` per
experiment, and the kymograph TIFFs.

```bash
conda run -n mmct --no-capture-output python 05_feature_extraction.py \
    --base-path /path/to/experiments \
    --trackastra-dir <stage 4 output dir> \
    --time-range-dict '<the same JSON as stage 4>'
```

Omitting `--time-range-dict` auto-discovers the outputs but assumes `start=0`, which
misaligns fluorescence for any trench tracked from a later frame. `--plot` draws
lineage-annotated kymographs afterwards.

### Stage 6 — cell tables

Combines the pickles and derives everything the analysis reads.

```bash
conda run -n mmct --no-capture-output python 06_build_cell_tables.py \
    --base-path <dir holding trackastra_cell_data_*.pkl> \
    --generation 082026
```

Seven CSVs under `--out-dir` (default `data/`): the combined cell table, the pair table, the
two re-measured fluorescence tables, the two annotated tables, and the divisions table. The
two image-reading steps need the share; `--skip-fiji` and `--filtered-csv <cached>` let the
rest run without it, and `--cells-csv` / `--pairs-csv` skip the earlier steps.

### Stage 7 — burst detection

```bash
conda run -n mmct --no-capture-output python 07_build_bursts.py
conda run -n mmct --no-capture-output python test_bursts.py
```

Rebuilds v1, v2, v3 and the hand-label layer from four frozen inputs in under a minute and
asserts every pinned count. `test_bursts.py` re-checks the files on disk. Neither needs the
share. `bursts.md` is the full reference; read it before changing a definition or quoting a
count.

---

## 3. Tools

**`generate_kymograph_viewer.py`** renders every trench into one self-contained HTML with
napari-like layer toggles — phase, mask, fluorescence, track labels, and four nested burst
outline layers. This is how the hand labels were made. It needs the share and writes a file
of roughly 165 MB.

**`test_bursts.py`** is the burst regression contract: counts, route splits, union
set-identity, model metadata, v3's holdout numbers, and the viewer's layer invariants.

---

## 4. Repository layout

```
01..07_*.py                  the pipeline, one script per stage
generate_kymograph_viewer.py the labelling viewer
test_bursts.py               the burst regression contract
mmtrack/                     the twelve modules the pipeline uses
notebooks/analysis_notebooks/trackastra_figures_102026.ipynb   the figures
notebooks/util_notebooks/    working tools
seg_notebooks/               one segmentation notebook per experiment
Slurm/                       cluster submission for stage 4
legacy/                      archived code, see legacy/README.md
data/, models/               git-ignored; DATA.md is the local inventory
```

`CLAUDE.md` is the working reference: the module map, the data structures and every trap
worth knowing before touching a number.

---

## 5. Licensing and citation

Adapted from two reference implementations:

- napari-mm3 — Thiermann et al., *eLife* 2023
- Cell-tracker-GNN — Ben-Haim & Riklin-Raviv, *ECCV* 2022

Tracking uses Trackastra (Gallusser & Weigert).
