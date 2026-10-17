# MASS_detect

Lung nodule malignancy prediction from CT, using radiomics features. Trains on
[LNDb](https://lndb.ush.pt/) and [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
and answers, for every nodule, both a **1–5 rating** and a binary **suspicious /
not suspicious** (suspicious = rating 4 or 5).

This is a research pipeline. The narrative lives in the
notebooks (`notebooks/00`–`06`); the reusable code lives under
`src/mass_detect/scripts/`. `notebooks/00_relatorio.ipynb` is the report: problem,
data, literature and the justification of every decision.

## Setup

```bash
uv sync                      # main environment (Python 3.10)
```

The main environment is **not** compatible with `pylidc`, needed for the LIDC-IDRI analysis: version 0.2.3 needs
`pkg_resources` (`setuptools<82`) and the numpy aliases removed in 1.24. A
separate environment is provided for the LIDC path (pylidc, SimpleITK, a Jupyter
kernel):

```bash
uv venv .venv-lidc --python 3.10
uv pip install --python .venv-lidc/bin/python -r requirements_lidc.txt
```

Run LIDC / pylidc steps with `.venv-lidc/bin/python`; everything else with
`uv run`. pylidc also reads `~/.pylidcrc`, which must point at the LIDC DICOM
folder and the LIDC XML folder.

## Data layout

`scripts/config.py` is the single source of truth for every path and constant.
Paths are relative to the repository root; `data/` is **not** in version
control. The data directory is `data/` unless the `MASS_DATA_DIR` environment
variable overrides it — nothing else reads environment variables.

### You must supply

| Path | What it is | Source |
|---|---|---|
| `data/LNDb/data/data0/` … `data5/` | LNDb training CTs, `LNDb-XXXX.mhd` + `.raw` (236 scans) | `data0.rar` … `data5.rar` |
| `data/LNDb/data/testdata0/` … `testdata1/` | LNDb test CTs, `LNDb-XXXX.mhd` + `.raw` | LNDb test archives (separate download) |
| `data/LNDb/data/chars_trainNodules.csv` | Per-reading malignancy ratings (1–5). Same rows, same order as `trainNodules.csv` | `trainset_csv.zip` |
| `data/LNDb/LNDb_csvs/trainNodules_gt.csv` | Merged training findings — the unit of analysis is the nodule, not the reading | `trainset_csv.zip` |
| `data/LNDb/LNDb_csvs/trainNodules.csv` | Per-reading findings (position of every radiologist's reading) | `trainset_csv.zip` |
| `data/LNDb/testNodules.csv` | Competition test candidates (centroids only) | `trainset_csv.zip` |
| `data/LNDb/LNDb_masks/masks/LNDb-XXXX_radN.mhd` + `.raw` | Per-radiologist segmentation masks (nodules ≥ 3 mm) | LNDb archives |
| `data/TCIA_LIDC-IDRI_20200921/lidc_idri/<patient_id>/` | LIDC-IDRI DICOM series, one folder per patient | LIDC-IDRI / TCIA |
| `~/.pylidcrc` | Points pylidc at the LIDC DICOM folder and the LIDC XML folder | created locally |

Supporting CSVs used by notebook 01 for exploration and label checks —
`LNDbAcqParams.csv`, `trainCTs.csv`, `trainFolds.csv`, `trainFleischner.csv`,
`allNods.csv` — are found by name anywhere under `data/`
(`lndb_exploration_utils.find_file` searches recursively).

### Tracked intermediate tables (`csv_files/`)

These are derived but kept in version control so results are reproducible
without rerunning the exploration notebooks.

| Path | Written by |
|---|---|
| `csv_files/lndb_nodules.csv` | notebook 01 |
| `csv_files/lidc_nodules.csv` | notebook 02 |
| `csv_files/nodule_table.csv` | notebook 03 — the shared table read by notebooks 04–06 |

`predictions.csv` (repo root) is the tracked competition submission, one row per
test candidate; validate it with `validation.check_submission` before uploading.

### Produced by the pipeline (not in version control)

Generated under `data/patches_0.6mm/` (patches, features, models) and
`data/voxel_size_sensitivity/` (the voxel-size analysis). Safe to delete and
rebuild. See `scripts/config.py` for the full list of file names.

| Path | Written by |
|---|---|
| `data/patches_0.6mm/train/`, `lidc/`, `manifests/`, `manifest_train.csv`, `manifest_lidc.csv` | notebook 04 |
| `data/patches_0.6mm/rois/`, `readings.csv`, `radiomics_rois*.csv`, `radiomics_rois_icc.csv` | notebook 05 |
| `data/patches_0.6mm/feature_filter_report.csv`, `feature_sets.json`, `model_features.csv`, `reading_features.csv` | notebook 05 |
| `data/patches_0.6mm/lidc_reader_ratings.csv` | notebook 02 |
| `data/patches_0.6mm/model_results/` | notebook 06 |
| `reports/figures/patch_qc.png` | notebook 04 |
| `data/voxel_size_sensitivity/extraction/`, `results/` | `run_voxel_sensitivity.py` |

## Pipeline

The notebooks run in order, for the most part (check the note below the table); each one consumes what the previous ones wrote and
is the readable version of the corresponding modules. Unit tests run at the top
of notebooks 04 and 06.

| Notebook | Content |
|---|---|
| `00_relatorio.ipynb` | Problem, data, literature, decisions |
| `01_lndb_exploration.ipynb` | LNDb exploration → `lndb_nodules.csv` |
| `02_lidc_exploration.ipynb` | LIDC exploration → `lidc_nodules.csv`, `lidc_reader_ratings.csv` |
| `03_tabela_nodulos.ipynb` | Joined table, cohort, labels on both scales → `nodule_table.csv` |
| `04_patches.ipynb` | 3D patches and their checks |
| `05_features.ipynb` | Radiologists' regions, PyRadiomics, robustness, feature filters |
| `06_modelo.ipynb` | Models, pooling, binary answer, nested validation, interpretability |

Notebooks 03 and 04 depend on each other, so the order is not strictly linear on
a cold checkout: notebook 03 writes `csv_files/nodule_table.csv`, but its
`select_cohort` cell reads the LIDC manifest `data/patches_0.6mm/manifest_lidc.csv`
(for the `position_verified` rule), which notebook 04 produces; notebook 04's
`cut_lidc_patches` in turn reads `csv_files/nodule_table.csv`. Run notebook 03
to write the table, run notebook 04 to cut the patches and the manifests, then
rerun notebook 03's cohort cell before notebooks 05 and 06.

Steps 04–06 can also run headless (these scripts are resumable: they skip
anything already on disk, so a stopped run continues where it stopped):

```bash
# rebuild every patch, mask and radiomic feature that notebooks 04 and 05 read (~2 h)
uv run python -m src.mass_detect.scripts.build_patches_and_features

# run every comparison of notebook 06 stage by stage
OMP_NUM_THREADS=1 uv run python -m src.mass_detect.scripts.run_models

# run the voxel-size sensitivity analysis of notebook 05
uv run python -m src.mass_detect.scripts.run_voxel_sensitivity
```

Run everything from the repository root: scripts are invoked as modules and
their imports assume `src` is on the path.

### Tests

The tests are plain `test_*` functions in `src/mass_detect/tests/` . They are imported and called at the top of notebooks 04 and 06.
From the terminal:

```bash
uv run --with pytest pytest src/mass_detect/tests -q      # 34 tests
```

## Structure

```
src/mass_detect/
  scripts/
    config.py                    paths and constants shared by everything (the source of truth)
    schema.py                    the shared column schema of the nodule table
    lidc.py                      LIDC nodule table via pylidc, reader ratings
    cohort.py                    study cohort and label rules (median rating, binary scale)
    patches.py                   cut 64 mm patches around a finding (LNDb and LIDC)
    qc.py                        patch quality checks
    rois.py                      radiologists' regions, their PyRadiomics features, reader ICC
    radiomics_features.py        PyRadiomics extraction settings; ICC robustness
    feature_tables.py            feature filters → the tables the models read
    features.py                  segmentation from the centre + handcrafted patch features
    sensitivity.py               cube-size and voxel-size analysis
    models.py                    ordinal logistic regression, size model, malignancy model, ComBat
    modelling.py                 models, pooling, nested cross validation, external validation
    validation.py                competition metrics and grouped cross validation
    build_patches_and_features.py, run_models.py, run_voxel_sensitivity.py
                                 headless runners for notebooks 04–06
    lndb_exploration_utils.py    helpers used by notebook 01
    legacy/                      legacy standalone scripts, superseded by the notebooks
  lndb_utils/                    utilities shipped with the LNDb dataset (vendored)
  tests/                         unit tests
notebooks/                       the report and the pipeline (00–06)
csv_files/                       tracked intermediate tables
info/                            notes, dataset guides, helpers
reports/                         figures and the submission explanation
```

### Nodule identifiers

One scheme everywhere: `nodule_uid` is `<dataset>-<image>_<finding>` for LNDb
(`LNDb-0001_7`) and the LIDC nodule name otherwise. The `patch_id` used by the
CSV tables and file names is `LNDb-XXXX_fNNN` for LNDb (`cohort.lndb_patch_id`)
and the `nodule_uid` unchanged for LIDC. It is derived from the `.mhd`
filename, so it propagates automatically from the preprocessing through feature
extraction into every CSV; the notebooks and models join on it. LNDb malignancy
is joined **by row position**, because `RadID`/`FindingID` mean different things
across the LNDb CSVs.

## Notes

- The label is `floor(median(reader ratings) + 0.5)`, rounded half up so that a
  tie between two readers goes in the cautious direction; the mean and the mode
  are kept only to compare with the median. A nodule is suspicious (binary
  scale) when the rounded rating is 4 or 5.
- PyRadiomics settings live in `scripts/radiomics_features.py`.
- Older scripts (`lndb_preprocessing.py`, `extract_features_pyrad.py`,
  `lidc_preprocessing.py`) belong to a previous pipeline and are kept for
  reference: they (probably) don't work
