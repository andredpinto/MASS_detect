# MASS_detect

Lung nodule malignancy prediction from CT, using radiomics features extracted
with PyRadiomics. Trains on [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
and [LNDb](https://lndb.ush.pt/) and predicts a 1–5 malignancy rating for every
LNDb test candidate.

## Data layout

Everything lives under `data/`, which is **not** in version control. Paths are
relative to the repository root — run every command from there. Scripts read
these locations from constants at the top of each file, so nothing is
configurable via environment variables.

### You must supply

| Path | What it is | Source |
|---|---|---|
| `data/LNDb/` | LNDb CTs, `LNDb-XXXX.mhd` + `.raw` | `data0.rar` … `data5.rar` (236 training CTs) |
| `data/LNDb/masks/` | Per-radiologist segmentations, `LNDb-XXXX_radN.mhd` + `.raw` | same archives |
| `data/LIDC/lidc_idri/` | LIDC-IDRI DICOMs, one folder per patient | LIDC-IDRI / TCIA |
| `data/trainNodules_gt.csv` | Training findings with radiologist annotations | `data/trainset_csv.zip` |
| `data/testNodules.csv` | Competition test candidates (centroids only) | `data/trainset_csv.zip` |

Supporting CSVs from the same archive — `trainNodules.csv`, `trainCTs.csv`,
`trainFleischner.csv`, `trainFolds.csv` — are used by the notebooks for
exploration and label checks.

Optional: `data/LNDb/chars_trainNodules.csv` gives per-radiologist malignancy
ratings for **all** five radiologists. Without it, notebook 01 falls back to
`notebooks/lndb_processed.csv`, which only covers radiologists 1–3 and leaves
563 LNDb nodules unlabelled. Adding it is the single biggest improvement
available to the labelled training set.

### Produced by the pipeline

Not in version control — generated locally, safe to delete and rebuild.

| Path | Written by |
|---|---|
| `data/nodules/<patient>_noduleNNN.mhd` + `_mask.mhd` | `lidc_preprocessing.py` |
| `data/nodules/meta_info_lidc.csv` | `lidc_preprocessing.py` (end of run) or `lidc_labels.py` (run manually if script doesn't end)|
| `data/nodules/lidc_labels.csv` | `lidc_labels.py` |
| `data/nodules/train/`, `data/nodules/test/` — `LNDb-XXXX_noduleNNN.mhd` + `_mask.mhd` + `lndb_manifest.csv` | `prepare_lndb_nodules.py` |
| `data/radiomics_features.csv` | `extract_features_pyrad.py` |
| `data/radiomics_features_processed.csv`, `radiomics_kept_features.csv`, `radiomics_processing_report.csv` | notebook 01 |
| `data/cv_metrics.csv`, `oof_predictions.csv`, `test_predictions.csv` | notebook 02 |

## Setup

```bash
uv sync                      # main environment (Python 3.10)
```

The LIDC scripts need a **separate** environment: pylidc (2018) breaks against
the `numpy>=2.2` the main environment requires.

```bash
uv venv .venv-lidc --python 3.10
uv pip install --python .venv-lidc/bin/python \
    "numpy==1.23.5" "pylidc==0.2.3" "setuptools<82" SimpleITK pandas jupyter
```

Run LIDC steps with `.venv-lidc/bin/python`, everything else with `uv run`.

## Pipeline

`info/extraction_pipeline.txt` has the full annotated version; this is the
order and the commands.

```bash
# 1. LIDC — nodule volumes + consensus masks  [.venv-lidc]
.venv-lidc/bin/python -m src.mass_detect.scripts.lidc_preprocessing

# 2. LNDb — cubes + masks
uv run python -m src.mass_detect.scripts.prepare_lndb_nodules \
    --findings-csv data/trainNodules_gt.csv --split train
uv run python -m src.mass_detect.scripts.prepare_lndb_nodules \
    --findings-csv data/testNodules.csv --split test

# 3. features
uv run python -m src.mass_detect.scripts.extract_features_pyrad \
    --input data/nodules --dataset LIDC --split train \
    --input data/nodules/train --dataset LNDb --split train \
    --input data/nodules/test  --dataset LNDb --split test \
    --params Params.yml --output data/radiomics_features.csv --jobs 12

# 4. clean + filter features, then train
uv run jupyter notebook notebooks/01_radiomics_features_analysis.ipynb
uv run jupyter notebook notebooks/02_train_model_and_predict.ipynb
```

Steps 1–3 are resumable: existing outputs are skipped, `--force` re-extracts.

## Structure

```
src/mass_detect/
  scripts/
    lidc_preprocessing.py        LIDC-IDRI -> volumes + consensus masks
    lidc_labels.py               LIDC malignancy labels
    prepare_lndb_nodules.py      LNDb -> cubes + radiologist-consensus masks,
                                  + malignancy/is_cancer in the train manifest
    extract_features_pyrad.py    volumes + masks -> PyRadiomics feature table
    join_lndb_characteristics.py joins lndb_processed.csv onto the features
    resampling.py                resampling helper (currently unused; PyRadiomics
                                 does the resampling, per Params.yml)
  lndb_utils/                    utilities shipped with the LNDb dataset

notebooks/
  00_check_cubes.ipynb           cube geometry / mask QC
  01_radiomics_features_analysis.ipynb   clean, filter, attach labels
  02_train_model_and_predict.ipynb      validation + final model
  explorar_lndb.ipynb            LNDb data exploration
  lndb_processed.csv             per-nodule LNDb characteristics (tracked); the
                                 train manifest gets its malignancy from here

Params.yml                       PyRadiomics settings (bin width, resampling, features)
info/                            notes, pipeline description, dataset guides
```

### Nodule identifiers

One scheme everywhere: `<dataset>-<image>_nodule<n>`, zero-padded to three
digits — `LNDb-0001_nodule001`, `LIDC-IDRI-0343_nodule005`. It is derived from
the `.mhd` filename, so it propagates automatically from the preprocessing
scripts through feature extraction into every CSV. The notebooks join on it;
changing it in one place means re-running the pipeline.

## Notes

- The test split needs its CTs extracted separately; as written, its mask step
  expects `RadID`/`RadFindingID` columns that only the training CSV has.
- `notebooks/lndb_processed.csv` keys its rows by the **radiologist's** finding
  number, not by the nodule group of `trainNodules_gt.csv`, and its `x/y/z` are
  voxel indices. Anything that binds it to a finding must therefore do it **by
  position**: `prepare_lndb_nodules.py` does (within 3 mm), while
  `join_lndb_characteristics.py` still joins on the id and so mislabels roughly
  a third of the LNDb rows.
- Only LNDb nodules are ever *validated*, since the competition test set is
  LNDb. LIDC, when enabled, is training-only.