"""Compute malignancy labels for the LIDC nodules in data/processed.

WHY THIS SCRIPT EXISTS
------------------------
The LIDC preprocessing script (``lidc_preprocessing.py``) saves the
nodule volumes and masks to ``data/processed`` but writes its metadata
file (``meta_info_lidc.csv``, which holds the malignancy of every
nodule) only at the very end of the run. The run so far was
interrupted, so the 1986 nodules already extracted have no labels.

This script recomputes the same labels, for the same nodules, so the
pipeline can run without waiting for a full re-run of the
preprocessing. After the preprocessing has been run to completion, its
``meta_info_lidc.csv`` takes precedence (the analysis notebook checks
for it first), and this file is no longer needed.

HOW THE LABELS ARE COMPUTED
-----------------------------
Exactly as in ``lidc_preprocessing.py``:

* the nodules of a scan are ``scan.cluster_annotations()`` (pylidc
  groups the annotations of the four radiologists that belong to the
  same physical nodule); ``nodule_idx`` is the position in that list,
  which is how the preprocessing script named the files
  (``<patient_id>_nodule<nodule_idx>.mhd``);
* the malignancy of a nodule is ``median_high`` of the malignancy
  ratings (1-5) of its annotations — the high median, the same
  function the preprocessing uses;
* ``is_cancer`` follows the preprocessing convention: True above 3,
  False below 3, 'Ambiguous' at exactly 3.

For flexibility we also store the number of annotations, their mean
rating and the individual ratings, so the analysis notebook can use a
different label (e.g. the mean) without re-running anything.

RUN IT WITH THE LIDC ENVIRONMENT (pylidc does not work in the main
environment; see AGENTS.md):

    .venv-lidc/bin/python -m src.mass_detect.scripts.lidc_labels

Outputs ``data/processed/lidc_labels.csv`` with one row per nodule of
every patient that has nodules in data/processed.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from statistics import median_high  # the same function lidc_preprocessing uses

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUTPUT_CSV = PROCESSED_DIR / "lidc_labels.csv"


def import_pylidc():
    """Import pylidc, patching the numpy aliases it still uses.

    pylidc 0.2.3 (2018) uses np.int / np.float / np.bool, which were
    removed from numpy; the LIDC environment pins numpy 1.23.5 where
    they still exist, so the patch is a no-op there. It is kept so the
    script also works in other environments.
    """
    for alias, real_type in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, real_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pylidc
    return pylidc


def main() -> None:
    pl = import_pylidc()

    # Patients that have at least one nodule volume in data/processed.
    patient_ids = sorted({p.stem.split("_nodule")[0]
                          for p in PROCESSED_DIR.glob("*_nodule[0-9]*.mhd")
                          if "mask" not in p.name})
    if not patient_ids:
        raise SystemExit(f"no nodule volumes found in {PROCESSED_DIR}")
    print(f"{len(patient_ids)} patients with nodules in {PROCESSED_DIR}")

    rows = []
    for pos, patient_id in enumerate(patient_ids, start=1):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
        if scan is None:
            print(f"  [warn] {patient_id}: no scan in pylidc, skipped")
            continue
        # Same enumeration as lidc_preprocessing.py: the file names are
        # indexed by the position in this list.
        for nodule_idx, annotations in enumerate(scan.cluster_annotations()):
            ratings = [int(a.malignancy) for a in annotations]
            rows.append({
                "patient_id": patient_id,
                "nodule_idx": nodule_idx,
                "nodule_id": f"{patient_id}_nodule{nodule_idx}",
                "n_annotations": len(ratings),
                "malignancy_median_high": int(median_high(ratings)) if ratings else None,
                "malignancy_mean": float(np.mean(ratings)) if ratings else None,
                "malignancy_min": min(ratings) if ratings else None,
                "malignancy_max": max(ratings) if ratings else None,
                "ratings": ",".join(map(str, ratings)),
            })
        if pos % 50 == 0 or pos == len(patient_ids):
            print(f"  {pos}/{len(patient_ids)} patients "
                  f"({len(rows)} nodules)", flush=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} nodules to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
