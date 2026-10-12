"""Extract PyRadiomics features from nodule volumes (.mhd/.raw pairs).

WHAT THIS SCRIPT DOES
---------------------
For every nodule volume ``<name>.mhd`` in an input directory that has a matching
mask ``<name>_mask.mhd``, it computes the PyRadiomics features configured in a
YAML parameter file (default: ``Params.yml`` at the repository root) and writes
one row per nodule to a CSV file.

The input directories are produced by *preprocessing* scripts, not by this one:

* ``data/nodules``          -> LIDC-IDRI nodule volumes + consensus masks,
                               produced by ``scripts/lidc_preprocessing.py``
                               (needs the ``.venv-lidc`` environment; see
                               AGENTS.md). Run it on the whole dataset before
                               expecting every LIDC nodule to be here.
* ``data/nodules/<split>``  -> LNDb 64 mm cubes around every finding + masks,
                               produced by ``scripts/prepare_lndb_nodules.py``
                               (automatic segmentation from the nodule centre,
                               the same procedure for training and test rows).

Both produce ``<image>.mhd`` + ``<image>_mask.mhd`` pairs, which is all this
script needs, so new datasets can be added by pointing ``--input`` at a new
directory. Nothing about LIDC or LNDb is hard-coded in the extraction itself.

WHY A SEPARATE SCRIPT (instead of doing it in a notebook)
---------------------------------------------------------
Extraction is slow (hundreds of features per nodule, thousands of nodules) and
must run once, not every time a notebook is opened. It is also embarrassingly
parallel, so it uses multiple processes. The notebooks only read the CSV.

RESUMABILITY
------------
If the output CSV already exists, nodule ids that are already present are
skipped (a run can therefore be interrupted and continued; delete the CSV, or
pass ``--force``, to re-extract everything). Rows keep the order in which they
were first extracted, with newly extracted rows appended at the end.

FAILURES
--------
A nodule whose extraction raises (empty mask, unreadable file, ROI below
``minimumROISize`` ...) is recorded as a row with ``status = error`` and the
exception message in ``error``, so the failure is visible in the CSV instead of
stopping the whole run. Failed rows are retried on the next run (only rows with
``status = ok`` are considered "already extracted").

USAGE (from the repository root)
--------------------------------
    uv run python -m src.mass_detect.scripts.extract_features_pyrad \
        --input data/nodules --dataset LIDC --split train \
        --input data/nodules/train --dataset LNDb --split train \
        --input data/nodules/test --dataset LNDb --split test \
        --params Params.yml \
        --output data/radiomics_features.csv \
        --jobs 12

Every option can also be set with an environment variable of the same name in
caps (e.g. ``MASS_OUTPUT_CSV``), which is handy for scheduled runs.

OUTPUT SCHEMA
-------------
nodule_id   unique id of the nodule (the image file name without extension,
            ``<dataset>-<image>_nodule<n>``, e.g. ``LIDC-IDRI-0001_nodule000``
            or ``LNDb-0001_nodule001``)
dataset     free label given by --dataset (e.g. LIDC / LNDb)
split       free label given by --split (e.g. train / test)
image_path, mask_path
            the files that were used (for traceability)
status      "ok" or "error"
error       exception message when status is "error", else empty
<features>  one column per PyRadiomics feature, named
            ``<imageType>_<featureClass>_<featureName>`` (e.g.
            ``original_firstorder_Mean``,
            ``log-sigma-2-0-mm-3D_glcm_Correlation``), plus the
            ``diagnostics_*`` columns that PyRadiomics returns (spacing,
            dimensions, ... — useful for quality control; the analysis notebook
            drops them before modelling).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration: defaults, overridable by CLI arguments and environment
# variables (see parse_args). Paths are relative to the current working
# directory, so run from the repository root (as everywhere else in this
# project) or pass absolute paths.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = "Params.yml"
DEFAULT_OUTPUT = "data/radiomics_features.csv"
DEFAULT_JOBS = min(12, (os.cpu_count() or 4))  # keep some cores free

# PyRadiomics logs one line per feature class per image; silence it. Errors
# still reach the worker's exception handler.
logging.getLogger("radiomics").setLevel(logging.ERROR)

# One extractor per worker process (PyRadiomics extractors are not picklable
# and should be created once per process, not once per nodule).
_EXTRACTOR = None
_PARAMS_PATH = None


def _init_worker(params_path: str) -> None:
    """Create the module-level extractor for one worker process."""
    global _EXTRACTOR, _PARAMS_PATH
    from radiomics import featureextractor  # imported here: only needed in workers

    # Workers are separate processes: silence PyRadiomics here too
    # (it logs one informational line per feature class per image type).
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    _PARAMS_PATH = params_path
    _EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(params_path)


def _extract_one(task: tuple) -> dict:
    """Extract the features of one nodule. Returns a row for the output CSV.

    Runs in a worker process; any exception is caught so that one bad nodule
    cannot stop the run (the row is written with status="error").
    """
    nodule_id, dataset, split, image_path, mask_path = task
    row = {
        "nodule_id": nodule_id,
        "dataset": dataset,
        "split": split,
        "image_path": image_path,
        "mask_path": mask_path,
        "status": "ok",
        "error": "",
    }
    try:
        # execute() takes file paths (strings): SimpleITK then reads the images
        # itself, which keeps memory usage low for thousands of nodules.
        result = _EXTRACTOR.execute(image_path, mask_path)
        for name, value in result.items():
            # Almost every value is a float, but the diagnostics include
            # strings (e.g. the PyRadiomics version), which are kept as is.
            try:
                row[name] = float(value)
            except (TypeError, ValueError):
                row[name] = value
    except Exception as error:  # noqa: BLE001 - a failed nodule must not stop the run
        row["status"] = "error"
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def find_nodule_pairs(input_dir: Path) -> list[tuple[str, str]]:
    """Find (image, mask) .mhd pairs in one directory.

    A nodule volume is every ``*.mhd`` whose name does not contain "mask"; its
    mask must be the sibling file ``<name>_mask.mhd``. This naming convention
    is what the preprocessing scripts produce. Volumes without a mask are
    reported (they are skipped — features need a region of interest).
    """
    pairs, missing_masks = [], []
    for image in sorted(input_dir.glob("*.mhd")):
        if "mask" in image.name:
            continue
        mask = image.with_name(image.stem + "_mask.mhd")
        if mask.exists():
            pairs.append((str(image), str(mask)))
        else:
            missing_masks.append(image.name)
    if missing_masks:
        print(f"  [warn] {len(missing_masks)} volumes in {input_dir} have no mask and were skipped "
              f"(e.g. {missing_masks[0]})", file=sys.stderr)
    return pairs


def already_done(output: Path) -> set[str]:
    """Nodule ids already extracted successfully in a previous run."""
    if not output.exists():
        return set()
    done = pd.read_csv(output, usecols=["nodule_id", "status"])
    return set(done.loc[done["status"] == "ok", "nodule_id"])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PyRadiomics features from nodule .mhd/_mask.mhd pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", action="append", required=True, metavar="DIR",
                        help="directory with nodule .mhd pairs; give once per "
                             "(directory, dataset, split) combination, in matching order")
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME",
                        help="dataset label for the corresponding --input (e.g. LIDC)")
    parser.add_argument("--split", action="append", required=True, metavar="NAME",
                        help="split label for the corresponding --input (e.g. train)")
    parser.add_argument("--params", default=DEFAULT_PARAMS,
                        help="PyRadiomics parameter file (YAML)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="output CSV with one row per nodule")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="number of parallel worker processes")
    parser.add_argument("--limit", type=int, default=0,
                        help="extract at most this many nodules per input dir (0 = no limit; "
                             "useful for smoke tests)")
    parser.add_argument("--force", action="store_true",
                        help="re-extract nodules that are already in the output CSV")
    args = parser.parse_args(argv)

    if not (len(args.input) == len(args.dataset) == len(args.split)):
        parser.error("--input, --dataset and --split must be given the same number of times")
    for path in [args.params, *args.input]:
        if not Path(path).exists():
            parser.error(f"path does not exist: {path}")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Collect every (nodule, dataset, split) to process, skipping what a
    # previous successful run already extracted.
    done = set() if args.force else already_done(output)
    tasks = []
    for input_dir, dataset, split in zip(args.input, args.dataset, args.split):
        pairs = find_nodule_pairs(Path(input_dir))
        for image_path, mask_path in pairs:
            nodule_id = Path(image_path).stem
            if nodule_id in done:
                continue
            tasks.append((nodule_id, dataset, split, image_path, mask_path))
        print(f"{input_dir} ({dataset}/{split}): {len(pairs)} nodule pairs found")
    if args.limit:
        # keep at most --limit new tasks per input directory (smoke tests)
        per_dir, limited = {}, []
        for task in tasks:
            key = str(Path(task[3]).parent)
            per_dir[key] = per_dir.get(key, 0) + 1
            if per_dir[key] <= args.limit:
                limited.append(task)
        tasks = limited
    print(f"to extract: {len(tasks)} nodules "
          f"({len(done)} already done in {output.name})" if done else
          f"to extract: {len(tasks)} nodules")
    if not tasks:
        print("nothing to do.")
        return

    started = time.time()
    columns = None
    rows_existing = None
    if output.exists():
        rows_existing = pd.read_csv(output)
        columns = list(rows_existing.columns)

    # One process per core chunk; each worker builds its own extractor once.
    with Pool(processes=args.jobs, initializer=_init_worker, initargs=(args.params,)) as pool:
        new_rows = []
        for i, row in enumerate(pool.imap(_extract_one, tasks), start=1):
            new_rows.append(row)
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - started
                rate = i / max(elapsed, 1e-9)
                eta = (len(tasks) - i) / max(rate, 1e-9)
                print(f"  {i}/{len(tasks)} nodules "
                      f"({rate:.1f} nodules/s, ETA {eta / 60:.0f} min)", flush=True)

    new_df = pd.DataFrame(new_rows)
    # Keep a stable column order: metadata first, then features in the order
    # PyRadiomics returned them (identical for every nodule).
    feature_cols = [c for c in new_df.columns
                    if c not in ("nodule_id", "dataset", "split", "image_path",
                                 "mask_path", "status", "error")]
    ordered = (["nodule_id", "dataset", "split", "image_path", "mask_path", "status", "error"]
               + sorted(feature_cols, key=lambda c: (not c.startswith("diagnostics"), c)))
    new_df = new_df[ordered]

    if rows_existing is not None:
        # Align columns (a re-run after a Params.yml change adds new feature
        # columns; old rows get NaN for them, which the notebooks impute).
        rows_existing = rows_existing.reindex(columns=ordered)
        final = pd.concat([rows_existing, new_df], ignore_index=True)
    else:
        final = new_df
    final.to_csv(output, index=False)

    ok = int((new_df["status"] == "ok").sum())
    print(f"done: {ok} extracted, {len(new_df) - ok} failed, "
          f"{time.time() - started:.0f} s total -> {output}")
    print(f"rows in file: {len(final)} | feature columns: {len(feature_cols)}")


if __name__ == "__main__":
    main()
