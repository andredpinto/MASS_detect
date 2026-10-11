"""Cut 64 mm cubes around LNDb findings and segment the nodule at their centre.

WHY THIS SCRIPT EXISTS
----------------------
The LNDb files give, for every finding of every CT, only a centroid in world
coordinates (mm) — both the training findings (``trainNodules_gt.csv``) and the
competition test candidates (``testNodules.csv``). The competition asks for a
malignancy rating for every test candidate, so every feature the model uses must
be computable from a centroid alone, with exactly the same procedure in
training and in test.

This script turns each centroid into the same kind of file that the LIDC
preprocessing script produces for LIDC: a nodule volume ``.mhd`` plus a nodule
mask ``.mhd`` (label 1 = nodule). The feature extraction script
(``extract_features_pyrad.py``) then works on both datasets unchanged.

WHAT A CUBE IS
--------------
* CUBE_SIZE_MM per side

USAGE (from the repository root)
--------------------------------
    # training findings (needs tmp/trainNodules_gt.csv and data/LNDb/)
    uv run python -m src.mass_detect.scripts.prepare_lndb_nodules \
        --findings-csv tmp/trainNodules_gt.csv --split train

    # competition test candidates (needs the test CTs extracted from
    # testdata0.rar / testdata1.rar into data/testdata0, data/testdata1)
    uv run python -m src.mass_detect.scripts.prepare_lndb_nodules \
        --findings-csv data/testNodules.csv --split test

Outputs go to ``data/lndb_nodules/<split>/``: ``LNDb-XXXX_fYYY.mhd`` (volume),
``LNDb-XXXX_fYYY_mask.mhd`` (mask) and ``manifest.csv`` (one row per finding).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Configuration (all overridable by CLI flags; see parse_args)
# ---------------------------------------------------------------------------
DEFAULT_CT_DIR = "data/LNDb"          # training CTs
DEFAULT_MASKS_DIR = "data/LNDb/masks"  # radiologist masks (radiologist mode)
DEFAULT_OUT_DIR = "data/nodules"
DEFAULT_FINDINGS = "data/trainNodules_gt.csv"

CUBE_SIZE_MM = 30.0      # physical side of the cube
SPACING_MM = 1.0         # voxel size of the cube (isotropic)
CENTRE_INDEX = CUBE_SIZE_MM // 2 // SPACING_MM  # voxel 32: the finding centre
AIR_HU = -1024           # value used for voxels outside the CT

# ---------------------------------------------------------------------------
# Geometry: cutting the cube
# ---------------------------------------------------------------------------
def centre_is_inside(image: sitk.Image, centre_mm: tuple[float, float, float]) -> bool:
    """True when a physical point falls inside the volume of the image."""
    index = np.array(image.TransformPhysicalPointToContinuousIndex(
        tuple(float(v) for v in centre_mm)))
    size = np.array(image.GetSize())
    return bool(np.all(index >= -0.5) and np.all(index <= size - 0.5))


def cut_cube(scan: sitk.Image, xyz
             ) -> sitk.Image:
    """Scan and spacing are from readMhd, xyz is in image coordinates,
    cube_size in voxels, cube_size_mm in mm. Returns cube of size cube_size^3 voxels."""
    # Extract cube of cube_size^3 voxels and world dimensions of cube_size_mm^3 mm from scan at image coordinates xyz
    center = scan.TransformPhysicalPointToIndex(xyz)   # Get indexes from world coordinates
    spacing = np.array(scan.GetSpacing())
    scan_halfcube_size = np.ceil((CUBE_SIZE_MM/spacing)/2).astype(int)
    if np.any(center<scan_halfcube_size) or np.any(center+scan_halfcube_size>scan.GetSize()): # check if padding is necessary (exceeds scan boundaries)
        maxsize = int(max(scan_halfcube_size))
        pad = [maxsize, maxsize, maxsize]
        scan = sitk.ConstantPad(    # Padding function, suggested by AI
            scan,
            padLowerBound=pad,
            padUpperBound=pad,
            constant=AIR_HU)
        
        center = scan.TransformPhysicalPointToIndex(xyz)   # get center matching the resampled image
    
    scancube = scan[center[0]-scan_halfcube_size[0]:center[0]+scan_halfcube_size[0], # extract cube from scan at xyz
                    center[1]-scan_halfcube_size[1]:center[1]+scan_halfcube_size[1],
                    center[2]-scan_halfcube_size[2]:center[2]+scan_halfcube_size[2]]

    return scancube

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut 64 mm cubes around LNDb finding centroids and segment them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--findings-csv", default=DEFAULT_FINDINGS,
                        help="CSV with columns LNDbID, FindingID, x, y, z "
                             "(trainNodules_gt.csv or testNodules.csv)")
    parser.add_argument("--ct-dir", default=DEFAULT_CT_DIR,
                        help="directory holding the CT scans as LNDb-XXXX.mhd")
    parser.add_argument("--split", default="train", choices=("train", "test"),
                    help="split label; recorded in the manifest and the output folder")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="output root; cubes go to <out-dir>/<split>/")
    parser.add_argument("--masks-dir", default=DEFAULT_MASKS_DIR,
                        help="radiologist masks, used only with --use-radiologist-masks")
    parser.add_argument("--limit", type=int, default=0,
                        help="process at most this many CTs (0 = all; for smoke tests)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    findings = pd.read_csv(args.findings_csv)
    for column in ("LNDbID", "FindingID", "x", "y", "z"):
        if column not in findings.columns:
            raise SystemExit(f"{args.findings_csv} has no column '{column}'")

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "lndb_manifest.csv"

    # A previous run's manifest is reused for the rows it already has, so the
    # script can be interrupted and continued (a cube is skipped when its
    # image file already exists).
    old_manifest = pd.read_csv(manifest_path) if manifest_path.exists() else None

    rows = []
    ct_ids = sorted(findings["LNDbID"].unique())
    if args.limit:
        ct_ids = ct_ids[:args.limit]
    print(f"{args.split}: {len(findings)} findings in {len(ct_ids)} CTs "
          f"({args.findings_csv})")

    for ct_pos, ct_id in enumerate(ct_ids, start=1):
        ct_path = Path(args.ct_dir) / f"LNDb-{int(ct_id):04d}.mhd"
        group = findings[findings["LNDbID"] == ct_id]
        if not ct_path.exists():
            for row in group.itertuples():
                rows.append({"nodule_id": f"LNDb-{int(ct_id):04d}_f{int(row.FindingID):03d}",
                             "LNDbID": int(ct_id), "FindingID": int(row.FindingID),
                             "status": "ct_missing", "error": str(ct_path)})
            continue

        image = sitk.ReadImage(str(ct_path))
        for row in group.itertuples():
            centre = (float(row.x), float(row.y), float(row.z))
            nodule_id = f"LNDb-{int(ct_id):04d}_nodule{int(row.FindingID)}"
            record = {"nodule_id": nodule_id, "LNDbID": int(ct_id),
                      "FindingID": int(row.FindingID), "x": row.x, "y": row.y, "z": row.z,
                      "status": "ok", "error": ""}

            # Keep columns that exist in the findings file (train has Nodule,
            # Volume, ...; the test file has only the coordinates).
            for column in ("Nodule", "Volume", "Text"):
                if hasattr(row, column):
                    record[column.lower()] = getattr(row, column)

            if not centre_is_inside(image, centre):
                record["status"] = "centre_outside_volume"
                rows.append(record)
                continue

            image_file, mask_file = out_dir / f"{nodule_id}.mhd", out_dir / f"{nodule_id}_mask.mhd"
            if image_file.exists():  # already cut by a previous run
                rows.append(record | {"status": "exists"})
                continue

            patch = cut_cube(image, centre)
            sitk.WriteImage(patch, image_file)

            # Mask: segment same way as the ct scan, for all available masks
            radiologists = [int(v) for v in str(row.RadID).split(",")]
            finding_numbers = [int(v) for v in str(row.RadFindingID).split(",")]
            for rad, finding_numbers in zip(radiologists, finding_numbers):
                mask_path = Path(DEFAULT_MASKS_DIR) / f"LNDb-{ct_id:04d}_rad{rad}.mhd"
                if not mask_path.exists():
                    continue
                try:
                    mask_image = sitk.ReadImage(str(mask_path))
                    mask_patch = cut_cube(mask_image, centre)
                    sitk.WriteImage(mask_patch, mask_file)
                except:
                    print(f"[warning] something wrong writing {mask_file}")

            rows.append(record)

        if ct_pos % 25 == 0 or ct_pos == len(ct_ids):
            print(f"  CT {ct_pos}/{len(ct_ids)} ({len(rows)} findings processed)", flush=True)

    manifest = pd.DataFrame(rows)
    if old_manifest is not None:  # keep rows of CTs this run did not touch
        done_ids = set(manifest["nodule_id"])
        manifest = pd.concat(
            [old_manifest[~old_manifest["nodule_id"].isin(done_ids)], manifest],
            ignore_index=True)
    # Stable column order; QC columns first, PyRadiomics needs none of these.
    preferred = ["nodule_id", "LNDbID", "FindingID", "x", "y", "z", "nodule", "volume",
                 "text", "status", "error"]
    columns = [c for c in preferred if c in manifest.columns]
    columns += [c for c in manifest.columns if c not in columns]
    manifest = manifest[columns].sort_values("nodule_id").reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)

    ok = (manifest["status"] == "ok").sum()
    print(f"manifest: {manifest_path}")
    print(f"status counts: {manifest['status'].value_counts().to_dict()}")
    print(f"cubes written to {out_dir}: {ok} ok")


if __name__ == "__main__":
    main()
