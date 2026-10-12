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

MASKS
-----
The radiologist masks are not binary. In ``LNDb-XXXX_radN.mhd`` every voxel
holds *that radiologist's* FindingID, so the same scan carries several
different labels. The mask of a nodule is therefore built by selecting each
annotating radiologist's own FindingID from their own mask and merging those
with a majority vote, the same rule pylidc applies to LIDC. Findings that no
radiologist segmented are recorded in the manifest instead of written out, so
PyRadiomics only ever sees ROIs it can measure. See ``merge_masks`` and
``radiologist_mask`` below, and ``--min-mask-mm3`` to tune the size floor.

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

Outputs go to ``data/nodules/<split>/`` (see ``--out-dir``): ``LNDb-XXXX_noduleNNN.mhd``
(volume), ``LNDb-XXXX_noduleNNN_mask.mhd`` (mask) and ``lndb_manifest.csv``
(one row per finding). The nodule id follows the same
``<dataset>-<image>_nodule<n>`` scheme as ``lidc_preprocessing.py``, so both
datasets share one identifier across every step of the pipeline.
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

# Smallest ROI worth keeping, as a physical volume in mm^3 rather than a voxel
# count: the CTs are anisotropically sampled (0.49-0.61 mm in plane, 1 mm
# between slices) and PyRadiomics resamples to 1 mm isotropic, so a fixed voxel
# count would mean a different size in every scan. 20 mm^3 is roughly a 3.4 mm
# sphere, below which neither the radiologist's outline nor the resampled ROI
# supports meaningful features. Smaller masks are reported as
# ``mask_too_small`` instead of being written out.
MIN_MASK_MM3 = 20.0

_NOT_READ = object()    # sentinel: "this radiologist's mask is not loaded yet"

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
# Masks: which voxels belong to the nodule we are cutting
# ---------------------------------------------------------------------------
# A radiologist mask LNDb-XXXX_radN.mhd is NOT binary: every voxel holds that
# radiologist's own FindingID, so two nodules annotated by the same radiologist
# carry different values. Verified on all 236 training CTs: for 1525 of 1527
# findings the centroid sits inside the voxel group of its own FindingID.
#
# trainNodules_gt.csv lists one RadID/RadFindingID pair per radiologist that
# annotated the nodule, positionally: "1,2,3" / "3,5,2" means radiologist 1 drew
# it as finding 3, radiologist 2 as finding 5 and radiologist 3 as finding 2.
# Each pair must therefore be read against *its own* mask. Testing every id
# against every mask picks up neighbouring nodules, which is why the previous
# version fell back to "anything non-zero is the nodule".

def radiologist_pairs(row) -> list[tuple[int, int]]:
    """(RadID, RadFindingID) pairs for one finding, read positionally.

    Returns an empty list when the findings file has no radiologist columns,
    which is the case for the competition test candidates.
    """
    if not (hasattr(row, "RadID") and hasattr(row, "RadFindingID")):
        return []
    radii = [int(v) for v in str(row.RadID).split(",")]
    finding_ids = [int(v) for v in str(row.RadFindingID).split(",")]
    return list(zip(radii, finding_ids))


def radiologist_mask(mask_image: sitk.Image, centre, rad_finding_id: int) -> np.ndarray:
    """Binary cube holding one radiologist's segmentation of one nodule.

    Selecting exactly ``rad_finding_id`` is what keeps neighbouring nodules out
    of the ROI: they are different findings of the same radiologist, so they
    carry different values.
    """
    patch = cut_cube(mask_image, centre)
    return (sitk.GetArrayFromImage(patch) == rad_finding_id).astype(np.uint8)


def merge_masks(per_radiologist: list[np.ndarray]) -> np.ndarray:
    """Combine the radiologists' masks of one nodule into a single binary ROI.

    A voxel is kept when at least half of the annotating radiologists included
    it, i.e. ``votes * 2 >= n_radiologists``. This is the same rule as the
    pylidc ``consensus`` used by lidc_preprocessing.py
    (``np.mean(masks, axis=0) >= clevel`` with clevel=0.5), so LNDb and LIDC
    nodules reach PyRadiomics the same way.

    Note the deliberate consequence of "at least half" with an even number of
    radiologists: with 2 annotators the threshold is one vote, so the result is
    their union. With 3 it is a true majority. One radiologist yields exactly
    that radiologist's own segmentation.
    """
    votes = np.sum(per_radiologist, axis=0)
    return (votes * 2 >= len(per_radiologist)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def nodule_id(ct_id: int, finding_id: int) -> str:
    """``LNDb-0031_nodule004``.

    Built in one place so every row of the manifest uses the same id, which is
    what lets a re-run recognise and replace the rows of an earlier run.
    """
    return f"LNDb-{int(ct_id):04d}_nodule{int(finding_id):03d}"


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
    parser.add_argument("--min-mask-mm3", type=float, default=MIN_MASK_MM3,
                        help="drop masks smaller than this volume (see MIN_MASK_MM3); "
                             "lower it to keep small nodules, at the risk of "
                             "PyRadiomics failing on them")
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
                rows.append({"nodule_id": nodule_id(ct_id, row.FindingID),
                             "LNDbID": int(ct_id), "FindingID": int(row.FindingID),
                             "status": "ct_missing", "error": str(ct_path)})
            continue

        image = sitk.ReadImage(str(ct_path))
        # Radiologist masks of this CT, read at most once each and reused by
        # every finding cut from it (they are full-size volumes).
        mask_cache: dict[int, sitk.Image | None] = {}
        for row in group.itertuples():
            centre = (float(row.x), float(row.y), float(row.z))
            this_nodule = nodule_id(ct_id, row.FindingID)
            record = {"nodule_id": this_nodule, "LNDbID": int(ct_id),
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

            image_file = out_dir / f"{this_nodule}.mhd"
            mask_file = out_dir / f"{this_nodule}_mask.mhd"
            if image_file.exists():  # already cut by a previous run
                rows.append(record | {"status": "exists"})
                continue

            # Findings marked as non-nodules have no segmentation: the radiologist
            # only marked a centroid, so their mask is a single voxel. A 1-voxel
            # ROI produces meaningless PyRadiomics features, so skip them.
            if getattr(row, "Nodule", 1) == 0:
                rows.append(record | {"status": "no_segmentation"})
                continue

            pairs = radiologist_pairs(row)
            if not pairs:
                # No radiologist columns (test candidates): cut the cube, but
                # there is no segmentation to extract.
                patch = cut_cube(image, centre)
                sitk.WriteImage(patch, image_file)
                rows.append(record | {"status": "ok_no_mask"})
                continue

            # One binary mask per annotating radiologist, each read against its
            # own RadFindingID.
            per_radiologist = []
            for rad, rad_finding_id in pairs:
                mask_image = mask_cache.get(rad, _NOT_READ)
                if mask_image is _NOT_READ:
                    mask_path = Path(args.masks_dir) / f"LNDb-{int(ct_id):04d}_rad{rad}.mhd"
                    mask_image = sitk.ReadImage(str(mask_path)) if mask_path.exists() else None
                    mask_cache[rad] = mask_image
                if mask_image is None:
                    continue
                try:
                    per_radiologist.append(radiologist_mask(mask_image, centre, rad_finding_id))
                except Exception as error:  # noqa: BLE001 - one bad mask must not stop the run
                    print(f"[warning] {this_nodule}: mask of rad{rad} failed ({error})")

            if not per_radiologist:
                rows.append(record | {"status": "masks_missing"})
                continue

            binary = merge_masks(per_radiologist)
            voxel_count = int(binary.sum())
            if voxel_count == 0:
                # The label existed in the mask but not inside this cube.
                rows.append(record | {"status": "empty_mask", "radiologists": len(per_radiologist),
                                      "mask_voxels": 0})
                continue

            patch = cut_cube(image, centre)

            # Measure the ROI in mm^3 using the CT patch's own spacing, so the
            # same threshold applies to every scan.
            voxel_mm3 = float(np.prod(patch.GetSpacing()))
            volume_mm3 = voxel_count * voxel_mm3
            if volume_mm3 < args.min_mask_mm3:
                # Some radiologists marked a centroid without segmenting it, so
                # their "mask" is a handful of voxels even though the finding is
                # a real nodule (Nodule=1). Too small for PyRadiomics to say
                # anything; lidc_preprocessing.py drops small masks for the same
                # reason.
                rows.append(record | {"status": "mask_too_small",
                                      "radiologists": len(per_radiologist),
                                      "mask_voxels": voxel_count,
                                      "mask_mm3": round(volume_mm3, 2)})
                continue

            sitk.WriteImage(patch, image_file)

            binary_mask = sitk.GetImageFromArray(binary)
            # CRITICAL: Copy spatial metadata (spacing, origin, direction) from the CT patch
            binary_mask.CopyInformation(patch)
            sitk.WriteImage(binary_mask, mask_file)

            rows.append(record | {"radiologists": len(per_radiologist),
                                  "mask_voxels": voxel_count,
                                  "mask_mm3": round(volume_mm3, 2)})

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
                 "text", "status", "error", "radiologists", "mask_voxels", "mask_mm3"]
    columns = [c for c in preferred if c in manifest.columns]
    columns += [c for c in manifest.columns if c not in columns]
    manifest = manifest[columns].sort_values("nodule_id").reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)

    ok = (manifest["status"] == "ok").sum()
    print(f"manifest: {manifest_path}")
    print(f"status counts: {manifest['status'].value_counts().to_dict()}")
    print(f"cubes written to {out_dir}: {ok} ok")
    if "radiologists" in manifest.columns:
        merged = manifest["radiologists"].fillna(1).astype(int)
        print(f"nodules annotated by >1 radiologist: {int((merged > 1).sum())}")


if __name__ == "__main__":
    main()
