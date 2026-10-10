"""Cutting 64 mm patches (1 mm voxels) around a finding: LNDb and LIDC-IDRI."""
import SimpleITK as sitk
import numpy as np
import pandas as pd
import time
from .config import AIR_HU, CENTRE_INDEX, LIDC_DIR, LNDB_DIR, NODULE_TABLE, PATCHES_DIR, PATCH_SIZE, SPACING_MM

def find_ct_files(): # Maps each LNDb CT id to its .mhd file, in the training and test folders
    files = {}
    for pattern in ("data/data[0-5]/LNDb-*.mhd", "data/testdata[01]/LNDb-*.mhd"):
        for path in LNDB_DIR.glob(pattern):
            files[int(path.stem.split("-")[1])] = path
    return files

def load_findings(split): # Findings of the split with their centre in world coordinates (mm)
    if split == "train": table = pd.read_csv(LNDB_DIR / "LNDb_csvs" / "trainNodules_gt.csv")
    else: table = pd.read_csv(LNDB_DIR / "testNodules.csv")
    return table[["LNDbID", "FindingID", "x", "y", "z"]]


def centre_is_inside(image, centre_mm): # Gives True when the physical point falls inside the volume of the image
    index = np.array(image.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in centre_mm)))
    size = np.array(image.GetSize())
    return bool(np.all(index >= -0.5) and np.all(index <= size - 0.5))

def extract_patch(image, centre_mm, interpolator=sitk.sitkLinear, outside_value=AIR_HU): # Cuts 64mm cube around a world point
    #SimpleITK looks up each voxel in the CT using origin, spacing and direction of the CT, so a tilted CT is handled correctly.
    origin = np.asarray(centre_mm, dtype=float) - CENTRE_INDEX * SPACING_MM
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([PATCH_SIZE] * 3)
    resampler.SetOutputSpacing([SPACING_MM] * 3)
    resampler.SetOutputDirection(tuple(np.eye(3).ravel()))
    resampler.SetOutputOrigin(tuple(origin))
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetDefaultPixelValue(float("nan"))                  # Marks the voxels that fall outside the CT

    patch = sitk.GetArrayFromImage(resampler.Execute(image))      # The numpy order is (z, y, x)
    outside = np.isnan(patch)
    patch[outside] = outside_value                                # Points outside the CT get outside_value
    return np.rint(patch).astype(np.int16), float(outside.mean()) # Returns (patch, outside_fraction); outside_fraction says how many voxels they are.

def cut_scan_patches(ct_id, ct_path, findings, split, out_dir): # Cuts and saves the patches of every finding of one CT. Returns one row per finding
    image = sitk.ReadImage(str(ct_path))
    rows = []
    for finding in findings.itertuples():
        patch_id = f"LNDb-{ct_id:04d}_f{finding.FindingID:03d}"
        centre = (finding.x, finding.y, finding.z)
        row = {"patch_id": patch_id, "split": split, "ct_id": ct_id, "finding_id": finding.FindingID,
               "x_mm": finding.x, "y_mm": finding.y, "z_mm": finding.z,
               "status": "ok", "outside_fraction": np.nan, "hu_min": np.nan, "hu_max": np.nan, "file": ""}
        try:
            index = image.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in centre))
            row.update(ct_index_x=index[0], ct_index_y=index[1], ct_index_z=index[2])
            if not centre_is_inside(image, centre):
                row["status"] = "centre_outside_volume"
            else:
                patch, outside_fraction = extract_patch(image, centre)
                relative_path = f"{split}/{patch_id}.npy"
                np.save(out_dir / relative_path, patch)
                row.update(outside_fraction=outside_fraction, hu_min=int(patch.min()), hu_max=int(patch.max()),
                           file=relative_path)
        except Exception as error:              
            row["status"] = f"error: {error}"
        rows.append(row)
    return rows

def cut_lndb_patches(split, limit_cts=None, overwrite=False, out_dir=PATCHES_DIR): # Processes one CT at a time (to keep the memory low) and writes one file per CT
    ct_files = find_ct_files()
    findings = load_findings(split)
    (out_dir / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(exist_ok=True)

    ct_ids = sorted(findings["LNDbID"].unique())[:limit_cts]
    seconds = []
    for ct_id in ct_ids:
        manifest_file = out_dir / "manifests" / f"{split}_LNDb-{ct_id:04d}.csv"
        if manifest_file.exists() and not overwrite: continue # already done, so a long run can be resumed
        started = time.time()
        rows = cut_scan_patches(ct_id, ct_files[ct_id], findings[findings["LNDbID"] == ct_id], split, out_dir)
        pd.DataFrame(rows).to_csv(manifest_file, index=False)
        seconds.append(time.time() - started)
        print(f"{split} CT {ct_id}: {len(rows)} findings in {seconds[-1]:.1f} s")

    manifests = sorted((out_dir / "manifests").glob(f"{split}_LNDb-*.csv"))
    manifest = pd.concat([pd.read_csv(f) for f in manifests], ignore_index=True)
    manifest.to_csv(out_dir / f"manifest_{split}.csv", index=False)
    return manifest, seconds

def summarise_manifest(manifest, split, expected_rows=None): # Prints the checks that matter after a run
    print(f"\n=== {split}: {len(manifest)} rows in the manifest" +
          (f" (expected {expected_rows})" if expected_rows else ""))
    print("status:", manifest["status"].str.split(":").str[0].value_counts().to_dict())
    ok = manifest[manifest["status"] == "ok"]
    for limit in (0.05, 0.25, 0.5):
        print(f"outside_fraction above {limit}: {(ok['outside_fraction'] > limit).sum()}")
    duplicated = manifest.round(3).duplicated(["ct_id", "x_mm", "y_mm", "z_mm"], keep=False)
    print(f"rows that share the same CT and position with another row: {int(duplicated.sum())}")
    if len(ok):
        print(f"HU range of the patches: min {ok['hu_min'].min()}, max {ok['hu_max'].max()}")

def read_series(patient_id, series_uid): # Reads one DICOM series as a SimpleITK image in Hounsfield units.
    # The series is found by its UID inside the patient folder (searched recursively), not by the folder name
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(LIDC_DIR / "lidc_idri" / patient_id), series_uid, False, True) # useSeriesDetails=False, recursive=True
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute()

def cut_series(image, nodules, out_dir): # Cuts and saves the patch of every nodule of one series. Returns one row per nodul
    rows = []
    for nodule in nodules.itertuples():
        # pylidc gives (row, column, slice); SimpleITK indices are (column, row, slice)
        centre = np.array(image.TransformContinuousIndexToPhysicalPoint(
            (float(nodule.vox_col), float(nodule.vox_row), float(nodule.vox_slice))))
        header_point = np.array([nodule.x_mm, nodule.y_mm, nodule.z_mm], dtype=float)
        row = {"patch_id": nodule.nodule_uid, "split": "lidc", "scan_id": nodule.scan_id,
               "x_mm": centre[0], "y_mm": centre[1], "z_mm": centre[2],
               "coord_check_mm": float(np.linalg.norm(centre - header_point)), # NaN when the scan is tilted
               "status": "ok", "outside_fraction": np.nan, "file": ""}
        if not centre_is_inside(image, centre):
            row["status"] = "centre_outside_volume"
        else:
            patch, row["outside_fraction"] = extract_patch(image, centre)
            row["file"] = f"lidc/{nodule.nodule_uid}.npy"
            np.save(out_dir / row["file"], patch)
        rows.append(row)
    return rows

def cut_lidc_patches(out_dir=PATCHES_DIR, limit_series=None): # One series at a time, with one manifest per series so that a long run can be resumed 
    table = pd.read_csv(NODULE_TABLE)
    lidc = table[table["dataset"] == "LIDC"]
    (out_dir / "lidc").mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(exist_ok=True)

    series = sorted(lidc["scan_id"].unique())[:limit_series]
    started = time.time()
    for number, series_uid in enumerate(series, start=1):
        manifest_file = out_dir / "manifests" / f"lidc_{series_uid}.csv"
        if manifest_file.exists():
            continue
        nodules = lidc[lidc["scan_id"] == series_uid]
        image = read_series(nodules["patient_id"].iloc[0], series_uid)
        rows = cut_series(image, nodules, out_dir)
        pd.DataFrame(rows).to_csv(manifest_file, index=False)
        if number % 50 == 0:
            print(f"{number}/{len(series)} series, {time.time() - started:.0f} s", flush=True)

    manifest = pd.concat([pd.read_csv(f) for f in sorted((out_dir / "manifests").glob("lidc_*.csv"))], ignore_index=True)
    # Only nodules whose position two independent computations agree on (within 1 mm) are used for training.
    # In a handful of series the slice order differs between pylidc and SimpleITK, and in tilted series the
    # header point does not exist, so their position cannot be confirmed.
    manifest["position_verified"] = manifest["coord_check_mm"] <= 1.0
    manifest.to_csv(out_dir / "manifest_lidc.csv", index=False)
    print(f"{len(manifest)} LIDC nodules, status: {manifest['status'].value_counts().to_dict()}")
    print(f"position verified (within 1 mm of the header point): {int(manifest['position_verified'].sum())}")
    return manifest
