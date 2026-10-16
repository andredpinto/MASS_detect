"""Regions drawn by the radiologists, their PyRadiomics features and how robust those features are between readers.

For every nodule of the cohort we keep one mask per reading and the consensus mask, cut on the same 64 mm grid
as the CT patch (voxels of SPACING_MM, centre on CENTRE_INDEX), so that a mask and its patch overlap voxel by voxel.
"""
from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
import pandas as pd
import time
from .config import (CENTRE_INDEX, PATCH_SIZE, SPACING_MM, LIDC_MANIFEST_FILE, LNDB_DIR, LNDB_MANIFEST_FILE, MASK_FILE, PATCHES_DIR,
                     READINGS_FILE, ROI_DIR, ROI_ICC_FILE, ROI_RADIOMICS_FILE, ROI_WAVELET_FILE)
from .lidc import import_pylidc
from .patches import extract_patch, read_series
from .radiomics_features import IMAGE_TYPES, ROBUST_ICC, WAVELET, extract, icc_2_1
from .schema import equivalent_diameter

CONSENSUS_LEVEL = 0.5   # a voxel is in the consensus when at least half of the readers drew it (as in pylidc)

# Readings: one row per radiologist who drew and rated a cohort nodule

def lndb_readings(cohort): # Who read each LNDb nodule, the label of the nodule in that reader's mask file, and the rating
    findings = pd.read_csv(LNDB_DIR / "LNDb_csvs" / "trainNodules.csv")
    # chars_trainNodules.csv lists the same readings in the same row order (checked in notebook explorar_lndb)
    findings["rating"] = pd.read_csv(LNDB_DIR / "data" / "chars_trainNodules.csv")["malignancy"].values
    rating_of = findings.set_index(["LNDbID", "RadID", "FindingID"])["rating"]

    merged = pd.read_csv(LNDB_DIR / "LNDb_csvs" / "trainNodules_gt.csv")
    merged["nodule_uid"] = [f"LNDb-{ct:04d}_{finding}" for ct, finding in zip(merged["LNDbID"], merged["FindingID"])]
    rows = []
    for nodule in merged[merged["nodule_uid"].isin(cohort["nodule_uid"])].itertuples():
        for reader, finding in zip(str(nodule.RadID).split(","), str(nodule.RadFindingID).split(",")):
            rating = rating_of[(nodule.LNDbID, int(reader), int(finding))]
            rows.append({"nodule_uid": nodule.nodule_uid, "reader": f"rad{reader}", "mask_label": int(finding),
                         "rating": rating if rating > 0 else np.nan})   # 0 means the rating was not filled in
    return pd.DataFrame(rows)

def lidc_readings(cohort): # Every pylidc annotation of each LIDC nodule. One annotation is one reading.
    pl = import_pylidc()
    rating_of = {a.id: int(a.malignancy) for a in pl.query(pl.Annotation).all()}
    lidc = cohort[cohort["dataset"] == "LIDC"]
    rows = []
    for nodule_uid, annotation_ids in zip(lidc["nodule_uid"], lidc["annotation_ids"]):
        for annotation_id in str(annotation_ids).split(";"):
            rows.append({"nodule_uid": nodule_uid, "reader": f"annotation{annotation_id}",
                         "mask_label": int(annotation_id), "rating": rating_of[int(annotation_id)]})
    return pd.DataFrame(rows)

def build_readings(cohort): # Both datasets in one table, saved to READINGS_FILE
    readings = pd.concat([lndb_readings(cohort), lidc_readings(cohort)], ignore_index=True)
    readings = readings.merge(cohort[["nodule_uid", "patch_id", "dataset", "patient_id"]], on="nodule_uid")
    readings.to_csv(READINGS_FILE, index=False)
    return readings

# Masks on the patch grid

def patch_centres(): # Centre (x, y, z in mm) of every patch, exactly as it was used to cut the CT patch
    manifest = pd.concat([pd.read_csv(LNDB_MANIFEST_FILE), pd.read_csv(LIDC_MANIFEST_FILE)], ignore_index=True)
    return {p: np.array([x, y, z]) for p, x, y, z in manifest[["patch_id", "x_mm", "y_mm", "z_mm"]].itertuples(index=False)}

def save_rois(patch_id, readers, masks): # One file per nodule with every reader mask and the consensus mask
    masks = np.array(masks, dtype=bool)
    consensus = masks.mean(axis=0) >= CONSENSUS_LEVEL
    np.savez_compressed(ROI_DIR / f"{patch_id}.npz", readers=np.array(readers), masks=masks, consensus=consensus)

def lndb_rois(ct_id, nodules, readings, centres): # Masks of the cohort nodules of one LNDb CT
    # Each LNDb reader has one mask file per CT, where every finding is drawn with its own finding number
    mask_images = {}
    for nodule in nodules.itertuples():
        mine = readings[readings["nodule_uid"] == nodule.nodule_uid]
        masks = []
        for reading in mine.itertuples():
            if reading.reader not in mask_images:
                path = str(MASK_FILE).format(ct=ct_id, rad=reading.reader.replace("rad", ""))
                mask_images[reading.reader] = sitk.ReadImage(path)
            patch, _ = extract_patch(mask_images[reading.reader], centres[nodule.patch_id],
                                     interpolator=sitk.sitkNearestNeighbor, outside_value=0)
            masks.append(patch == reading.mask_label)
        save_rois(nodule.patch_id, mine["reader"].tolist(), masks)

def lidc_annotation_mask(annotation, image, centre_mm): # Mask of one pylidc annotation on the patch grid around centre_mm
    box_mask = annotation.boolean_mask()                  # (row, column, slice), only inside the annotation's bounding box
    row, column, slice_index = (s.start for s in annotation.bbox())   # where that box starts inside the CT
    box = sitk.GetImageFromArray(np.transpose(box_mask, (2, 0, 1)).astype(np.uint8))   # SimpleITK arrays are (slice, row, column)
    box.SetSpacing(image.GetSpacing())
    box.SetDirection(image.GetDirection())
    box.SetOrigin(image.TransformIndexToPhysicalPoint((int(column), int(row), int(slice_index))))
    patch, _ = extract_patch(box, centre_mm, interpolator=sitk.sitkNearestNeighbor, outside_value=0)
    return patch > 0

def lidc_rois(series_uid, nodules, readings, centres): # Masks of the cohort nodules of one LIDC series
    pl = import_pylidc()
    image = read_series(nodules["patient_id"].iloc[0], series_uid)   # read only for its geometry (origin, spacing, direction)
    for nodule in nodules.itertuples():
        mine = readings[readings["nodule_uid"] == nodule.nodule_uid]
        annotations = {a.id: a for a in pl.query(pl.Annotation).filter(pl.Annotation.id.in_(mine["mask_label"].tolist()))}
        masks = [lidc_annotation_mask(annotations[label], image, centres[nodule.patch_id]) for label in mine["mask_label"]]
        save_rois(nodule.patch_id, mine["reader"].tolist(), masks)

def _roi_job(job): # Runs one CT (LNDb) or one series (LIDC) in a worker process
    dataset, scan, nodules, readings, centres = job
    try:
        if dataset == "LNDb":
            lndb_rois(int(scan.split("-")[1]), nodules, readings, centres)
        else:
            lidc_rois(scan, nodules, readings, centres)
        return scan, ""
    except Exception as error:      # one failing scan must not stop the run; it is reported at the end
        return scan, str(error)

def build_rois(cohort, readings, processes=6): # Writes ROI_DIR/<patch_id>.npz for every cohort nodule that has no file yet
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    centres = patch_centres()
    todo = cohort[[not (ROI_DIR / f"{p}.npz").exists() for p in cohort["patch_id"]]]
    jobs = []
    for (dataset, scan), nodules in todo.groupby(["dataset", "scan_id"]):
        mine = readings[readings["nodule_uid"].isin(nodules["nodule_uid"])]
        jobs.append((dataset, scan, nodules, mine, {p: centres[p] for p in nodules["patch_id"]}))
    started, failures = time.time(), []
    with Pool(processes) as pool:
        for number, (scan, error) in enumerate(pool.imap_unordered(_roi_job, jobs), start=1):
            if error:
                failures.append((scan, error))
            if number % 200 == 0:
                print(f"{number}/{len(jobs)} scans, {time.time() - started:.0f} s", flush=True)
    print(f"{len(jobs)} scans processed in {time.time() - started:.0f} s; failures: {failures}")

def roi_quality(cohort): # Checks per nodule: readers, consensus size and how far the consensus sits from the patch centre
    rows = []
    for patch_id in cohort["patch_id"]:
        rois = np.load(ROI_DIR / f"{patch_id}.npz")
        consensus = rois["consensus"]
        reader_voxels = rois["masks"].sum(axis=(1, 2, 3))
        offset = np.linalg.norm(np.argwhere(consensus).mean(axis=0) - CENTRE_INDEX) * SPACING_MM if consensus.any() else np.nan   # in mm
        rows.append({"patch_id": patch_id, "readers": len(rois["readers"]), "consensus_voxels": int(consensus.sum()),
                     "empty_reader_masks": int((reader_voxels == 0).sum()), "centre_offset_mm": offset,
                     "binary": rois["masks"].dtype == bool and consensus.dtype == bool,                 # masks hold only 0 and 1
                     "shape_ok": rois["masks"].shape[1:] == consensus.shape == (PATCH_SIZE,) * 3})    # same grid as the CT patch
    return cohort[["patch_id", "dataset"]].merge(pd.DataFrame(rows), on="patch_id")

# PyRadiomics on every region

def _region_features(item): # Features of the consensus and of every reader mask of one nodule (runs in a worker)
    patch_id, patch_file, image_types = item
    patch = np.load(PATCHES_DIR / patch_file)
    rois = np.load(ROI_DIR / f"{patch_id}.npz")
    regions = [("consensus", rois["consensus"])] + list(zip(rois["readers"], rois["masks"]))
    rows = []
    for roi, mask in regions:
        try:
            rows.append({"patch_id": patch_id, "roi": str(roi), **extract(patch, mask, image_types)})
        except Exception as error:   # one failing region must not stop the run
            rows.append({"patch_id": patch_id, "roi": str(roi), "error": str(error)})
    return rows

def extract_roi_radiomics(cohort, processes=12, image_types=IMAGE_TYPES, output_file=ROI_RADIOMICS_FILE, part_size=250):
    """Saves one row per region to output_file.

    The rows are first written in parts of part_size nodules to a folder next to output_file, so a run that stops
    (a restart of the machine) continues from the last part instead of starting again.
    """
    manifest = pd.concat([pd.read_csv(LNDB_MANIFEST_FILE), pd.read_csv(LIDC_MANIFEST_FILE)], ignore_index=True)
    file_of = dict(zip(manifest["patch_id"], manifest["file"]))
    parts_dir = output_file.with_name(output_file.stem + "_parts")
    parts_dir.mkdir(exist_ok=True)
    done = set()
    for part in parts_dir.glob("part_*.csv"):
        done |= set(pd.read_csv(part, usecols=["patch_id"])["patch_id"])
    items = [(p, file_of[p], image_types) for p in cohort["patch_id"] if p not in done]
    print(f"{len(done)} nodules already extracted, {len(items)} to go", flush=True)

    started, rows, part = time.time(), [], len(list(parts_dir.glob("part_*.csv")))
    with Pool(processes) as pool:
        for number, nodule_rows in enumerate(pool.imap_unordered(_region_features, items, chunksize=2), start=1):
            rows += nodule_rows
            if number % part_size == 0 or number == len(items):
                pd.DataFrame(rows).to_csv(parts_dir / f"part_{part:03d}.csv", index=False)
                part, rows = part + 1, []
                print(f"{number}/{len(items)} nodules, {time.time() - started:.0f} s", flush=True)
    features = pd.concat([pd.read_csv(p) for p in sorted(parts_dir.glob("part_*.csv"))], ignore_index=True)
    features.to_csv(output_file, index=False)
    print(f"{len(features)} regions saved to {output_file.name} in {time.time() - started:.0f} s")
    return features

def extract_roi_wavelet(cohort, processes=12): # The same regions on the wavelet images, saved to ROI_WAVELET_FILE
    return extract_roi_radiomics(cohort, processes, image_types=WAVELET, output_file=ROI_WAVELET_FILE)

def load_roi_radiomics(): # Every region with its features from the original, LoG and wavelet images
    features = pd.read_csv(ROI_RADIOMICS_FILE)
    wavelet = pd.read_csv(ROI_WAVELET_FILE)
    # PyRadiomics always adds the shape features of the mask, which are already in the first table
    wavelet = wavelet[["patch_id", "roi"] + [c for c in wavelet.columns if c.startswith("wavelet-")]]
    return features.merge(wavelet, on=["patch_id", "roi"], how="left", validate="one_to_one")

def radiomic_columns(features): # The PyRadiomics columns: original image, Laplacian of Gaussian and wavelet images
    return [c for c in features.columns if c.startswith(("original_", "log-", "wavelet-"))]

def add_roi_diameters(features): # Size of the radiologists' region, from the PyRadiomics shape features
    features = features.copy()
    features["roi_diameter_mm"] = equivalent_diameter(features["original_shape_MeshVolume"])   # sphere with the same volume
    features["roi_axial_diameter_mm"] = features["original_shape_Maximum2DDiameterSlice"]      # longest diameter in an axial slice
    return features

# Robustness between readers

def first_two_readers(rows, names): # Feature values of the first two readers of every nodule read by at least two
    rows = rows.dropna(subset=names).sort_values(["patch_id", "roi"])   # regions under 8 voxels have no features
    rows = rows.assign(position=rows.groupby("patch_id").cumcount())
    first = rows[rows["position"] == 0].set_index("patch_id")[names]
    second = rows[rows["position"] == 1].set_index("patch_id")[names]
    first = first.loc[second.index]
    return first.to_numpy(), second.to_numpy()

def reader_icc(features, readings): # ICC(2,1) of every feature between two readers of the same nodule, in each dataset.
    # A feature is robust when its ICC is at least ROBUST_ICC in both datasets.
    names = radiomic_columns(features)
    dataset_of = readings.drop_duplicates("patch_id").set_index("patch_id")["dataset"]
    reader_rows = features[features["roi"] != "consensus"]
    table = pd.DataFrame({"feature": names})
    nodules = {}
    for dataset in ("LNDb", "LIDC"):
        first, second = first_two_readers(reader_rows[reader_rows["patch_id"].map(dataset_of) == dataset], names)
        table[f"icc_{dataset}"] = [icc_2_1(np.column_stack([first[:, i], second[:, i]])) for i in range(len(names))]
        nodules[dataset] = len(first)
    table["image"] = table["feature"].str.split("_").str[0]
    table["feature_class"] = table["feature"].str.split("_").str[1]
    table["robust"] = (table["icc_LNDb"] >= ROBUST_ICC) & (table["icc_LIDC"] >= ROBUST_ICC)
    table.to_csv(ROI_ICC_FILE, index=False)
    return table, nodules
