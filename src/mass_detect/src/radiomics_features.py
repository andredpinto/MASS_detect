"""PyRadiomics extraction with an IBSI oriented configuration, and the ICC robustness analysis."""
from multiprocessing import Pool
from radiomics import featureextractor
import SimpleITK as sitk
import logging
import numpy as np
import pandas as pd
from .config import LNDB_DIR, PATCHES_DIR, REPO_ROOT
from .features import HU_RANGE, segment_from_centre
from .patches import extract_patch

SETTINGS = {"binWidth": 25, "resampledPixelSpacing": None, "interpolator": "sitkLinear", "label": 1,
            "minimumROIDimensions": 1, "minimumROISize": 8, "geometryTolerance": 1e-3}
LOG_SIGMAS_MM = [1.0, 2.0, 3.0]
ROBUST_ICC = 0.75
logging.getLogger("radiomics").setLevel(logging.ERROR)

def make_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor(**SETTINGS)
    extractor.enableAllFeatures()
    extractor.enableImageTypes(Original={}, LoG={"sigma": LOG_SIGMAS_MM})
    return extractor

_EXTRACTOR = None

def extract(patch, mask): # Radiomic features of one region. Returns an empty dict when the region is too small.
    global _EXTRACTOR
    _EXTRACTOR = _EXTRACTOR or make_extractor() # one extractor per worker process
    if mask.sum() < SETTINGS["minimumROISize"]: return {}
    image = sitk.GetImageFromArray(np.clip(patch.astype(np.float32), *HU_RANGE))   # spacing 1 mm by construction
    region = sitk.GetImageFromArray(mask.astype(np.uint8))
    values = _EXTRACTOR.execute(image, region)
    return {name: float(value) for name, value in values.items() if not name.startswith("diagnostics")}

def icc_2_1(ratings): # ICC(2,1) of an (n subjects x k raters) array: two way random effects, absolute agreement, single rater.
    n, k = ratings.shape
    grand = ratings.mean()
    ms_rows = k * ((ratings.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((ratings.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = ratings - ratings.mean(axis=1, keepdims=True) - ratings.mean(axis=0, keepdims=True) + grand
    ms_error = (residual ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)

def _work(item):
    patch_id, roi, patch, mask = item
    try: return {"patch_id": patch_id, "roi": roi, **extract(patch, mask)}
    except Exception as error: return {"patch_id": patch_id, "roi": roi, "error": str(error)} # one failing region must not stop the run

def training_items(): # Automatic mask and every radiologist mask of each rated LNDb training nodule.
    table = pd.read_csv(REPO_ROOT / "notebooks" / "nodule_table.csv")
    rated = table[(table["dataset"] == "LNDb") & table["mal_mean"].notna()]
    merged = pd.read_csv(LNDB_DIR / "LNDb_csvs" / "trainNodules_gt.csv")
    merged["nodule_uid"] = [f"LNDb-{c:04d}_{f}" for c, f in zip(merged["LNDbID"], merged["FindingID"])]
    merged = merged[merged["nodule_uid"].isin(rated["nodule_uid"])]
    for ct_id, findings in merged.groupby("LNDbID"):
        masks = {}
        for f in findings.itertuples():
            patch_id = f"LNDb-{ct_id:04d}_f{f.FindingID:03d}"
            patch = np.load(PATCHES_DIR / "train" / f"{patch_id}.npy")
            yield patch_id, "auto", patch, segment_from_centre(patch)[0]
            for rad, finding in zip(str(f.RadID).split(","), str(f.RadFindingID).split(",")):
                path = LNDB_DIR / "LNDb_masks" / "masks" / f"LNDb-{ct_id:04d}_rad{rad}.mhd"
                if not path.exists(): continue
                masks.setdefault(rad, sitk.ReadImage(str(path)))
                mask_patch, _ = extract_patch(masks[rad], (f.x, f.y, f.z), interpolator=sitk.sitkNearestNeighbor, outside_value=0)
                yield patch_id, f"rad{rad}", patch, mask_patch == int(finding)

def robustness_table(features): # ICC(2,1) of every feature between the first two radiologist masks of the same nodule.
    readers = features[features["roi"].str.startswith("rad")].dropna(axis=1, how="all")
    names = [c for c in readers.columns if c.startswith(("original_", "log-"))]
    readers = readers.dropna(subset=names) # masks under 8 voxels (findings < 3 mm have only a centre) give no features
    pairs =readers.groupby("patch_id").filter(lambda g: len(g) >= 2).groupby("patch_id").head(2).sort_values(["patch_id", "roi"])
    first, second = pairs.groupby("patch_id").nth(0)[names].to_numpy(), pairs.groupby("patch_id").nth(1)[names].to_numpy()
    icc = {name: icc_2_1(np.column_stack([first[:, i], second[:, i]])) for i, name in enumerate(names)}
    table = pd.DataFrame({"feature": list(icc), "icc": list(icc.values())})
    table["image"] = table["feature"].str.split("_").str[0]
    table["feature_class"] = table["feature"].str.split("_").str[1]
    table["robust"] = table["icc"] >= ROBUST_ICC
    return table, len(first)

def extract_training_radiomics(recompute=True):
    if recompute:
        with Pool(4) as pool:
            rows = pool.map(_work, list(training_items()), chunksize=8)
        pd.DataFrame(rows).to_csv(PATCHES_DIR / "radiomics_train.csv", index=False)
    features = pd.read_csv(PATCHES_DIR / "radiomics_train.csv")
    table, n_nodules = robustness_table(features)
    table.to_csv(PATCHES_DIR / "radiomics_icc.csv", index=False)
    print(f"{len(features)} regions; ICC from {n_nodules} nodules with two masks; robust: {table['robust'].mean():.0%}")
