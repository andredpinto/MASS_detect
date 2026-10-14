"""How the geometry of the patch affects the radiologists' regions and their features: cube size and voxel size.

Cube size: how far the radiologists' regions reach from the centre, read from the ROI files of notebook 05.
Voxel size: the CT and the radiologists' masks are cut again from the original scans at several voxel sizes, keeping
the cube at 64 mm and every PyRadiomics setting unchanged, and the sizes are compared on four criteria:
coverage, robustness between readers, dependence on the acquisition and prediction.
"""
from multiprocessing import Pool
import time
import SimpleITK as sitk
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import AIR_HU, CENTRE_INDEX, CUBE_MM, MASK_FILE, PATCH_SIZE, ROI_DIR, SENSITIVITY_DIR, SPACING_MM
from .features import HU_RANGE
from .lidc import import_pylidc
from .models import CorrelationFilter, OrdinalLogisticRegression
from .patches import find_ct_files, read_series
from .radiomics_features import SETTINGS, icc_2_1, make_extractor
from .rois import CONSENSUS_LEVEL, patch_centres

SPACINGS_MM = [0.6, 1.0, 1.5, 2.0]     # voxel sizes compared (1 mm is the one used everywhere else)
LOG_MARGIN_MM = 9.0                    # 3 sigma of the widest LoG filter (sigma 3 mm): context the filter needs around the region
SPACING_DIR = SENSITIVITY_DIR / "extraction"   # one csv per scan, so that a long run can be stopped and resumed
RESULTS_DIR = SENSITIVITY_DIR / "results"       # the four result tables, read by notebook 05
ICC_SAMPLE_LIDC = 400                  # LIDC nodules with two readers used for the ICC (every LNDb one is used)

# Cube size

def cube_extents(cohort): # Distance in mm from the patch centre to the farthest voxel of each nodule's consensus region
    rows = []
    for patch_id in cohort["patch_id"]:
        rois = np.load(ROI_DIR / f"{patch_id}.npz")
        voxels = np.argwhere(rois["consensus"])
        if len(voxels) == 0:
            rows.append({"patch_id": patch_id})
            continue
        half_extent = (np.abs(voxels - CENTRE_INDEX).max() + 0.5) * SPACING_MM   # mm from the centre to the outer edge of the last voxel
        touches = bool(voxels.min() == 0 or voxels.max() == PATCH_SIZE - 1)
        rows.append({"patch_id": patch_id, "half_extent_mm": half_extent, "touches_border": touches})
    return cohort[["patch_id", "dataset", "diameter_mm"]].merge(pd.DataFrame(rows), on="patch_id")

def cube_fit_table(extents, sides=(30, 32, 40, 51, 57, 64)): # Per cube side: regions that would be cut, and regions without the LoG margin
    rows = {}
    for side in sides:
        cut = extents["half_extent_mm"] > side / 2
        no_margin = extents["half_extent_mm"] + LOG_MARGIN_MM > side / 2
        rows[f"{side} mm"] = {**{f"cut {d}": int(cut[extents.dataset == d].sum()) for d in ("LNDb", "LIDC")},
                              **{f"no LoG margin {d}": int(no_margin[extents.dataset == d].sum()) for d in ("LNDb", "LIDC")}}
    return pd.DataFrame(rows).T

# Voxel size: cutting and extracting again

def cut_cube(image, centre_mm, spacing, interpolator=sitk.sitkLinear, outside_value=AIR_HU): # 64 mm cube around a world point, voxels of 'spacing' mm
    size = int(round(CUBE_MM / spacing))
    centre_index = size // 2
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([size] * 3)
    resampler.SetOutputSpacing([spacing] * 3)
    resampler.SetOutputDirection(tuple(np.eye(3).ravel()))
    resampler.SetOutputOrigin(tuple(np.asarray(centre_mm, dtype=float) - centre_index * spacing))
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetDefaultPixelValue(float("nan"))                 # marks the voxels outside the CT
    cube = sitk.GetArrayFromImage(resampler.Execute(image))
    cube[np.isnan(cube)] = outside_value
    return cube

_EXTRACTOR = None

def region_features(patch, mask, spacing): # PyRadiomics of one region; the image carries its real voxel size so shape and LoG stay in mm
    global _EXTRACTOR
    _EXTRACTOR = _EXTRACTOR or make_extractor()                  # one extractor per worker process
    if mask.sum() < SETTINGS["minimumROISize"]:
        return {"error": "fewer than 8 voxels"}
    image = sitk.GetImageFromArray(np.clip(np.rint(patch), *HU_RANGE).astype(np.float32))
    region = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetSpacing([spacing] * 3)
    region.SetSpacing([spacing] * 3)
    try:
        values = _EXTRACTOR.execute(image, region)
        return {name: float(value) for name, value in values.items() if not name.startswith("diagnostics")}
    except Exception as error:      # one failing region must not stop the run
        return {"error": str(error)[:100]}

def lidc_annotation_image(annotation, image): # The pylidc mask of one annotation as an image placed in the geometry of the CT
    box_mask = annotation.boolean_mask()                          # (row, column, slice), inside the annotation's bounding box
    row, column, slice_index = (s.start for s in annotation.bbox())
    box = sitk.GetImageFromArray(np.transpose(box_mask, (2, 0, 1)).astype(np.uint8))
    box.SetSpacing(image.GetSpacing())
    box.SetDirection(image.GetDirection())
    box.SetOrigin(image.TransformIndexToPhysicalPoint((int(column), int(row), int(slice_index))))
    return box

def _scan_job(job): # Every nodule of one scan at every voxel size; writes SPACING_DIR/<scan>.csv
    dataset, scan, nodules, readings, centres, icc_ids, ct_path, out_dir, spacings = job
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)       # the parallelism comes from the processes
    try:
        if dataset == "LNDb":
            image, mask_images = sitk.ReadImage(str(ct_path)), {}
        else:
            pl = import_pylidc()
            image = read_series(nodules["patient_id"].iloc[0], scan)
        rows = []
        for nodule in nodules.itertuples():
            mine = readings[readings["nodule_uid"] == nodule.nodule_uid].sort_values("reader")
            sources = []                                          # (reader, image that holds the mask, label of the nodule in it)
            for reading in mine.itertuples():
                if dataset == "LNDb":
                    if reading.reader not in mask_images:
                        path = str(MASK_FILE).format(ct=int(scan.split("-")[1]), rad=reading.reader.replace("rad", ""))
                        mask_images[reading.reader] = sitk.ReadImage(path)
                    sources.append((reading.reader, mask_images[reading.reader], int(reading.mask_label)))
                else:
                    annotation = pl.query(pl.Annotation).filter(pl.Annotation.id == int(reading.mask_label)).first()
                    sources.append((reading.reader, lidc_annotation_image(annotation, image), 1))
            for spacing in spacings:
                centre = centres[nodule.patch_id]
                patch = cut_cube(image, centre, spacing)
                masks = [cut_cube(img, centre, spacing, sitk.sitkNearestNeighbor, 0) == label for _, img, label in sources]
                regions = [("consensus", np.mean(masks, axis=0) >= CONSENSUS_LEVEL)]
                if nodule.patch_id in icc_ids:                    # the first two readers, for the ICC
                    regions += [(sources[k][0], masks[k]) for k in range(2)]
                for roi, mask in regions:
                    rows.append({"patch_id": nodule.patch_id, "spacing": spacing, "roi": roi, "voxels": int(mask.sum()),
                                 **region_features(patch, mask, spacing)})
        pd.DataFrame(rows).to_csv(out_dir / f"{scan}.csv", index=False)
        return scan, ""
    except Exception as error:      # one failing scan must not stop the run; it is reported at the end
        return scan, str(error)[:200]

def icc_sample(cohort, readings, seed=0): # Nodules whose first two readers enter the ICC: every LNDb one and a random LIDC sample
    readers = readings.groupby("patch_id").size()
    multi = cohort[cohort["patch_id"].isin(readers[readers >= 2].index)]
    lidc = multi.loc[multi["dataset"] == "LIDC", "patch_id"]
    chosen = np.random.default_rng(seed).choice(lidc, min(ICC_SAMPLE_LIDC, len(lidc)), replace=False)
    return set(multi.loc[multi["dataset"] == "LNDb", "patch_id"]) | set(chosen)

def extract_at_spacings(cohort, readings, spacings=SPACINGS_MM, out_dir=SPACING_DIR, processes=10, limit_scans=None):
    # Cuts and extracts every cohort nodule at every voxel size. Scans already on disk are skipped, so the run can be resumed.
    out_dir.mkdir(parents=True, exist_ok=True)
    centres, ct_files, icc_ids = patch_centres(), find_ct_files(), icc_sample(cohort, readings)
    jobs = []
    for (dataset, scan), nodules in cohort.groupby(["dataset", "scan_id"]):
        if (out_dir / f"{scan}.csv").exists():
            continue
        ct_path = ct_files.get(int(scan.split("-")[1])) if dataset == "LNDb" else None
        jobs.append((dataset, scan, nodules, readings[readings["nodule_uid"].isin(nodules["nodule_uid"])],
                     {p: centres[p] for p in nodules["patch_id"]}, icc_ids & set(nodules["patch_id"]), ct_path, out_dir, spacings))
    jobs = jobs[:limit_scans] if limit_scans else jobs
    started, failures = time.time(), []
    with Pool(processes, maxtasksperchild=20) as pool:
        for number, (scan, error) in enumerate(pool.imap_unordered(_scan_job, jobs), start=1):
            if error:
                failures.append((scan, error))
            if number % 50 == 0:
                print(f"{number}/{len(jobs)} scans, {time.time() - started:.0f} s", flush=True)
    print(f"{len(jobs)} scans processed in {time.time() - started:.0f} s; failures: {failures}")

def load_spacing_features(out_dir=SPACING_DIR): # Every region at every voxel size, from the per scan files
    return pd.concat([pd.read_csv(path, low_memory=False) for path in sorted(out_dir.glob("*.csv"))], ignore_index=True)

# Voxel size: the four criteria

def feature_columns(table): # The PyRadiomics columns: original image and Laplacian of Gaussian images
    return [c for c in table.columns if c.startswith(("original_", "log-"))]

def consensus_rows(table, cohort): # Consensus regions with the nodule's information and a flag for 'has features'
    names = feature_columns(table)
    rows = table[table["roi"] == "consensus"].merge(
        cohort[["patch_id", "dataset", "patient_id", "rating", "diameter_mm", "slice_thickness_mm", "pixel_spacing_mm"]], on="patch_id")
    rows["has_features"] = rows[names].notna().all(axis=1)
    return rows

def common_nodules(rows): # Nodules with features at every voxel size: the fair common ground for criteria 2 to 4
    sizes = rows[rows["has_features"]].groupby("patch_id")["spacing"].nunique()
    return sizes[sizes == rows["spacing"].nunique()].index

def coverage_table(rows): # 1. Share of consensus regions that still have features, per dataset and per nodule size
    rows = rows.assign(size=pd.cut(rows["diameter_mm"], [0, 5, 10, 100], labels=["under 5 mm", "5 to 10 mm", "10 mm or more"]))
    by_dataset = rows.pivot_table(index="spacing", columns="dataset", values="has_features", aggfunc="mean")
    by_size = rows.pivot_table(index="spacing", columns="size", values="has_features", aggfunc="mean", observed=False)
    return by_dataset.join(by_size).round(3)

def robustness_table(table, cohort): # 2. ICC(2,1) between the first two readers, per voxel size and dataset
    names = feature_columns(table)
    readers = table[(table["roi"] != "consensus") & table[names].notna().all(axis=1)]
    pairs = readers.groupby(["patch_id", "spacing"]).size()
    complete = pairs[pairs == 2].reset_index().groupby("patch_id")["spacing"].nunique()
    ids = complete[complete == table["spacing"].nunique()].index
    dataset_of = cohort.set_index("patch_id")["dataset"]
    rows = []
    for spacing in sorted(table["spacing"].unique()):
        for dataset in ("LNDb", "LIDC"):
            mine = readers[(readers["spacing"] == spacing) & readers["patch_id"].isin(ids)
                           & (readers["patch_id"].map(dataset_of) == dataset)].sort_values(["patch_id", "roi"])
            if mine.empty:
                continue
            first = mine.groupby("patch_id").nth(0).set_index("patch_id")[names]
            second = mine.groupby("patch_id").nth(1).set_index("patch_id").loc[first.index, names]
            icc = np.array([icc_2_1(np.column_stack([first[f], second[f]])) for f in names])
            rows.append({"spacing": spacing, "dataset": dataset, "nodules": len(first),
                         "median ICC": np.nanmedian(icc), "share with ICC of 0.75 or more": np.nanmean(icc >= 0.75)})
    return pd.DataFrame(rows).pivot(index="spacing", columns="dataset").round(3)

def partial_spearman(x, z, control): # Spearman correlation of x and z after removing, from both, their relation with 'control'
    ranks = pd.DataFrame({"x": x, "z": z, "c": control}).rank()
    residual = lambda a: a - np.polyval(np.polyfit(ranks["c"], a, 1), ranks["c"])
    return np.corrcoef(residual(ranks["x"]), residual(ranks["z"]))[0, 1]

def acquisition_table(rows, names): # 3. How much the features follow the scan instead of the nodule, given its diameter (lower is better)
    common = common_nodules(rows)
    result = []
    for spacing, group in rows[rows["patch_id"].isin(common)].groupby("spacing"):
        for dataset, acquisition in (("LIDC", "slice_thickness_mm"), ("LNDb", "pixel_spacing_mm")):
            mine = group[group["dataset"] == dataset]
            if len(mine) < 20 or mine[acquisition].nunique() < 2:
                continue
            rho = np.abs([partial_spearman(mine[f], mine[acquisition], np.log(mine["diameter_mm"])) for f in names])
            result.append({"spacing": spacing, "check": f"{dataset}: {acquisition}", "median |rho|": np.nanmedian(rho),
                           "share with |rho| above 0.2": np.nanmean(rho > 0.2)})
    return pd.DataFrame(result).pivot(index="spacing", columns="check").round(3)

def _outer_fold(job): # One outer fold of the nested cross validation: tunes alpha on the training part, predicts the held out part
    spacing, seed, train, test, X, y, groups = job
    model = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(), StandardScaler(),
                          CorrelationFilter(0.9), OrdinalLogisticRegression())
    search = GridSearchCV(model, {"ordinallogisticregression__alpha": [1.0, 10.0, 100.0]},
                          scoring=make_scorer(cohen_kappa_score, weights="quadratic"),
                          cv=StratifiedGroupKFold(3, shuffle=True, random_state=seed))
    search.fit(X[train], y[train], groups=groups[train])
    return spacing, seed, test, search.predict(X[test])

def prediction_table(rows, names, spacings=None, seeds=(0, 1, 2), processes=10, n_boot=300):
    # 4. Nested grouped cross validation of the ordinal model on the common nodules, LNDb and LIDC together.
    # Returns the mean QWK per dataset and their average (the primary metric), and the difference of each size to 1 mm.
    # 'spacings' limits the comparison to some sizes, so that the common nodules are not reduced by the coarsest one.
    if spacings is not None:
        rows = rows[rows["spacing"].isin(spacings)]
    common = common_nodules(rows)
    info = rows.drop_duplicates("patch_id").set_index("patch_id").loc[common]
    y, groups, dataset = info["rating"].to_numpy(), info["patient_id"].to_numpy(), info["dataset"].to_numpy()
    spacings = sorted(rows["spacing"].unique())
    jobs = []
    for spacing in spacings:
        X = rows[rows["spacing"] == spacing].set_index("patch_id").loc[common, names].to_numpy(float)
        for seed in seeds:
            for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
                jobs.append((spacing, seed, train, test, X, y, groups))
    with Pool(processes) as pool:
        results = pool.map(_outer_fold, jobs)
    predictions = {}
    for spacing, seed, test, predicted in results:
        predictions.setdefault((spacing, seed), np.zeros(len(y), dtype=int))[test] = predicted

    def score(predicted, index): # QWK in each dataset and their average
        q = {d: cohen_kappa_score(y[index][dataset[index] == d], predicted[index][dataset[index] == d], weights="quadratic")
             for d in np.unique(dataset[index])}
        return q, np.mean(list(q.values()))

    everyone = np.arange(len(y))
    table = {}
    for spacing in spacings:
        per_seed = [score(predictions[(spacing, s)], everyone) for s in seeds]
        table[spacing] = {**{f"QWK {d}": np.mean([q[d] for q, _ in per_seed]) for d in per_seed[0][0]},
                          "mean QWK": np.mean([m for _, m in per_seed]), "sd over seeds": np.std([m for _, m in per_seed])}
    table = pd.DataFrame(table).T

    # paired difference to 1 mm, resampling whole patients (cluster bootstrap)
    rng, patients = np.random.default_rng(0), np.unique(groups)
    rows_of = {p: np.where(groups == p)[0] for p in patients}
    for spacing in spacings:
        if spacing == 1.0:
            continue
        differences = []
        for _ in range(n_boot):
            index = np.concatenate([rows_of[p] for p in rng.choice(patients, len(patients))])
            differences.append(np.mean([score(predictions[(spacing, s)], index)[1] - score(predictions[(1.0, s)], index)[1] for s in seeds]))
        table.loc[spacing, "difference to 1 mm"] = np.mean(differences)
        table.loc[spacing, "95% interval low"] = np.percentile(differences, 2.5)
        table.loc[spacing, "95% interval high"] = np.percentile(differences, 97.5)
    return table.round(3), len(common)
