"""From the PyRadiomics features of the radiologists' regions to the tables the models read (notebook 05).

The filters never look at the malignancy rating, so they can be applied to every nodule before the cross validation.
Every choice that uses the rating is made only inside the training folds of notebook 06 (the penalties of the
models, and the elastic net, which sets the coefficients of unhelpful features to zero).
"""
import numpy as np
import pandas as pd
from .config import SPACING_MM
from .radiomics_features import ROBUST_ICC

MAX_MISSING = 0.20         # a feature missing in more regions than this would mostly describe the imputation
MAX_CORRELATION = 0.90     # usual cutoff for redundant radiomic features (Traverso et al., 2018)

def feature_inventory(consensus, names): # Features per image type and class, with how many are missing or constant
    table = pd.DataFrame({"feature": names, "image": [n.split("_")[0] for n in names], "feature_class": [n.split("_")[1] for n in names],
                          "missing_share": consensus[names].isna().mean().to_numpy(),
                          "constant": (consensus[names].nunique() <= 1).to_numpy()})
    return table

def volume_check(consensus, quality): # PyRadiomics volume against the voxels of the mask times the voxel volume: equal when the spacing reached PyRadiomics
    merged = consensus[["patch_id", "original_shape_VoxelVolume"]].merge(quality[["patch_id", "consensus_voxels"]], on="patch_id")
    merged["expected_mm3"] = merged["consensus_voxels"] * SPACING_MM ** 3
    merged["ratio"] = merged["original_shape_VoxelVolume"] / merged["expected_mm3"]
    return merged

def label_free_filter(consensus, icc, names): # Keeps the features that are present, variable, robust between readers and not redundant
    report, kept = [], []
    missing = consensus[names].isna().mean()
    for name in names:                                             # 1. present in enough regions and 2. not constant
        if missing[name] > MAX_MISSING:
            report.append({"feature": name, "removed_by": "missing", "detail": f"missing in {missing[name]:.0%} of the regions"})
        elif consensus[name].nunique() <= 1:
            report.append({"feature": name, "removed_by": "constant", "detail": "one value only"})
        else:
            kept.append(name)
    icc = icc.set_index("feature")
    robust = []
    for name in kept:                                              # 3. robust between readers in both datasets
        lndb, lidc = icc.loc[name, "icc_LNDb"], icc.loc[name, "icc_LIDC"]
        if lndb >= ROBUST_ICC and lidc >= ROBUST_ICC:
            robust.append(name)
        else:
            report.append({"feature": name, "removed_by": "not robust", "detail": f"ICC {lndb:.2f} in LNDb, {lidc:.2f} in LIDC"})
    # 4. not redundant: between two features with absolute Spearman correlation above MAX_CORRELATION, keep the one with fewer
    #    missing values; on a tie, the one from the original image (easier to interpret), then the one most robust between
    #    readers (the higher of the two lowest ICCs), and finally the alphabetical order, so that the result is deterministic
    lowest_icc = icc[["icc_LNDb", "icc_LIDC"]].min(axis=1)
    order = sorted(robust, key=lambda name: (missing[name], not name.startswith("original"), -lowest_icc[name], name))
    correlation = consensus[order].rank().corr().abs()
    final = []
    for name in order:
        partners = [k for k in final if correlation.loc[name, k] > MAX_CORRELATION]
        if partners:
            report.append({"feature": name, "removed_by": "redundant", "detail": f"Spearman {correlation.loc[name, partners[0]]:.2f} with {partners[0]}"})
        else:
            final.append(name)
    return final, pd.DataFrame(report)

def model_tables(cohort, features, readings, kept): # One row per nodule (consensus region) and one row per reading (reader region)
    keys = ["patch_id", "nodule_uid", "dataset", "patient_id", "scan_id", "rating", "suspicious", "rating_mean", "rating_mode", "n_rated",
            "manufacturer", "kernel", "slice_thickness_mm", "pixel_spacing_mm"]
    sizes = ["roi_diameter_mm", "roi_axial_diameter_mm"]
    nodules = cohort[keys].merge(features.loc[features["roi"] == "consensus", ["patch_id"] + sizes + kept], on="patch_id", how="left")
    reader_rows = features[features["roi"] != "consensus"].rename(columns={"roi": "reader"})[["patch_id", "reader"] + sizes + kept]
    reading = (readings[["patch_id", "reader", "rating"]].rename(columns={"rating": "reader_rating"})
               .merge(reader_rows, on=["patch_id", "reader"], how="left")
               .merge(cohort[keys], on="patch_id"))
    reading["weight"] = 1 / reading.groupby("patch_id")["reader"].transform("size")    # each nodule weighs 1 in total
    return nodules, reading
