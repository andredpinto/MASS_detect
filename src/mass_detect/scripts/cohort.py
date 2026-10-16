"""The study cohort: which nodules enter the models, and their labels.

The label of a nodule is the median of its radiologists' ratings rounded half up, on the 1 to 5 scale, and the binary
scale derived from it (suspicious when the rating is 4 or 5). The mean and the mode of the ratings are kept as well,
only to compare ways of combining the radiologists.
"""
import numpy as np
import pandas as pd
from .config import LIDC_MANIFEST_FILE, NODULE_TABLE

def rounded_rating(rating): # A rating rounded half up, so 2.5 becomes 3 (the mean, the median and the mode can end in .5)
    return np.floor(np.asarray(rating, dtype=float) + 0.5).astype(int)

def median_rating(ratings): # Median of the ratings of one nodule; with an even number of readers it can end in .5
    ratings = np.asarray(ratings, dtype=float)
    ratings = ratings[~np.isnan(ratings)]                  # ratings left blank do not count
    return float(np.median(ratings)) if len(ratings) else np.nan

def mode_rating(ratings): # The most frequent rating of one nodule; on a tie, the median of the tied ratings
    ratings = np.asarray(ratings, dtype=float)
    ratings = ratings[~np.isnan(ratings)]
    if not len(ratings):
        return np.nan
    values, counts = np.unique(ratings, return_counts=True)
    return float(np.median(values[counts == counts.max()]))   # two readers who disagree, 2 and 4, give 3

def suspicious(rating): # Binary scale from a rounded 1 to 5 rating: 1 for ratings 4 and 5, 0 for ratings 1 to 3
    return (np.asarray(rating) >= 4).astype(int)

def lndb_patch_id(nodule_uid): # LNDb-0001_7 becomes LNDb-0001_f007, the name of its patch file
    ct, finding = nodule_uid.rsplit("_", 1)
    return f"{ct}_f{int(finding):03d}"

def select_cohort(table=None, lidc_manifest=None): # Applies the inclusion rules one after the other.
    # Returns the selected nodules and a table that counts, per dataset, how many nodules are left after each rule.
    table = pd.read_csv(NODULE_TABLE) if table is None else table
    lidc_manifest = pd.read_csv(LIDC_MANIFEST_FILE) if lidc_manifest is None else lidc_manifest
    verified = lidc_manifest.loc[lidc_manifest["position_verified"], "patch_id"]

    rules = {
        "all findings": pd.Series(True, index=table.index),
        "is a nodule": table["is_nodule"] == 1,
        "has a malignancy rating": table["mal_median"].notna(),
        "has a radiologist mask (3 mm or more)": ~table["volume_is_placeholder"].astype(bool),
        "at most four annotations (LIDC)": ~table["oversized_group"].fillna(False).astype(bool),
        "position confirmed in its patch (LIDC)": (table["dataset"] == "LNDb") | table["nodule_uid"].isin(verified),
    }
    keep = pd.Series(True, index=table.index)
    funnel = {}
    for name, rule in rules.items():
        keep &= rule
        funnel[name] = table[keep].groupby("dataset").size()
    funnel = pd.DataFrame(funnel).T.fillna(0).astype(int)
    funnel["total"] = funnel.sum(axis=1)

    cohort = table[keep].copy()
    cohort["patch_id"] = [lndb_patch_id(u) if d == "LNDb" else u for u, d in zip(cohort["nodule_uid"], cohort["dataset"])]
    cohort["rating"] = rounded_rating(cohort["mal_median"])          # the label: the median, rounded half up
    cohort["suspicious"] = suspicious(cohort["rating"])              # the binary scale
    cohort["rating_mean"] = rounded_rating(cohort["mal_mean"])       # the other two ways of combining the readers,
    cohort["rating_mode"] = rounded_rating(cohort["mal_mode"])       # kept only to compare them with the median
    return cohort.reset_index(drop=True), funnel
