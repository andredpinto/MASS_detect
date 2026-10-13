"""The study cohort: which nodules enter the models, and their labels on the 1 to 5 and the 1 to 3 scale."""
import numpy as np
import pandas as pd
from .config import LIDC_MANIFEST_FILE, NODULE_TABLE

def rounded_rating(mean_rating): # Mean of the radiologists' ratings rounded half up, so 2.5 becomes 3
    return np.floor(np.asarray(mean_rating, dtype=float) + 0.5).astype(int)

def three_level(rating): # 1 to 3 scale from a 1 to 5 rating: ratings 1 and 2 give 1, rating 3 gives 2, ratings 4 and 5 give 3
    return np.digitize(rating, [2.5, 3.5]) + 1

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
        "has a malignancy rating": table["mal_mean"].notna(),
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
    cohort["rating"] = rounded_rating(cohort["mal_mean"])
    cohort["rating_3"] = three_level(cohort["rating"])
    return cohort.reset_index(drop=True), funnel
