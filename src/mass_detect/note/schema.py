"""The shared column schema of the nodule table and the equivalent diameter."""
import math
import numpy as np

SCHEMA = [
    "dataset",                # LNDb or LIDC
    "scan_id",                # unique CT id: LNDb-0001, or the series UID in LIDC
    "patient_id",             # grouping key for validation (in LNDb we assume one CT per patient)
    "nodule_uid",             # unique nodule id across both datasets
    "x_mm", "y_mm", "z_mm",   # nodule centre in world coordinates (mm), in each dataset's own CT frame
    "diameter_mm",            # diameter of the sphere with the same volume as the segmentation
    "volume_mm3",             # mean volume of the segmentation
    "volume_is_placeholder",  # LNDb: 3 mm sphere stored when radiologists only marked a centre
    "is_nodule",              # 1 nodule, 0 non nodule (LIDC only contains nodules of 3 mm or more)
    "mal_mean", "mal_min", "mal_max",  # malignancy ratings from 1 to 5, NaN when unrated
    "n_rated",                # readings that carry a malignancy rating
    "n_readings",             # readings merged into this nodule
    "reclustered",            # LIDC: group produced by our proximity rule (see section 0)
    "oversized_group",        # LIDC: still more than 4 annotations after the rule, kept but flagged
    "lobe",                   # LNDb only: 1 LUL, 2 LLL, 3 RUL, 4 ML, 5 RLL
    "slice_thickness_mm", "pixel_spacing_mm", "manufacturer", "model", "kernel", "kvp",
    "vox_row", "vox_col", "vox_slice",  # LIDC only: centre in voxel indices, to cut patches later
]


def equivalent_diameter(volume_mm3):
    """Diameter of the sphere that has the given volume."""
    volume = np.asarray(volume_mm3, dtype=float)
    return 2 * (3 * volume / (4 * math.pi)) ** (1 / 3)
