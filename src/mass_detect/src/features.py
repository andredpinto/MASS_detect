"""Segmentation from the centre and the handcrafted features measured in a patch."""
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from skimage.measure import mesh_surface_area
import numpy as np
import pandas as pd
import time
from .config import CENTRE_INDEX, FEATURES_FILE, PATCHES_DIR

SMOOTHING_SIGMA_MM = 0.7
CENTRE_RADIUS_MM = 2          # HU at the centre is the median inside this radius
LUNG_SHELL_MM = (24, 31)      # distances from the centre where we measure the surrounding lung
MAX_EXTENT_MM = 28            # nothing farther than this from the centre can belong to the nodule
INSCRIBED_SEARCH_MM = 1.5     # the widest sphere is searched this close to the centre
GROWTH = 1.2                  # the region is cut at GROWTH x (radius of the widest sphere) + PADDING_MM
PADDING_MM = 1.0
RING_WIDTH_MM = 3
SOLID_HU = -300               # above this we call the tissue solid, below (down to -750) ground glass
GROUND_GLASS_LOW_HU = -750
HU_RANGE = (-1024, 3071)      # range of a 12 bit CT; anything outside is padding or artefact
SPHERE_RADII_MM = (3, 5, 8)   # sphere features: the smallest LNDb nodules, a typical one, a large one
_z, _y, _x = np.indices((64, 64, 64))
DISTANCE_TO_CENTRE = np.sqrt((_z - CENTRE_INDEX) ** 2 + (_y - CENTRE_INDEX) ** 2 + (_x - CENTRE_INDEX) ** 2)

def segment_from_centre(patch): # Returns (mask, info). The mask is empty when there is no nodule like structure at the centre.
    hu = np.clip(patch.astype(np.float32), -1024, 400)
    smooth = ndi.gaussian_filter(hu, SMOOTHING_SIGMA_MM)
    centre_hu = float(np.median(smooth[DISTANCE_TO_CENTRE <= CENTRE_RADIUS_MM]))
    shell = (DISTANCE_TO_CENTRE >= LUNG_SHELL_MM[0]) & (DISTANCE_TO_CENTRE <= LUNG_SHELL_MM[1]) & (smooth < -500)
    lung_hu = float(np.median(smooth[shell])) if shell.sum() > 200 else -850.0
    threshold = min((centre_hu + lung_hu) / 2, SOLID_HU)
    info = {"centre_hu": centre_hu, "lung_hu": lung_hu, "threshold_hu": threshold}
    empty = np.zeros(patch.shape, dtype=bool)
    if centre_hu < lung_hu + 150: return empty, info  # the centre is not denser than the lung

    region = touching_centre(smooth > threshold, DISTANCE_TO_CENTRE <= MAX_EXTENT_MM)
    if region is None:
        return empty, info
    inside_distance = ndi.distance_transform_edt(region)
    near_centre = np.where(DISTANCE_TO_CENTRE <= INSCRIBED_SEARCH_MM, inside_distance, 0)
    widest = np.unravel_index(near_centre.argmax(), region.shape)
    radius = inside_distance[widest]
    distance_to_widest = np.sqrt((_z - widest[0]) ** 2 + (_y - widest[1]) ** 2 + (_x - widest[2]) ** 2)
    cut = region & (distance_to_widest <= GROWTH * radius + PADDING_MM)
    labels, _ = ndi.label(cut)
    return (labels == labels[widest]) if labels[widest] > 0 else cut, info

def touching_centre(mask, allowed): # The connected region of mask & allowed that touches the centre, or None.
    labels, _ = ndi.label(mask & allowed)
    near = labels[DISTANCE_TO_CENTRE <= 3]
    near = near[near > 0]
    return None if len(near) == 0 else labels == np.bincount(near).argmax()

def sphere_features(clipped): # HU statistics in spheres centred on the finding. They need no mask, so every finding has them.
    features = {}
    for radius in SPHERE_RADII_MM:
        values = clipped[DISTANCE_TO_CENTRE <= radius]
        features[f"sphere{radius}_mean_hu"] = float(values.mean())
        features[f"sphere{radius}_p90_hu"] = float(np.percentile(values, 90))
        features[f"sphere{radius}_solid_fraction"] = float((values > SOLID_HU).mean())
    return features

def measure(patch): # Segments the patch and returns a dict with the features of the nodule.
    mask, info = segment_from_centre(patch)                  # the segmentation clips its own copy to [-1024, 400]
    clipped = np.clip(patch.astype(np.float32), *HU_RANGE)   # every intensity feature uses the clipped patch
    features = {**info, "segmented": bool(mask.any()), **sphere_features(clipped)}
    names = ["volume_mm3", "diameter_mm", "max_extent_mm", "elongation", "flatness", "sphericity",
             "hu_mean", "hu_std", "hu_p10", "hu_p50", "hu_p90", "hu_skew", "solid_fraction", "ground_glass_fraction",
             "edge_gradient", "ring_mean_hu", "ring_solid_fraction", "contrast_hu"]
    if not mask.any() or mask.sum() < 8:
        return {**features, **dict.fromkeys(names, np.nan)}

    points = np.argwhere(mask).astype(float)
    volume = float(mask.sum())
    axes = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(points.T)), 1e-9))[::-1]       # large to small
    area = surface_area(mask)
    hu = clipped[mask].astype(float)
    ring = ndi.binary_dilation(mask, iterations=RING_WIDTH_MM) & ~ndi.binary_dilation(mask, iterations=1)
    ring_hu = clipped[ring].astype(float)
    edge = mask & ~ndi.binary_erosion(mask)
    gradient = ndi.gaussian_gradient_magnitude(clipped, SMOOTHING_SIGMA_MM)
    return {**features,
            "volume_mm3": volume,
            "diameter_mm": 2 * (3 * volume / (4 * np.pi)) ** (1 / 3),
            "max_extent_mm": float(np.ptp(points @ axis_of_largest_spread(points))) + 1,
            "elongation": axes[1] / axes[0], "flatness": axes[2] / axes[0],
            "sphericity": np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area,
            "hu_mean": hu.mean(), "hu_std": hu.std(), "hu_p10": np.percentile(hu, 10),
            "hu_p50": np.percentile(hu, 50), "hu_p90": np.percentile(hu, 90),
            "hu_skew": float(((hu - hu.mean()) ** 3).mean() / (hu.std() ** 3 + 1e-9)),
            "solid_fraction": (hu > SOLID_HU).mean(),
            "ground_glass_fraction": ((hu > GROUND_GLASS_LOW_HU) & (hu <= SOLID_HU)).mean(),
            "edge_gradient": float(gradient[edge].mean()),
            "ring_mean_hu": ring_hu.mean() if len(ring_hu) else np.nan,
            "ring_solid_fraction": (ring_hu > SOLID_HU).mean() if len(ring_hu) else np.nan,
            "contrast_hu": hu.mean() - (ring_hu.mean() if len(ring_hu) else np.nan)}

def axis_of_largest_spread(points): # Unit vector of the direction in which the points are most spread out.
    return np.linalg.eigh(np.cov(points.T))[1][:, -1]

def surface_area(mask): # Area in mm2 of the surface of the mask (marching cubes on the padded mask).
    vertices, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), level=0.5)
    return float(mesh_surface_area(vertices, faces))

def measure_file(path):
    try: return measure(np.load(PATCHES_DIR / path))
    except Exception as error: return {"segmented": False, "error": str(error)}

def measure_all_patches(): # LNDb train and test, plus the LIDC patches when they have been cut (see lidc_patches.py)
    splits = [s for s in ("train", "test", "lidc") if (PATCHES_DIR / f"manifest_{s}.csv").exists()]
    manifest = pd.concat([pd.read_csv(PATCHES_DIR / f"manifest_{s}.csv") for s in splits], ignore_index=True)
    ok = manifest[manifest["status"] == "ok"]
    started = time.time()
    with Pool(6) as pool:
        rows = pool.map(measure_file, ok["file"].tolist(), chunksize=20)
    features = pd.concat([ok[["patch_id", "split"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    features.to_csv(FEATURES_FILE, index=False)
    print(f"{len(features)} patches measured in {time.time() - started:.0f} s, saved {FEATURES_FILE}")
    print("segmented:", features.groupby("split")["segmented"].mean().round(3).to_dict())
