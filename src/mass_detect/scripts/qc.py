"""Quality checks of the patches: coordinates against masks and the visual figure."""
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from .config import CENTRE_INDEX, CUBE_MM, LNDB_DIR, MASK_FILE, PATCH_SIZE, PATCHES_DIR, QC_FIGURE_FILE, SPACING_MM
from .patches import extract_patch, find_ct_files

PLACEHOLDER_VOLUME = 4 * np.pi * 1.5 ** 3 / 3        # centroid only findings: no mask
CTS_PER_GROUP = 6

def tilt_degrees(path): # Total rotation of the direction matrix of a CT, read from the header only.
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    direction = np.array(reader.GetDirection()).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip((np.trace(direction) - 1) / 2, -1, 1))))

def choose_cts(readings, ct_files): # Six tilted CTs and six untilted ones that have at least one reading with a mask.
    with_mask = sorted(readings["LNDbID"].unique())
    tilt = {ct: tilt_degrees(ct_files[ct]) for ct in with_mask}
    rng = random.Random(0)
    tilted = rng.sample([ct for ct in with_mask if tilt[ct] > 3], CTS_PER_GROUP)
    flat = rng.sample([ct for ct in with_mask if tilt[ct] < 0.5], CTS_PER_GROUP)
    return tilt, tilted, flat

def check_patch_coordinates():
    ct_files = find_ct_files()
    readings = pd.read_csv(LNDB_DIR / "LNDb_csvs" / "trainNodules.csv")
    readings = readings[(readings["Nodule"] == 1) & (readings["Volume"] > PLACEHOLDER_VOLUME + 0.1)]

    tilt, tilted, flat = choose_cts(readings, ct_files)
    rows = []
    for group, cts in (("tilted", tilted), ("not tilted", flat)):
        for ct in cts:
            for reading in readings[readings["LNDbID"] == ct].itertuples():
                mask = sitk.ReadImage(str(MASK_FILE).format(ct=ct, rad=reading.RadID))
                patch, _ = extract_patch(mask, (reading.x, reading.y, reading.z),
                                         interpolator=sitk.sitkNearestNeighbor, outside_value=0)
                inside = np.argwhere(patch == reading.FindingID)       # voxels of this finding, (z, y, x)
                if len(inside) == 0:
                    rows.append({"group": group, "ct": ct, "tilt": tilt[ct], "distance_mm": np.nan, "voxels": 0})
                    continue
                distance = np.linalg.norm(inside.mean(axis=0) - CENTRE_INDEX) * SPACING_MM   # in mm
                rows.append({"group": group, "ct": ct, "tilt": tilt[ct], "distance_mm": distance, "voxels": len(inside)})

    result = pd.DataFrame(rows)
    print(f"{len(result)} readings with a mask, from {result['ct'].nunique()} CTs "
          f"(tilt of the chosen CTs: {result.groupby('group')['tilt'].agg(['min', 'max']).round(1).to_dict('index')})")
    print(f"readings whose mask is empty inside the patch: {int(result['distance_mm'].isna().sum())}")
    valid = result.dropna(subset=["distance_mm"])
    quantiles = valid.groupby("group")["distance_mm"].quantile([0.5, 0.9, 1.0]).unstack().round(2)
    quantiles.columns = ["median", "90th percentile", "max"]
    print("\ndistance between the mask centroid and the centre voxel (mm):")
    print(quantiles.to_string())
    for limit in (1, 2, 5):
        print(f"share below {limit} mm: {(valid['distance_mm'] < limit).mean():.1%}")
    print("\nlargest distances:")
    print(valid.nlargest(5, "distance_mm").round(2).to_string(index=False))

PATCHES_PER_SPLIT = 6
WINDOW_LOW, WINDOW_HIGH = -1350, 150          # lung window: level -600, width 1500

def pick(split, rng):
    manifest = pd.read_csv(PATCHES_DIR / f"manifest_{split}.csv")
    ok = manifest[manifest["status"] == "ok"]
    return ok.iloc[rng.choice(len(ok), size=min(PATCHES_PER_SPLIT, len(ok)), replace=False)]

def draw_patch(axes, row):
    patch = np.load(PATCHES_DIR / row.file)
    views = [("axial", patch[CENTRE_INDEX, :, :], "upper"),       # (y, x)
             ("coronal", patch[:, CENTRE_INDEX, :], "lower"),     # (z, x), head up
             ("sagittal", patch[:, :, CENTRE_INDEX], "lower")]    # (z, y), head up
    for axis, (name, image, origin) in zip(axes, views):
        axis.imshow(image, cmap="gray", vmin=WINDOW_LOW, vmax=WINDOW_HIGH, origin=origin)
        axis.plot(CENTRE_INDEX, CENTRE_INDEX, marker="+", color="red", markersize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(name, fontsize=8)
    centre_hu = patch[CENTRE_INDEX - 1:CENTRE_INDEX + 2, CENTRE_INDEX - 1:CENTRE_INDEX + 2, CENTRE_INDEX - 1:CENTRE_INDEX + 2].mean()
    axes[0].set_ylabel(f"{row.split} {row.patch_id}\noutside {row.outside_fraction:.2f}, centre {centre_hu:.0f} HU", fontsize=7)

def make_patch_qc_figure():
    rng = np.random.default_rng(0)
    rows = [row for split in ("train", "lidc") for row in pick(split, rng).itertuples()]   # six LNDb and six LIDC IDRI patches
    fig, axes = plt.subplots(len(rows) // 2, 6, figsize=(12, 2.1 * len(rows) // 2))
    for number, row in enumerate(rows):
        line, start = divmod(number, 2)
        draw_patch(axes[line, start * 3:start * 3 + 3], row)
    fig.suptitle(f"Random patches ({CUBE_MM:.0f} mm cube), centre marked in red, lung window for display only", fontsize=10)
    plt.tight_layout()
    QC_FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(QC_FIGURE_FILE, dpi=110)
    print(f"saved {QC_FIGURE_FILE}")

def check_patch_files(manifest): # The checklist of a correct CT patch, applied to every patch of a manifest; one row per patch, True when it passes
    rows = []
    for row in manifest[manifest["status"] == "ok"].itertuples():
        patch = np.load(PATCHES_DIR / row.file, mmap_mode="r")
        sample = np.asarray(patch[::4, ::4, ::4])                    # enough voxels to judge the content, and fast
        rows.append({"patch_id": row.patch_id,
                     "stored as int16": patch.dtype == np.int16,
                     "shape of the configured cube": patch.shape == (PATCH_SIZE,) * 3,
                     "looks like a CT, not a mask": len(np.unique(sample)) > 50 and float(sample.std()) > 50,
                     "contains lung air (HU of -700 or less)": int(patch.min()) <= -700,
                     "contains tissue (HU of 0 or more)": int(patch.max()) >= 0,
                     "mostly inside the scan": row.outside_fraction < 0.5})
    return pd.DataFrame(rows)
