from src.mass_detect.lndb_utils import utils
from pathlib import Path
import numpy as np

def find_file(name, base_folder="../data"):
    results = list(Path(base_folder).rglob(name))
    if not results:
        raise FileNotFoundError(f"File '{name}' not found in '{base_folder}'.")
    return str(results[0])

def read_nods_in_scan(LNDb_id, nods_table): #Reads one CT Scan and returns array, metadata and scan nodules with converted coordinates.
    path_mhd = find_file(f'LNDb-{LNDb_id:04d}.mhd')
    scan, spacing, origin, transfmat = utils.readMhd(path_mhd) # read metadata from scan, necessary for coordinate conversion
    transfmat_toimg, transfmat_toworld = utils.getImgWorldTransfMats(spacing, transfmat) #
    nods_scan = nods_table[nods_table['LNDbID'] == LNDb_id].copy()

    coords_img = []
    masks_img = []
    for _, line in nods_scan.iterrows():
        xyz_world = np.array([line['x'], line['y'], line['z']])
        xyz_img = utils.convertToImgCoord(xyz_world, origin, transfmat_toimg)
        coords_img.append(xyz_img)
        try: mask_mhd = find_file(f"LNDb-{LNDb_id:04d}_rad{int(line['RadID'])}.mhd")
        except FileNotFoundError: continue
        mask, _, _, _ = utils.readMhd(mask_mhd)
        masks_img.append(mask)

    nods_scan['x_img'] = [c[0] for c in coords_img]
    nods_scan['y_img'] = [c[1] for c in coords_img]
    nods_scan['z_img'] = [c[2] for c in coords_img]
    nods_scan['mask'] = [m for m in masks_img]

    return scan, spacing, origin, nods_scan

def aggregate_columns(series, start_value = 1, end_value = 5, coor = False):
    if coor: return np.median(vals)                                          # Return the median of the coordinates
    vals = sorted(series.dropna())                                           # Get sorted values
    if not vals: return np.nan                                               # If it is emptly, return NaN
    if len(set(vals)) == 1: return vals[0]                                   # If only one value, return it
    if len(set(vals)) == 2: return np.ceil(np.mean(vals))                    # If only two values, return mean rounded up
    middle_value = np.ceil((end_value - start_value + 1)/2)                  # Separation point between high and low values
    val_range = max(vals) - min(vals)                                        # Range in classifications
    if val_range >= end_value - start_value - 1:                             # Handle large discrepancies
            high_vals = [v for v in vals if v > middle_value]                # High classifications
            low_vals = [v for v in vals if v < middle_value]                 # Low classifications
            middle_vals = [v for v in vals if v == middle_value]             # Middle classifications
            if len(vals) == 3:
                if len(high_vals) == 2: return np.ceil(np.mean(high_vals))   # Use mean of high values rounded up
                if len(low_vals) == 2: return np.ceil(np.mean(low_vals)) + 1 # Use mean of low values rounded up, adding 1 to prioritize severity
                return np.median(vals)                                       # Otherwise, use the median
            if len(vals) == 4:
                if len(high_vals) > 2: return np.ceil(np.mean(high_vals))    # Use mean of high values rounded up
                if len(low_vals) > 2: return np.ceil(np.mean(low_vals)) + 1  # Use mean of low values rounded up, adding 1 to prioritize severity
                if len(middle_vals) == 2 and len(high_vals) == 2: return middle_vals + 1 #If two values = 3, then check if others are high or low
                if len(middle_vals) == 2 and len(low_vals) == 2: return middle_vals   
                return np.median(vals)                                       # Otherwise, use the median
    return np.ceil(series.mean())                                            # If not a big range in classifications, use the mean rounded up.