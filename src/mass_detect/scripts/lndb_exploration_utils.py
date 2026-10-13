from src.mass_detect.lndb_utils import utils
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as plt

BLUE, ORANGE, GREY, OLIVE = "blue", "orange", "grey", "olive"
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GREY,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "w",
    "grid.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "figure.dpi": 110,
})


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

def aggregate_columns(series, start_value = None, end_value = None, mode = "characteristic"):
    vals = sorted(series.dropna()) #Get sorted values
    if not vals: return np.nan #If it is emptly, return NaN
    if len(set(vals)) == 1: return vals[0] #If only one value, return it
    val_range = max(vals) - min(vals) #Range in classifications
    match mode:
        case "coordinate":
            if val_range < 3.0 or len(vals) == 2: return np.mean(vals)
            range_1 = abs(vals[0] - vals[1])
            range_2 = abs(vals[1] - vals[2])
            range_3 = abs(vals[0] - vals[2])
            if range_1 < 3.0: return np.mean([vals[0], vals[1]])
            if range_2 < 3.0: return np.mean([vals[1], vals[2]])
            if range_3 < 3.0: return np.mean([vals[0], vals[2]])
            return np.mean(vals)
        case "volume":
            relative_diff = val_range / min(vals) if min(vals) > 0 else 0
            if relative_diff <= 0.3 or len(vals) == 2: return np.mean(vals)
            diff_1 = abs(vals[0] - vals[1]) / min([vals[0], vals[1]]) 
            diff_2 = abs(vals[1] - vals[2]) / min([vals[1], vals[2]])
            diff_3 = abs(vals[0] - vals[2]) / min ([vals[0], vals[2]])
            if diff_1 <= 0.3: return np.mean([vals[0], vals[1]])
            if diff_2 <= 0.3: return np.mean([vals[1], vals[2]])
            if diff_3 <= 0.3: return np.mean([vals[0], vals[2]])
            return np.median(vals)
        case "characteristic":
            middle_value = np.ceil((end_value - start_value)/2) #Defines separation point between high and low values
            if val_range >= end_value - start_value - 1: # Handle large discrepancies
                if vals.count(max(vals)) == 2: return max(vals) #If majority is the max value, use it
                if vals.count(min(vals)) == 2: return min(vals) + 1 #If majority is the min value, use it (plus 1 to prioritize detecting classification severity)
                #If there are 3 different values, use the dominant high/low values
                high_vals = [v for v in vals if v > middle_value] #High classifications
                low_vals = [v for v in vals if v < middle_value] #Low classifications
                if len(high_vals) > 1: return np.mean(high_vals) # Use the mean of the high values
                if len(low_vals) > 1: return max(low_vals) # Use the max of the low values to prioritize detecting classification severity
                return middle_value #Return middle value if high/low value counts are balanced
            return series.mean() #If there is not a big range in classifications, use the mean.