"""LIDC-IDRI nodule table (pylidc, rule for oversized groups) and the individual reader ratings."""
from pathlib import Path
from scipy.sparse.csgraph import connected_components
import numpy as np
import pandas as pd
import pydicom
import warnings
from .config import READER_RATINGS_FILE
from .schema import SCHEMA, equivalent_diameter

MAX_READERS = 4                # LIDC has four radiologists
MIN_SAME_NODULE_RADIUS_MM = 3.0

def import_pylidc(): # Imports pylidc and explains what is missing if it cannot be used
    # pylidc 0.2.3 still uses numpy aliases (np.int, np.float, np.bool) that were removed from numpy
    for alias, real_type in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, real_type)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pylidc
    except Exception as error:
        raise RuntimeError(
            "pylidc is not ready."
            f"Original error: {error}") from error
    return pylidc

def read_first_slice_header(scan): # DICOM header of the first slice: origin of the image plane and scanner fields
    folder = Path(scan.get_path_to_dicom_files())
    slice_files = sorted(f for f in folder.glob("*.dcm") if not f.name.startswith("._"))   # skip macOS copies
    return pydicom.dcmread(slice_files[0], stop_before_pixels=True)

def describe_annotation(annotation, slice_positions_mm, pixel_spacing_mm): # Keeps only what we need from a pylidc annotation: rating, centre and size
    row, col, slice_index = annotation.centroid
    z_mm = np.interp(slice_index, np.arange(len(slice_positions_mm)), slice_positions_mm)
    volume = float(annotation.volume)
    return {
        "id": annotation.id,
        "malignancy": float(annotation.malignancy),
        "voxel_centre": (row, col, slice_index),
        "centre_mm": np.array([col * pixel_spacing_mm, row * pixel_spacing_mm, z_mm]),
        "volume_mm3": volume,
        "radius_mm": float(equivalent_diameter(volume)) / 2,
    }

def centre_spread(annotations): # Largest distance between two annotation centres, divided by the mean diameter.
    #Small values mean that the annotations sit on top of each other, as when one nodule is annotated
    #several times. Values above 1 suggest that the group holds more than one nodule.
    centres = np.array([a["centre_mm"] for a in annotations])
    mean_diameter = 2 * np.mean([a["radius_mm"] for a in annotations])
    distances = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    return float(distances.max() / mean_diameter)

def split_oversized_group(annotations):
    centres = np.array([a["centre_mm"] for a in annotations])
    radii = np.array([a["radius_mm"] for a in annotations])

    distance_between = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    larger_radius = np.maximum(radii[:, None], radii[None, :])
    allowed_distance = np.maximum(larger_radius, MIN_SAME_NODULE_RADIUS_MM)

    is_neighbour = distance_between <= allowed_distance
    _, group_of = connected_components(is_neighbour, directed=False)
    return [[annotations[i] for i in np.flatnonzero(group_of == g)] for g in range(group_of.max() + 1)]

def build_lidc_table():
    pl = import_pylidc()

    rows = []
    scans_where_pylidc_gave_up = []
    oversized_groups = []          # groups that are still above four annotations after the rule
    all_scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).all()

    for position, scan in enumerate(all_scans):
        default_groups = scan.cluster_annotations(verbose=False)
        if not default_groups:
            continue

        try:
            header = read_first_slice_header(scan)
        except OSError:
            print(f"Missing scan {scan}")
            continue

        origin_xy = np.array(header.ImagePositionPatient[:2], dtype=float)
        slices_are_axis_aligned = np.allclose(header.ImageOrientationPatient, [1, 0, 0, 0, 1, 0])
        kernel = getattr(header, "ConvolutionKernel", None)
        if isinstance(kernel, pydicom.multival.MultiValue):
            kernel = kernel[0]
        pixel_spacing = float(scan.pixel_spacing)
        slice_positions = np.asarray(scan.slice_zvals, dtype=float)

        # Decide the final groups of this scan
        final_groups = []     # pairs of (annotations, was it regrouped by our rule?)
        for group in default_groups:
            annotations = [describe_annotation(a, slice_positions, pixel_spacing) for a in group]
            if len(annotations) <= MAX_READERS:
                final_groups.append((annotations, False))
                continue

            # pylidc gave up on this group, so we apply our rule once and keep every resulting group
            scans_where_pylidc_gave_up.append(scan.patient_id)
            final_groups += [(part, True) for part in split_oversized_group(annotations)]

        # One row per final group
        for number, (annotations, regrouped) in enumerate(final_groups, start=1):
            if len(annotations) > MAX_READERS:
                oversized_groups.append({
                    "patient_id": scan.patient_id,
                    "annotations": len(annotations),
                    "ratings": sorted(a["malignancy"] for a in annotations),
                    "spread_over_diameter": round(centre_spread(annotations), 2),
                })
            ratings = np.array([a["malignancy"] for a in annotations])
            row, col, slice_index = np.mean([a["voxel_centre"] for a in annotations], axis=0)
            volume = float(np.mean([a["volume_mm3"] for a in annotations]))
            rows.append({
                "dataset": "LIDC",
                "scan_id": scan.series_instance_uid,
                "patient_id": scan.patient_id,
                "nodule_uid": f"{scan.patient_id}_{scan.series_instance_uid[-6:]}_{number}",
                "x_mm": origin_xy[0] + col * pixel_spacing if slices_are_axis_aligned else np.nan,
                "y_mm": origin_xy[1] + row * pixel_spacing if slices_are_axis_aligned else np.nan,
                "z_mm": float(np.interp(slice_index, np.arange(len(slice_positions)), slice_positions)),
                "diameter_mm": float(equivalent_diameter(volume)),
                "volume_mm3": volume,
                "volume_is_placeholder": False,
                "is_nodule": 1,
                "mal_mean": ratings.mean(), "mal_min": ratings.min(), "mal_max": ratings.max(),
                "n_rated": len(ratings),
                "n_readings": len(annotations),
                "reclustered": regrouped,
                "oversized_group": len(annotations) > MAX_READERS,
                "lobe": np.nan,
                "slice_thickness_mm": float(scan.slice_thickness),
                "pixel_spacing_mm": pixel_spacing,
                "manufacturer": str(getattr(header, "Manufacturer", "")).upper().split(" ")[0] or None,
                "model": getattr(header, "ManufacturerModelName", None),
                "kernel": kernel,
                "kvp": float(getattr(header, "KVP", np.nan)),
                "vox_row": row, "vox_col": col, "vox_slice": slice_index,
                "annotation_ids": ";".join(str(a["id"]) for a in annotations),
            })
        if position % 200 == 0:
            print(f"{position}/{len(all_scans)} scans")

    gave_up = sorted(set(scans_where_pylidc_gave_up))
    print(f"pylidc could not keep every group at four annotations or fewer in {len(gave_up)} scans: {gave_up}")
    print(f"After applying the rule once, {len(oversized_groups)} groups still have more than four annotations. "
          "They are kept and flagged in oversized_group:")
    for item in oversized_groups:
        print(f"  {item['patient_id']}: {item['annotations']} annotations, ratings {item['ratings']}, "
              f"spread {item['spread_over_diameter']} diameters")
    return pd.DataFrame(rows)[SCHEMA]

def save_lidc_reader_ratings():
    for alias, real_type in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, real_type)      # pylidc 0.2.3 still uses aliases removed from numpy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pylidc as pl

    rows = []
    for scan in pl.query(pl.Scan).order_by(pl.Scan.patient_id).all():
        for number, group in enumerate(scan.cluster_annotations(verbose=False)):
            if len(group) > 4:
                continue
            for annotation in group:
                rows.append({"patient_id": scan.patient_id, "nodule": f"{scan.series_instance_uid}_{number}",
                             "malignancy": int(annotation.malignancy)})

    READER_RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(READER_RATINGS_FILE, index=False)
    print(f"{len(rows)} ratings saved to {READER_RATINGS_FILE}")
