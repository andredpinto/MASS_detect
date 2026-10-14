"""Builds every patch, mask and radiomic feature of the study at the voxel size of config.py, from the terminal.

Notebooks 04 and 05 read what this script writes, so a voxel size change only needs this run (about two hours).
Every step skips what is already on disk, so a stopped run resumes where it stopped.

    uv run python -m src.mass_detect.scripts.build_patches_and_features
"""
import argparse
import time
import pandas as pd
from .cohort import select_cohort
from .config import PATCHES_DIR, READINGS_FILE, ROI_DIR, ROI_ICC_FILE, ROI_RADIOMICS_FILE, SPACING_MM
from .patches import cut_lidc_patches, cut_lndb_patches
from .rois import build_readings, build_rois, extract_roi_radiomics, reader_icc

def main(): # Patches, then the radiologists' masks, then PyRadiomics on every region and the robustness between readers
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=10, help="worker processes for the radiomic extraction")
    args = parser.parse_args()
    started = time.time()
    print(f"voxel size {SPACING_MM} mm, writing to {PATCHES_DIR}", flush=True)

    cut_lndb_patches("train")                              # LNDb training findings
    print(f"LNDb patches done after {time.time() - started:.0f} s", flush=True)
    cut_lidc_patches()                                     # LIDC nodules, with their position check
    print(f"LIDC patches done after {time.time() - started:.0f} s", flush=True)

    cohort, funnel = select_cohort()
    print(funnel.to_string(), flush=True)
    readings = pd.read_csv(READINGS_FILE) if READINGS_FILE.exists() else build_readings(cohort)
    if len(list(ROI_DIR.glob("*.npz"))) < len(cohort):
        build_rois(cohort, readings, processes=6)
    print(f"masks done after {time.time() - started:.0f} s", flush=True)

    features = pd.read_csv(ROI_RADIOMICS_FILE) if ROI_RADIOMICS_FILE.exists() else extract_roi_radiomics(cohort, processes=args.processes)
    print(f"radiomics done after {time.time() - started:.0f} s", flush=True)
    if not ROI_ICC_FILE.exists():
        icc, pairs = reader_icc(features, readings)
        print(f"robust features: {int(icc['robust'].sum())} of {len(icc)}, nodules with two readers: {pairs}", flush=True)
    print(f"done in {time.time() - started:.0f} s", flush=True)

if __name__ == "__main__":
    main()
