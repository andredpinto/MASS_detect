"""Runs the voxel size sensitivity analysis of notebook 05 from the terminal, so that it can run unattended.

The extraction writes one file per scan (data/voxel_size_sensitivity/extraction/), so a stopped run resumes where it stopped.
The four result tables are saved to data/voxel_size_sensitivity/results/, where notebook 05 reads them.

    uv run python -m src.mass_detect.scripts.run_voxel_sensitivity
"""
import argparse
import time
import pandas as pd
from .cohort import select_cohort
from .config import READINGS_FILE
from .sensitivity import (RESULTS_DIR, acquisition_table, consensus_rows, coverage_table, extract_at_spacings,
                          feature_columns, load_spacing_features, prediction_table, robustness_table)

def run(processes=10, limit_scans=None, only_extract=False): # Extraction at every voxel size, then the four criteria, each saved as soon as it is ready
    started = time.time()
    cohort, _ = select_cohort()
    readings = pd.read_csv(READINGS_FILE)
    extract_at_spacings(cohort, readings, processes=processes, limit_scans=limit_scans)
    if only_extract:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    voxel = load_spacing_features()
    names = feature_columns(voxel)
    rows = consensus_rows(voxel, cohort)
    coverage_table(rows).to_csv(RESULTS_DIR / "coverage.csv")
    robustness_table(voxel, cohort).to_csv(RESULTS_DIR / "robustness.csv")
    acquisition_table(rows, names).to_csv(RESULTS_DIR / "acquisition.csv")
    print(f"coverage, robustness and acquisition saved after {time.time() - started:.0f} s", flush=True)

    every_size, nodules_every = prediction_table(rows, names, processes=processes)
    every_size.assign(nodules=nodules_every).to_csv(RESULTS_DIR / "prediction_every_size.csv")
    fine, nodules_fine = prediction_table(rows, names, spacings=[0.6, 1.0], processes=processes)
    fine.assign(nodules=nodules_fine).to_csv(RESULTS_DIR / "prediction_06_against_1mm.csv")
    print(f"done in {time.time() - started:.0f} s; results in {RESULTS_DIR}", flush=True)

def main(): # Command line entry point
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=10, help="worker processes (10 fit in 11 GB of memory)")
    parser.add_argument("--limit-scans", type=int, default=None, help="only this many scans, to try the run first")
    parser.add_argument("--only-extract", action="store_true", help="stop after the extraction")
    args = parser.parse_args()
    run(args.processes, args.limit_scans, args.only_extract)

if __name__ == "__main__":
    main()
