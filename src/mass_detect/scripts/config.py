"""Paths and constants shared by every notebook and by the other modules of src/mass_detect/scripts/."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]   # repository root: note -> mass_detect -> src -> root (holds src/, notebooks/, data/)

# INPUT DIRS ===============================================
DATA_DIR = Path(os.environ.get("MASS_DATA_DIR", REPO_ROOT / "data")) #where our 'data' folder is located
LNDB_DIR = DATA_DIR / "LNDb"           # LNDb dataset location
LIDC_DIR = DATA_DIR / "TCIA_LIDC-IDRI_20200921"
# ==========================================================

# Parameters
CUBE_MM = 64.0                         # Physical side of the patch, in mm
SPACING_MM = 0.6                       # Voxel size of the patch, chosen with the sensitivity analysis of notebook 05
PATCH_SIZE = round(CUBE_MM / SPACING_MM)   # Voxels per side: 107
CENTRE_INDEX = PATCH_SIZE // 2         # Voxel that holds the centre: 53
AIR_HU = -1024                         # Value used outside the CT, the floor of the usual HU normalisation

# OUTPUTS ====================================================================================
PATCHES_DIR = DATA_DIR / "patches_0.6mm"   # Patches, masks and features at the chosen voxel size
SENSITIVITY_DIR = DATA_DIR / "voxel_size_sensitivity"   # Extraction and results of the voxel size analysis
NODULE_TABLE = REPO_ROOT / "csv_files" / "nodule_table.csv"
FEATURES_FILE = PATCHES_DIR / "features.csv"
MASK_FILE = LNDB_DIR / "LNDb_masks" / "masks" / "LNDb-{ct:04d}_rad{rad}.mhd"
QC_FIGURE_FILE = REPO_ROOT / "reports" / "figures" / "patch_qc.png"
READER_RATINGS_FILE = PATCHES_DIR / "lidc_reader_ratings.csv"
LNDB_MANIFEST_FILE = PATCHES_DIR / "manifest_train.csv"   # where each LNDb training patch was cut
LIDC_MANIFEST_FILE = PATCHES_DIR / "manifest_lidc.csv"    # where each LIDC patch was cut, and whether its position was confirmed

# Radiologists' regions (src/mass_detect/scripts/rois.py)
ROI_DIR = PATCHES_DIR / "rois"                              # one .npz per nodule: every reader mask and the consensus mask
READINGS_FILE = PATCHES_DIR / "readings.csv"                # one row per reading: nodule, reader, rating
ROI_RADIOMICS_FILE = PATCHES_DIR / "radiomics_rois.csv"     # PyRadiomics features of every region (original and LoG images)
ROI_WAVELET_FILE = PATCHES_DIR / "radiomics_rois_wavelet.csv"   # PyRadiomics features of every region on the wavelet images
ROI_ICC_FILE = PATCHES_DIR / "radiomics_rois_icc.csv"       # robustness of every feature between two readers
FEATURE_REPORT_FILE = PATCHES_DIR / "feature_filter_report.csv"   # every feature removed by the filters of notebook 05, and why
FEATURE_SETS_FILE = PATCHES_DIR / "feature_sets.json"      # the kept features with and without the wavelet images
MODEL_FEATURES_FILE = PATCHES_DIR / "model_features.csv"     # one row per cohort nodule: kept features, label and groups (read by notebook 06)
READING_FEATURES_FILE = PATCHES_DIR / "reading_features.csv" # one row per reading: kept features of the reader's region and that reader's rating
MODEL_RESULTS_DIR = PATCHES_DIR / "model_results"          # every configuration of notebook 06 and the summary table

# Intermediate tables written by notebooks 01 and 02 and read by notebook 03
LNDB_NODULES_FILE = REPO_ROOT / "csv_files" / "lndb_nodules.csv"
LIDC_NODULES_FILE = REPO_ROOT / "csv_files" / "lidc_nodules.csv"
# ============================================================================================