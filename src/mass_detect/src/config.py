"""Paths and constants shared by every notebook and by the other modules of src/."""
import os
from pathlib import Path

PATCH_SIZE = 64                        # Voxels per side
SPACING_MM = 1.0                       # Voxel size of the patch
CENTRE_INDEX = PATCH_SIZE // 2         # Voxel that holds the centre: 32
AIR_HU = -1024                         # Value used outside the CT, the floor of the usual HU normalisation
REPO_ROOT = Path(__file__).resolve().parents[1]   # repository root (the folder that holds src/ and notebooks/)
DATA_DIR = Path(os.environ.get("MASS_DATA_DIR", REPO_ROOT / "data")) #where our 'data' folder is located
LNDB_DIR = DATA_DIR / "LNDb"           # LNDb dataset location
PATCHES_DIR = DATA_DIR / "patches"     # Patches location
LIDC_DIR = DATA_DIR / "TCIA_LIDC-IDRI_20200921"
NODULE_TABLE = REPO_ROOT / "notebooks" / "nodule_table.csv"
FEATURES_FILE = PATCHES_DIR / "features.csv"
MASK_FILE = LNDB_DIR / "LNDb_masks" / "masks" / "LNDb-{ct:04d}_rad{rad}.mhd"
QC_FIGURE_FILE = REPO_ROOT / "reports" / "figures" / "patch_qc.png"
READER_RATINGS_FILE = PATCHES_DIR / "lidc_reader_ratings.csv"

# Intermediate tables written by notebooks 01 and 02 and read by notebook 03
LNDB_NODULES_FILE = REPO_ROOT / "notebooks" / "lndb_nodules.csv"
LIDC_NODULES_FILE = REPO_ROOT / "notebooks" / "lidc_nodules.csv"
