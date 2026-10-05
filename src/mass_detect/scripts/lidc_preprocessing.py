# THIS IS AN ADAPTATION FROM THE SCRIPT https://github.com/jaeho3690/LIDC-IDRI-Preprocessing/blob/master/prepare_dataset.py
# Big thanks to this guy
# A custom environment is needed to run this (check requirements_lidc.txt)
# If you dont know how to, research or ask AI "how to install dependencies from requirements.txt pip"

import os
import numpy as np
import pandas as pd
import warnings
import SimpleITK as sitk
import pylidc as pl
from statistics import median_high

from pylidc.utils import consensus

warnings.filterwarnings(action='ignore')

# === Configuration ===
DICOM_DIR = "data/LIDC/lidc_idri"
OUTPUT_DIR = "data/processed"
MASK_THRESHOLD = 50
CONFIDENCE_LEVEL = 0.5
PADDING_SIZE = 10


class MakeDataSet:
    def __init__(self, LIDC_Patients_list, output_dir, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.output_dir = output_dir
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding, padding), (padding, padding), (0, 0)]
        self.meta = pd.DataFrame(columns=['patient_id', 'nodule_idx', 'image_path', 'mask_path', 'malignancy', 'is_cancer'])

    def calculate_malignancy(self, nodule):
        """Calculate malignancy score and cancer label from radiologist annotations."""
        list_of_malignancy = [annotation.malignancy for annotation in nodule]
        malignancy = median_high(list_of_malignancy)
        if malignancy > 3:
            return malignancy, True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'

    def save_meta(self, meta_list):
        """Append a row to the metadata DataFrame."""
        tmp = pd.Series(meta_list, index=['patient_id', 'nodule_idx', 'image_path', 'mask_path', 'malignancy', 'is_cancer'])
        self.meta = pd.concat([self.meta, tmp.to_frame().T], ignore_index=True)

    def prepare_dataset(self):
        """Process all patients: extract 3D nodule volumes and save as .mhd files."""
        os.makedirs(self.output_dir, exist_ok=True)

        for patient in self.IDRI_list:
            pid = patient
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            try:
                nodules_annotation = scan.cluster_annotations()
                vol = scan.to_volume()
            except:
                print(f"Something wrong with patient {pid}... Oh well, moving on")
                continue
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid, vol.shape, len(nodules_annotation)))

            if len(nodules_annotation) == 0:
                continue  # Skip clean patients (no nodules to extract)

            for nodule_idx, nodule in enumerate(nodules_annotation):
                mask, cbbox, masks = consensus(nodule, self.c_level, self.padding)
                lung_np_array = vol[cbbox]

                # Skip if mask is too small
                if np.sum(mask) <= self.mask_threshold:
                    continue

                malignancy, cancer_label = self.calculate_malignancy(nodule)

                # Build 3D file paths
                image_name = "{}_nodule{}.mhd".format(pid, nodule_idx)
                mask_name = "{}_nodule{}_mask.mhd".format(pid, nodule_idx)
                image_path = os.path.join(self.output_dir, image_name)
                mask_path = os.path.join(self.output_dir, mask_name)

                # Convert to SimpleITK images
                volume_sitk = sitk.GetImageFromArray(lung_np_array)
                mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))

                # Set spacing — pylidc returns (z, y, x), SimpleITK expects (x, y, z)
                spacing_zyx = scan.spacings
                volume_sitk.SetSpacing(spacing_zyx[::-1])
                mask_sitk.SetSpacing(spacing_zyx[::-1])

                # Save as .mhd (SimpleITK writes .mhd + .raw pair)
                sitk.WriteImage(volume_sitk, image_path)
                sitk.WriteImage(mask_sitk, mask_path)

                # Save metadata
                meta_list = [pid, nodule_idx, image_name, mask_name, malignancy, cancer_label]
                self.save_meta(meta_list)

        print("Saved Meta data")
        self.meta.to_csv(os.path.join(self.output_dir, 'meta_info_lidc.csv'), index=False)


if __name__ == '__main__':
    LIDC_IDRI_list = [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()

    dataset = MakeDataSet(LIDC_IDRI_list, OUTPUT_DIR, MASK_THRESHOLD, PADDING_SIZE, CONFIDENCE_LEVEL)
    dataset.prepare_dataset()
