import SimpleITK as sitk
# We could use the lndb_utils/utils.py for handling of mhd files, but these use SimpleITK anyway,
# so might as well go to the source
import numpy as np
from pathlib import Path

def resample_image(image: sitk.Image, target_spacing: float, is_mask=False):
    """It is assumed we want cubic voxels, so target spacing only gets a float"""
    target_spacing = [target_spacing for _ in range(3)]

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()   # Constant in LNDb?

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    # Masks are binary, so only 0 or 1 should be produced when resampling (therefore KNearestNeighbour)
    # For the CT scans, read somewhere that either first or third order spline interpolation, but third order
    # can over(or under)shoot the values, meaning that values outside of intensity range can be produced
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000 if not is_mask else 0)
    resampler.SetInterpolator(interpolator)
    
    return resampler.Execute(image)



if __name__ == "__main__":
    # Inputs
    DATA_DIR = "data"
    MASK_DIR = "masks"

    # Outputs (created if not exist)
    OUTPUT_DIR = Path("resampled")
    MASK_OUTPUT_DIR = OUTPUT_DIR / "masks"
    OUTPUT_DIR.mkdir(exist_ok=True)
    MASK_OUTPUT_DIR.mkdir(exist_ok=True)

    # Script paramaters
    TARGET_SPACING = 0.65

    FILE_LIST = set(Path(DATA_DIR).glob("*.mhd"))    # Searches for .mhd files
    masks_path = Path(MASK_DIR)

    for ct_scan in FILE_LIST:
        filename = ct_scan.name.split(".")[0]   # Remove extension
        #try:
        image = sitk.ReadImage(ct_scan)
        print("Resampling scan", filename)
        resampled = resample_image(image, TARGET_SPACING)
        sitk.WriteImage(resampled, f"{OUTPUT_DIR}/{filename}_resampled.mhd")

        masks = set(masks_path.glob(filename + "_rad*.mhd"))
        if len(masks) < 1:
            print(f"WARNING: Found no mask for image {filename}")
        else:
            for mask in masks:
                maskname = mask.name.split(".")[0].split("_")[1]   # Extract rad* part
                loaded_mask = sitk.ReadImage(mask)
                resampled_mask = resample_image(loaded_mask, TARGET_SPACING, is_mask=True)
                sitk.WriteImage(resampled_mask, f"{OUTPUT_DIR}/masks/{filename}_{maskname}_resampled.mhd")


            

