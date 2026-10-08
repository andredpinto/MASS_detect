from radiomics import featureextractor
from pathlib import Path

DATA_DIR = Path("data", "processed")

config_file = "Params.yml"

extractor = featureextractor.RadiomicsFeatureExtractor(config_file)

scans = [file for file in DATA_DIR.glob("*.mhd") if "mask" not in file.name]

limit = 1
counter = 0

extracted_features = []

for scan in scans:
    counter += 1
    if counter > limit: break

    name = scan.name.split(".")[0]
    print(f"Processing {name}")
    mask = Path(DATA_DIR, name + "_mask.mhd")

    result = extractor.execute(str(scan), str(mask))    # For some reason Path arguments don't work

    extracted_features.append(result)
