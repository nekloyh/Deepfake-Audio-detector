import pandas as pd
from pathlib import Path

BASE_DIR = Path("F:/Deepfake-Audio-Detector/datasets/final_dataset")
MODEL_DATASETS = ["cnn_3s_dataset", "vit_3s_dataset"]
SET_TYPES = ["train", "val", "test"]
KAGGLE_PREFIX = "/kaggle/input"


def transform_path(x, kaggle_dataset, model_dataset, set_type, skipped):
    try:
        parts = Path(x).parts
        if len(parts) >= 2:
            new_path = Path(
                KAGGLE_PREFIX,
                kaggle_dataset,
                model_dataset,
                set_type,
                parts[-2],
                parts[-1],
            )
            return str(new_path).replace("\\", "/")
        else:
            skipped.append(x)
            return x
    except Exception:
        skipped.append(x)
        return x


def process_metadata(model_dataset, set_type):
    kaggle_dataset = model_dataset.replace("_", "-")
    metadata_path = BASE_DIR / model_dataset / set_type / "metadata.csv"

    if not metadata_path.exists():
        print(f"❌ File not found: {metadata_path}. Skipping {set_type} set.")
        return

    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"❌ Error reading {metadata_path}: {e}. Skipping {set_type} set.")
        return

    if "npy_path" not in df.columns:
        print(
            f"❌ Column 'npy_path' not found in {metadata_path}. Skipping {set_type} set."
        )
        return

    skipped = []
    df["npy_path"] = df["npy_path"].apply(
        lambda x: transform_path(x, kaggle_dataset, model_dataset, set_type, skipped)
    )

    if skipped:
        print(
            f"⚠️  Skipped {len(skipped)} invalid npy_path rows in {model_dataset}/{set_type}."
        )

    output_metadata_path = BASE_DIR / model_dataset / set_type / "kaggle_metadata.csv"

    try:
        df.to_csv(output_metadata_path, index=False)
        print(
            f"✅ Created kaggle_metadata.csv for {set_type} at: {output_metadata_path}"
        )
    except Exception as e:
        print(f"❌ Error writing file {output_metadata_path}: {e}")


if __name__ == "__main__":
    for model_dataset in MODEL_DATASETS:
        for set_type in SET_TYPES:
            process_metadata(model_dataset, set_type)

    print("🎉 Processing completed.")
