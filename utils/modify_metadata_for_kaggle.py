import pandas as pd
import os

# Định nghĩa các hằng số
MODEL_DATASETS = ['cnn_balanced_dataset', 'vit_balanced_dataset', 'cnn_performance_dataset', 'vit_performance_dataset']
BASE_DIR = "F:\\Deepfake-Audio-Detector\\processed_dataset\\"
SET_TYPES = ["train", "val", "test"]
KAGGLE_PREFIX = "/kaggle/input/"

for model_dataset in MODEL_DATASETS:
    kaggle_dataset = model_dataset.replace("_", "-")
    # Xử lý từng set_type
    for set_type in SET_TYPES:
        # Đường dẫn đến file metadata.csv gốc
        metadata_path = os.path.join(BASE_DIR, model_dataset, set_type, "metadata.csv")

        # Kiểm tra file metadata.csv có tồn tại không
        if not os.path.exists(metadata_path):
            print(f"File not found: {metadata_path}. Skipping {set_type} set.")
            continue

        # Đọc file metadata.csv
        try:
            df = pd.read_csv(metadata_path)
        except Exception as e:
            print(f"Error reading {metadata_path}: {e}. Skipping {set_type} set.")
            continue

        # Kiểm tra cột 'npy_path' có tồn tại không
        if "npy_path" not in df.columns:
            print(
                f"Column 'npy_path' not found in {metadata_path}. Skipping {set_type} set."
            )
            continue

        # Sửa npy_path
        # Từ: F:\Deepfake-Audio-Detector\processed_dataset\{model_dataset}\{set_type}\{label}\{filename}.npy
        # Thành: /kaggle/input/{kaggle_dataset}/{model_dataset}/{set_type}/{label}/{filename}.npy
        df["npy_path"] = df["npy_path"].apply(
            lambda x: os.path.join(
                KAGGLE_PREFIX,
                kaggle_dataset,
                model_dataset,
                set_type,
                x.split(os.sep)[-2],  # Lấy thư mục label (real/fake)
                x.split(os.sep)[-1],  # Lấy tên file
            ).replace("\\", "/")  # Đảm bảo dấu phân cách là /
        )

        # Đường dẫn đến file kaggle_metadata.csv mới
        output_metadata_path = os.path.join(BASE_DIR, model_dataset, set_type, "kaggle_metadata.csv")
        
        if os.path.exists(output_metadata_path):
            print(f"Removing existing {output_metadata_path}.")
            os.remove(output_metadata_path)
        
        # Lưu file kaggle_metadata.csv
        try:
            df.to_csv(output_metadata_path, index=False)
            print(f"Created kaggle_metadata.csv for {set_type} at: {output_metadata_path}")
        except Exception as e:
            print(f"Failed to save {output_metadata_path}: {e}")

print("Processing completed.")
