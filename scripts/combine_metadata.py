import os
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cấu hình
DATASET_OUTPUT_ROOT = "dataset"
SET_TYPES = ["train", "val", "test"]
# Các tiền tố của file metadata mà bạn muốn tổng hợp
# Thêm các loại fake audio khác vào đây nếu bạn tạo thêm
METADATA_PREFIXES = [
    "real_audio_metadata",
    "fake_audio_metadata_tts_basic",  # Mức 1-2
    "fake_audio_metadata_vc_basic",  # Mức 3 (Voice Cloning)
    "fake_audio_metadata_vc_simulated",  # Mức 4 (Voice Conversion mô phỏng)
]


def combine_all_metadata():
    logging.info("Bắt đầu quá trình tổng hợp metadata...")

    for set_type in SET_TYPES:
        combined_df = pd.DataFrame()
        set_dir = os.path.join(DATASET_OUTPUT_ROOT, set_type)

        if not os.path.exists(set_dir):
            logging.warning(
                f"Thư mục '{set_dir}' không tồn tại. Bỏ qua tập '{set_type}'."
            )
            continue

        logging.info(f"Tổng hợp metadata cho tập '{set_type}'...")

        found_files_for_set = False
        for prefix in METADATA_PREFIXES:
            if prefix == "real_audio_metadata":
                filename = f"{prefix}_{set_type}.csv"
            else:
                # Đối với fake_audio_metadata_tts_basic, fake_audio_metadata_vc_basic, fake_audio_metadata_vc_simulated
                # Lấy phần sau 'fake_audio_metadata_'
                tool_name = prefix.replace("fake_audio_metadata_", "")
                filename = f"fake_audio_metadata_{tool_name}_{set_type}.csv"

            filepath = os.path.join(set_dir, filename)

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    logging.info(
                        f"Đã thêm '{filename}' ({len(df)} bản ghi) vào tập '{set_type}'."
                    )
                    found_files_for_set = True
                except Exception as e:
                    logging.error(f"Lỗi khi đọc file '{filepath}': {e}")
            else:
                logging.debug(f"File '{filepath}' không tìm thấy. Bỏ qua.")

        if found_files_for_set and not combined_df.empty:
            output_filepath = os.path.join(set_dir, f"combined_metadata_{set_type}.csv")
            # Đảm bảo các cột quan trọng có kiểu dữ liệu phù hợp nếu có giá trị NaN
            if "speaker_id" in combined_df.columns:
                combined_df["speaker_id"] = combined_df["speaker_id"].astype(str)
            if "text_id" in combined_df.columns:
                combined_df["text_id"] = combined_df["text_id"].astype(str)

            combined_df.to_csv(output_filepath, index=False)
            logging.info(
                f"Tổng hợp {len(combined_df)} bản ghi cho tập '{set_type}' đã lưu tại: {output_filepath}"
            )
        else:
            logging.warning(f"Không có metadata nào để tổng hợp cho tập '{set_type}'.")

    logging.info("Quá trình tổng hợp metadata hoàn tất.")


if __name__ == "__main__":
    combine_all_metadata()
