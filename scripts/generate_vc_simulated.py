import os
import pandas as pd
import json
import logging
import random
from tqdm import tqdm
from process_fake_audio_util import process_and_save_fake_clip, ClipIdCounter
import torch  # Import torch

from TTS.api import TTS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Cấu hình chung ---
DATASET_OUTPUT_ROOT = "dataset"
FAKE_AUDIO_TYPE = "vc_simulated"  # Tên thư mục con cho loại fake audio này

# Cấu hình mô hình TTS/VC multispeaker
VC_SIMULATED_MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"  # Model này có thể dùng để giả lập VC

# Đường dẫn đến file CSV chứa các cặp (path_to_real_audio, speaker_id, text, text_id)
REAL_AUDIO_PATHS_FILE_TEMPLATE = "real_audio_paths_for_fake_{set_type}.csv"
REAL_SPEAKER_IDS_FILE_TEMPLATE = "real_speaker_ids_{set_type}.json"

# Đường dẫn cố định đến các file CSV
train_csv_path = (
    "F:\\Deepfake-Audio-Detector\\dataset\\train\\real_audio_paths_for_fake_train.csv"
)
val_csv_path = (
    "F:\\Deepfake-Audio-Detector\\dataset\\val\\real_audio_paths_for_fake_val.csv"
)
test_csv_path = (
    "F:\\Deepfake-Audio-Detector\\dataset\\test\\real_audio_paths_for_fake_test.csv"
)

# Số lượng fake cần tạo cho tập train
target_train_vc_sim = 10000  # Hoặc số lượng bạn muốn cho VC mô phỏng

# Tính toán TARGET_FAKE_CLIPS_PER_SET dựa trên tỷ lệ
try:
    train_df_real_paths = pd.read_csv(train_csv_path)
    total_train_rows = len(train_df_real_paths)

    if total_train_rows == 0:
        raise ValueError("Tập tin real_audio_paths_for_fake_train.csv trống rỗng.")

    ratio_vc_sim = target_train_vc_sim / total_train_rows
    print(f"Tỷ lệ mẫu được chọn từ train cho VC mô phỏng: {ratio_vc_sim:.2%}")

    val_df_real_paths = pd.read_csv(val_csv_path)
    test_df_real_paths = pd.read_csv(test_csv_path)

    target_val_vc_sim = int(len(val_df_real_paths) * ratio_vc_sim)
    target_test_vc_sim = int(len(test_df_real_paths) * ratio_vc_sim)

    TARGET_FAKE_CLIPS_PER_SET_VC_SIM = {
        "train": target_train_vc_sim,
        "val": target_val_vc_sim,
        "test": target_test_vc_sim,
    }
except FileNotFoundError as e:
    logging.error(
        f"Lỗi: Không tìm thấy một trong các file CSV: {e}. Đảm bảo bạn đã chạy prepare_real_audio.py."
    )
    TARGET_FAKE_CLIPS_PER_SET_VC_SIM = {"train": None, "val": None, "test": None}
except ValueError as e:
    logging.error(f"Lỗi dữ liệu: {e}. Vui lòng kiểm tra nội dung các file CSV.")
    TARGET_FAKE_CLIPS_PER_SET_VC_SIM = {"train": None, "val": None, "test": None}


# Cache real_audio_metadata để tránh đọc lại nhiều lần
real_audio_metadata_cache = {}


def get_real_audio_metadata_df(set_type):
    if set_type not in real_audio_metadata_cache:
        real_metadata_filepath = os.path.join(
            DATASET_OUTPUT_ROOT, set_type, f"real_audio_metadata_{set_type}.csv"
        )
        if not os.path.exists(real_metadata_filepath):
            logging.error(
                f"File metadata real không tồn tại: {real_metadata_filepath}. Bỏ qua."
            )
            return pd.DataFrame()
        real_audio_metadata_cache[set_type] = pd.read_csv(real_metadata_filepath)
        # Đảm bảo speaker_id là string để so sánh nhất quán
        real_audio_metadata_cache[set_type]["speaker_id"] = real_audio_metadata_cache[
            set_type
        ]["speaker_id"].astype(str)
    return real_audio_metadata_cache[set_type]


def generate_fake_vc_simulated_audio(set_type):
    logging.info(
        f"Bắt đầu tạo audio fake Voice Conversion (mô phỏng) cho tập '{set_type}'"
    )

    real_audio_paths_filepath = os.path.join(
        DATASET_OUTPUT_ROOT,
        set_type,
        REAL_AUDIO_PATHS_FILE_TEMPLATE.format(set_type=set_type),
    )
    real_speaker_ids_filepath = os.path.join(
        DATASET_OUTPUT_ROOT,
        set_type,
        REAL_SPEAKER_IDS_FILE_TEMPLATE.format(set_type=set_type),
    )

    if not os.path.exists(real_audio_paths_filepath):
        logging.error(
            f"File '{real_audio_paths_filepath}' không tồn tại. Hãy chạy prepare_real_audio.py trước."
        )
        return
    if not os.path.exists(real_speaker_ids_filepath):
        logging.error(
            f"File '{real_speaker_ids_filepath}' không tồn tại. Yêu cầu speaker IDs để thực hiện VC mô phỏng. Bỏ qua."
        )
        return

    df_real_audio_paths = pd.read_csv(real_audio_paths_filepath)
    df_real_audio_paths["speaker_id"] = df_real_audio_paths["speaker_id"].astype(
        str
    )  # Đảm bảo speaker_id là string

    # Load speaker IDs
    with open(real_speaker_ids_filepath, "r") as f:
        all_speaker_ids = [
            str(s_id) for s_id in json.load(f)
        ]  # Đảm bảo speaker IDs là string

    if len(all_speaker_ids) < 2:
        logging.error(
            f"Không đủ speaker IDs ({len(all_speaker_ids)}) trong tập '{set_type}' để thực hiện Voice Conversion mô phỏng. Cần ít nhất 2 speaker."
        )
        return

    # Giới hạn số lượng bản ghi nếu có TARGET_FAKE_CLIPS_PER_SET_VC
    max_samples = TARGET_FAKE_CLIPS_PER_SET_VC_SIM.get(set_type)
    if max_samples is not None:
        if len(df_real_audio_paths) > max_samples:
            df_real_audio_paths = df_real_audio_paths.sample(
                n=max_samples, random_state=42
            )
            logging.info(
                f"Giới hạn số lượng bản ghi để tạo fake audio xuống còn {TARGET_FAKE_CLIPS_PER_SET_VC_SIM[set_type]}."
            )

    # Khởi tạo mô hình TTS/VC
    logging.info(f"Đang tải mô hình VC (mô phỏng): {VC_SIMULATED_MODEL_NAME}...")
    try:
        tts = TTS(model_name=VC_SIMULATED_MODEL_NAME, gpu=torch.cuda.is_available())
        logging.info("Đã tải mô hình VC (mô phỏng) thành công.")
    except Exception as e:
        logging.error(
            f"Không thể tải mô hình VC (mô phỏng) '{VC_SIMULATED_MODEL_NAME}': {e}"
        )
        logging.info("Hãy đảm bảo bạn đã cài đặt Coqui-TTS và mô hình đã được tải về.")
        return

    fake_metadata = []
    fake_clip_counter = ClipIdCounter()

    fake_audio_output_dir = os.path.join(
        DATASET_OUTPUT_ROOT, set_type, "fake", FAKE_AUDIO_TYPE
    )
    os.makedirs(fake_audio_output_dir, exist_ok=True)

    logging.info(
        f"Bắt đầu tạo {len(df_real_audio_paths)} clip fake VC (mô phỏng) cho tập '{set_type}'..."
    )

    # Load metadata real audio một lần để tăng hiệu quả
    real_metadata_df = get_real_audio_metadata_df(set_type)
    if real_metadata_df.empty:
        logging.error(
            f"Không thể tải metadata real audio cho tập '{set_type}'. Không thể tiến hành VC mô phỏng."
        )
        return

    for index, row in tqdm(
        df_real_audio_paths.iterrows(),
        total=len(df_real_audio_paths),
        desc=f"Generating {FAKE_AUDIO_TYPE} for {set_type}",
    ):
        text = row["text"]
        original_speaker_id = str(
            row["speaker_id"]
        )  # Speaker ID của audio real gốc (đảm bảo là string)
        text_id = row["text_id"]  # ID của đoạn văn bản gốc

        # Chọn một speaker_id đích khác với speaker gốc
        possible_target_speaker_ids = [
            sid for sid in all_speaker_ids if sid != original_speaker_id
        ]

        if not possible_target_speaker_ids:
            logging.warning(
                f"Không tìm thấy speaker ID khác cho speaker gốc '{original_speaker_id}' trong tập '{set_type}'. Bỏ qua bản ghi."
            )
            continue

        target_speaker_id = random.choice(possible_target_speaker_ids)

        # Tìm một clip real bất kỳ của target_speaker_id để làm speaker_wav
        ref_audio_for_target_speaker_paths = real_metadata_df[
            real_metadata_df["speaker_id"] == target_speaker_id
        ]["path"].tolist()

        if not ref_audio_for_target_speaker_paths:
            logging.warning(
                f"Không tìm thấy clip real cho speaker '{target_speaker_id}' để làm ref. Bỏ qua bản ghi."
            )
            continue

        # Chọn ngẫu nhiên một audio reference từ speaker đích
        ref_audio_path_relative = random.choice(ref_audio_for_target_speaker_paths)
        full_ref_audio_path = os.path.join(DATASET_OUTPUT_ROOT, ref_audio_path_relative)

        if not os.path.exists(full_ref_audio_path):
            logging.warning(
                f"File tham chiếu audio real không tồn tại: {full_ref_audio_path}. Bỏ qua bản ghi."
            )
            continue

        try:
            # Tạo audio fake bằng cách sử dụng text gốc nhưng với giọng của speaker đích
            waveform = tts.tts(text=text, speaker_wav=full_ref_audio_path)
            sample_rate = tts.synthesizer.output_sample_rate  # type: ignore

            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform).float()

            clip_metadata = process_and_save_fake_clip(
                waveform=waveform,
                sample_rate=sample_rate,
                output_base_dir=fake_audio_output_dir,
                speaker_id=original_speaker_id,  # Vẫn giữ speaker_id gốc để dễ theo dõi nguồn text
                text_id=text_id,
                clip_id_counter=fake_clip_counter,
                source_dataset_name="LibriTTS",
                synthesis_tool=VC_SIMULATED_MODEL_NAME.split("/")[-1]
                + "_simulated",  # Tên công cụ/mô hình
                fake_level=4,  # Cấp độ fake 4 cho Voice Conversion
                reference_path=ref_audio_path_relative,  # Lưu đường dẫn đến audio real dùng làm reference giọng đích
            )
            fake_metadata.extend(clip_metadata)

        except Exception as e:
            logging.error(
                f"Lỗi khi tạo fake audio VC mô phỏng cho text_id '{text_id}' với ref '{full_ref_audio_path}': {e}"
            )
            continue

    df_fake_metadata = pd.DataFrame(fake_metadata)
    if not df_fake_metadata.empty:
        fake_metadata_filepath = os.path.join(
            DATASET_OUTPUT_ROOT,
            set_type,
            f"fake_audio_metadata_{FAKE_AUDIO_TYPE}_{set_type}.csv",
        )
        df_fake_metadata.to_csv(fake_metadata_filepath, index=False)
        logging.info(
            f"Metadata fake audio cho tập '{set_type}' đã lưu tại: {fake_metadata_filepath}"
        )
    else:
        logging.warning(
            f"Không có clip fake nào được tạo cho tập '{set_type}' bằng công cụ '{FAKE_AUDIO_TYPE}'."
        )

    logging.info(
        f"Hoàn tất tạo audio fake Voice Conversion (mô phỏng) cho tập '{set_type}'."
    )


if __name__ == "__main__":
    for set_type in ["train", "val", "test"]:
        generate_fake_vc_simulated_audio(set_type)

    logging.info(
        "Tất cả quá trình tạo audio fake Voice Conversion (mô phỏng) đã hoàn tất."
    )
