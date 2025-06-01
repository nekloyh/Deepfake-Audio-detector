import os
import pandas as pd
import logging
from tqdm import tqdm
from process_fake_audio_util import process_and_save_fake_clip, ClipIdCounter
import torch  # Import torch

# Import các thư viện TTS cụ thể (ví dụ: Coqui-TTS)
from TTS.api import TTS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Cấu hình chung ---
DATASET_OUTPUT_ROOT = "dataset"
FAKE_AUDIO_TYPE = "vc_basic"  # Tên thư mục con cho loại fake audio này

# Cấu hình mô hình TTS/VC multispeaker
VC_MODEL_NAME = (
    "tts_models/multilingual/multi-dataset/your_tts"  # Model này hỗ trợ cloning
)

# Đường dẫn đến file CSV chứa các cặp (path_to_real_audio, speaker_id, text, text_id)
REAL_AUDIO_PATHS_FILE_TEMPLATE = "real_audio_paths_for_fake_{set_type}.csv"
REAL_SPEAKER_IDS_FILE_TEMPLATE = "real_speaker_ids_{set_type}.json"  # Mặc dù không dùng trực tiếp, nhưng vẫn cần để xác định speaker IDs nếu bạn muốn

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
target_train_vc = 10000  # Hoặc số lượng bạn muốn cho VC

# Tính toán TARGET_FAKE_CLIPS_PER_SET dựa trên tỷ lệ
try:
    train_df_real_paths = pd.read_csv(train_csv_path)
    total_train_rows = len(train_df_real_paths)

    if total_train_rows == 0:
        raise ValueError("Tập tin real_audio_paths_for_fake_train.csv trống rỗng.")

    ratio_vc = target_train_vc / total_train_rows
    print(f"Tỷ lệ mẫu được chọn từ train cho VC: {ratio_vc:.2%}")

    val_df_real_paths = pd.read_csv(val_csv_path)
    test_df_real_paths = pd.read_csv(test_csv_path)

    target_val_vc = int(len(val_df_real_paths) * ratio_vc)
    target_test_vc = int(len(test_df_real_paths) * ratio_vc)

    TARGET_FAKE_CLIPS_PER_SET_VC = {
        "train": target_train_vc,
        "val": target_val_vc,
        "test": target_test_vc,
    }
except FileNotFoundError as e:
    logging.error(
        f"Lỗi: Không tìm thấy một trong các file CSV: {e}. Đảm bảo bạn đã chạy prepare_real_audio.py."
    )
    TARGET_FAKE_CLIPS_PER_SET_VC = {
        "train": None,
        "val": None,
        "test": None,
    }  # Đặt None để không giới hạn nếu lỗi
except ValueError as e:
    logging.error(f"Lỗi dữ liệu: {e}. Vui lòng kiểm tra nội dung các file CSV.")
    TARGET_FAKE_CLIPS_PER_SET_VC = {"train": None, "val": None, "test": None}


def generate_fake_vc_audio(set_type):
    logging.info(f"Bắt đầu tạo audio fake Voice Cloning cho tập '{set_type}'")

    real_audio_paths_filepath = os.path.join(
        DATASET_OUTPUT_ROOT,
        set_type,
        REAL_AUDIO_PATHS_FILE_TEMPLATE.format(set_type=set_type),
    )

    if not os.path.exists(real_audio_paths_filepath):
        logging.error(
            f"File '{real_audio_paths_filepath}' không tồn tại. Hãy chạy prepare_real_audio.py trước."
        )
        return

    df_real_audio_paths = pd.read_csv(real_audio_paths_filepath)

    # Giới hạn số lượng bản ghi nếu có TARGET_FAKE_CLIPS_PER_SET_VC
    max_samples = TARGET_FAKE_CLIPS_PER_SET_VC.get(set_type)
    if max_samples is not None:
        if len(df_real_audio_paths) > max_samples:
            df_real_audio_paths = df_real_audio_paths.sample(n=max_samples, random_state=42)
            logging.info(
                f"Giới hạn số lượng bản ghi để tạo fake audio xuống còn {TARGET_FAKE_CLIPS_PER_SET_VC[set_type]}."
            )

    # Khởi tạo mô hình VC
    logging.info(f"Đang tải mô hình VC: {VC_MODEL_NAME}...")
    try:
        tts = TTS(model_name=VC_MODEL_NAME, gpu=torch.cuda.is_available())
        logging.info("Đã tải mô hình VC thành công.")
    except Exception as e:
        logging.error(f"Không thể tải mô hình VC '{VC_MODEL_NAME}': {e}")
        logging.info("Hãy đảm bảo bạn đã cài đặt Coqui-TTS và mô hình đã được tải về.")
        return

    fake_metadata = []
    fake_clip_counter = ClipIdCounter()

    fake_audio_output_dir = os.path.join(
        DATASET_OUTPUT_ROOT, set_type, "fake", FAKE_AUDIO_TYPE
    )
    os.makedirs(fake_audio_output_dir, exist_ok=True)

    logging.info(
        f"Bắt đầu tạo {len(df_real_audio_paths)} clip fake cho tập '{set_type}'..."
    )
    for index, row in tqdm(
        df_real_audio_paths.iterrows(),
        total=len(df_real_audio_paths),
        desc=f"Generating {FAKE_AUDIO_TYPE} for {set_type}",
    ):
        text = row["text"]
        original_speaker_id = row["speaker_id"]  # Speaker ID của audio real gốc
        text_id = row["text_id"]  # ID của đoạn văn bản gốc
        real_audio_ref_path = row[
            "path"
        ]  # Đường dẫn tương đối đến clip real dùng làm reference

        # Đường dẫn tuyệt đối đến file audio real để làm reference
        full_real_audio_ref_path = os.path.join(
            DATASET_OUTPUT_ROOT, real_audio_ref_path
        )

        if not os.path.exists(full_real_audio_ref_path):
            logging.warning(
                f"File tham chiếu audio real không tồn tại: {full_real_audio_ref_path}. Bỏ qua bản ghi."
            )
            continue

        try:
            # Tạo audio fake
            waveform = tts.tts(text=text, speaker_wav=full_real_audio_ref_path)
            sample_rate = tts.synthesizer.output_sample_rate  # type: ignore

            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform).float()

            clip_metadata = process_and_save_fake_clip(
                waveform=waveform,
                sample_rate=sample_rate,
                output_base_dir=fake_audio_output_dir,
                speaker_id=original_speaker_id,  # Giữ speaker_id gốc
                text_id=text_id,
                clip_id_counter=fake_clip_counter,
                source_dataset_name="LibriTTS",
                synthesis_tool=VC_MODEL_NAME.split("/")[-1],  # Tên công cụ/mô hình
                fake_level=3,  # Cấp độ fake 3 cho voice cloning
                reference_path=real_audio_ref_path,
            )
            fake_metadata.extend(clip_metadata)

        except Exception as e:
            logging.error(
                f"Lỗi khi tạo fake audio VC cho text_id '{text_id}' với ref '{real_audio_ref_path}': {e}"
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

    logging.info(f"Hoàn tất tạo audio fake Voice Cloning cho tập '{set_type}'.")


if __name__ == "__main__":
    for set_type in ["train", "val", "test"]:
        generate_fake_vc_audio(set_type)

    logging.info("Tất cả quá trình tạo audio fake Voice Cloning đã hoàn tất.")
