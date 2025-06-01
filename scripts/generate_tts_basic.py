import os
import pandas as pd
import json
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
DATASET_OUTPUT_ROOT = "dataset"  # Đường dẫn gốc dataset đã xử lý
FAKE_AUDIO_TYPE = "tts_basic"  # Tên thư mục con cho loại fake audio này

# Cấu hình mô hình TTS
# Sử dụng mô hình TTS đơn speaker. Ví dụ: VITS trên ljspeech
TTS_MODEL_NAME = "tts_models/en/ljspeech/vits"

# Đường dẫn đến file CSV chứa các cặp (path_to_real_audio, speaker_id, text, text_id)
REAL_AUDIO_PATHS_FILE_TEMPLATE = "real_audio_paths_for_fake_{set_type}.csv"
REAL_SPEAKER_IDS_FILE_TEMPLATE = "real_speaker_ids_{set_type}.json"

# Số lượng clip fake mục tiêu (mỗi text_id sẽ chỉ được tạo fake 1 lần)
# Đặt None để xử lý tất cả các bản ghi có sẵn trong file CSV.
train_csv_path = "F:\\Deepfake-Audio-Detector\\dataset\\train\\real_audio_paths_for_fake_train.csv"
val_csv_path = "F:\\Deepfake-Audio-Detector\\dataset\\val\\real_audio_paths_for_fake_val.csv"
test_csv_path = "F:\\Deepfake-Audio-Detector\\dataset\\test\\real_audio_paths_for_fake_test.csv"

# Số lượng fake cần tạo cho tập train
target_train = 10000

# 1. Tính tỷ lệ trên tập train
train_df = pd.read_csv(train_csv_path)
total_train_rows = len(train_df)

ratio = target_train / total_train_rows
print(f"Tỷ lệ mẫu được chọn từ train: {ratio:.2%}")

val_df = pd.read_csv(val_csv_path)
test_df = pd.read_csv(test_csv_path)

target_val = int(len(val_df) * ratio)
target_test = int(len(test_df) * ratio)

TARGET_FAKE_CLIPS_PER_SET = {
    "train": target_train,
    "val": target_val,
    "test": target_test,
}


def generate_fake_tts_audio(set_type):
    logging.info(f"Bắt đầu tạo audio fake TTS cơ bản cho tập '{set_type}'")

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

    df_real_audio_paths = pd.read_csv(real_audio_paths_filepath)

    # Giới hạn số lượng bản ghi nếu có TARGET_FAKE_CLIPS_PER_SET
    if TARGET_FAKE_CLIPS_PER_SET.get(set_type) is not None:
        if len(df_real_audio_paths) > TARGET_FAKE_CLIPS_PER_SET[set_type]:
            df_real_audio_paths = df_real_audio_paths.sample(
                n=TARGET_FAKE_CLIPS_PER_SET[set_type], random_state=42
            )
            logging.info(
                f"Giới hạn số lượng bản ghi để tạo fake audio xuống còn {TARGET_FAKE_CLIPS_PER_SET[set_type]}."
            )

    # Load speaker IDs (cho metadata, không dùng để chọn speaker cho TTS này)
    if os.path.exists(real_speaker_ids_filepath):
        with open(real_speaker_ids_filepath, "r") as f:
            real_speaker_ids = json.load(f)
        logging.info(
            f"Đã tải {len(real_speaker_ids)} speaker IDs từ '{real_speaker_ids_filepath}'."
        )
    else:
        logging.warning(
            f"File speaker IDs không tìm thấy: {real_speaker_ids_filepath}. Tiếp tục mà không có danh sách speaker."
        )
        real_speaker_ids = []

    # Khởi tạo mô hình TTS
    logging.info(f"Đang tải mô hình TTS: {TTS_MODEL_NAME}...")
    try:
        # TTS cơ bản, không cần speaker_wav
        tts = TTS(model_name=TTS_MODEL_NAME, gpu=torch.cuda.is_available())
        logging.info("Đã tải mô hình TTS thành công.")
    except Exception as e:
        logging.error(f"Không thể tải mô hình TTS '{TTS_MODEL_NAME}': {e}")
        logging.info("Hãy đảm bảo bạn đã cài đặt Coqui-TTS và mô hình đã được tải về.")
        return

    fake_metadata = []
    fake_clip_counter = ClipIdCounter()

    # Thư mục output cho audio fake của subset hiện tại (e.g., dataset/train/fake/tts_basic)
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

        try:
            # Tạo audio fake
            waveform = tts.tts(text=text)
            sample_rate = getattr(tts, "samplerate", 16000)
            # Lấy sample_rate từ synthesizer

            # waveform từ tts.tts có thể là numpy array, chuyển về torch.Tensor nếu cần
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform).float()

            clip_metadata = process_and_save_fake_clip(
                waveform=waveform,
                sample_rate=sample_rate,
                output_base_dir=fake_audio_output_dir,
                speaker_id=original_speaker_id,  # Giữ speaker_id gốc để dễ nhóm và so sánh
                fake_level=1,  # Mức độ giả mạo: 1 (TTS cơ bản)  
                text_id=text_id,
                clip_id_counter=fake_clip_counter,
                source_dataset_name="LibriTTS",
                synthesis_tool=TTS_MODEL_NAME.split("/")[-1],  # Tên công cụ/mô hình
                reference_path=real_audio_ref_path,
            )
            fake_metadata.extend(clip_metadata)

        except Exception as e:
            logging.error(f"Lỗi khi tạo fake audio TTS cho text_id '{text_id}': {e}")
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

    logging.info(f"Hoàn tất tạo audio fake TTS cơ bản cho tập '{set_type}'.")


if __name__ == "__main__":
    # Chạy cho từng loại tập dữ liệu
    for set_type in ["train", "val", "test"]:
        generate_fake_tts_audio(set_type)

    logging.info("Tất cả quá trình tạo audio fake TTS cơ bản đã hoàn tất.")
