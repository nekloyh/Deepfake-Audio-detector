import os
import pandas as pd
from pydub import AudioSegment
import torchaudio
import torchaudio.transforms as T
import random
import json
import logging
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Cấu hình chung ---
TARGET_SR = 16000  # Tốc độ lấy mẫu mục tiêu
MIN_LEN_SEC = 5  # Độ dài clip tối thiểu (giây)
MAX_LEN_SEC = 10  # Độ dài clip tối đa (giây)
MIN_LEN_MS = MIN_LEN_SEC * 1000
MAX_LEN_MS = MAX_LEN_SEC * 1000

# Đường dẫn gốc đến dữ liệu thô LibriTTS
RAW_DATA_ROOT = "raw_data"
LIBRI_TTS_DIR_NAME = "LibriTTS"  # Thư mục chứa các subset LibriTTS

# Đường dẫn gốc đến dataset đã xử lý
DATASET_OUTPUT_ROOT = "dataset"

# Cấu hình các subset LibriTTS và ánh xạ tới loại tập dữ liệu (train/val/test)
LIBRI_TTS_SUBSET_MAPPING = {
    "train-clean-100": "train",
    "train-clean-360": "train",
    "dev-clean": "val",
    "test-clean": "test",
}


# --- Hàm xử lý audio (CÓ THAY ĐỔI NHỎ) ---
def process_audio_clip(
    input_path,
    output_base_dir,
    speaker_id,
    clip_base_name,
    source_dataset_name,
    clip_counter,
):
    """
    Cắt, chuẩn hóa, và lưu một clip audio.
    input_path: Đường dẫn đến file audio gốc.
    output_base_dir: Thư mục để lưu clip đã xử lý (ví dụ: dataset/train/real/LibriTTS).
    speaker_id: ID của speaker.
    clip_base_name: Tên cơ sở của clip gốc (ví dụ: '27_124992_000000_000002').
    source_dataset_name: Tên dataset gốc (ví dụ: 'LibriTTS').
    clip_counter: Đối tượng đếm số lượng clip để tạo ID duy nhất.
    """
    processed_clips_info = []
    try:
        audio = AudioSegment.from_file(input_path)

        # Chuyển về mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Cắt thành các đoạn 5-10s
        if len(audio) < MIN_LEN_MS:
            # logging.debug(f"Clip quá ngắn, bỏ qua: {input_path}") # Tắt debug ở đây
            return []

        segments = []
        current_pos_ms = 0
        while current_pos_ms < len(audio):
            remaining_len_ms = len(audio) - current_pos_ms
            if remaining_len_ms < MIN_LEN_MS:
                break

            segment_len_ms = random.randint(
                MIN_LEN_MS, min(MAX_LEN_MS, remaining_len_ms)
            )

            segment = audio[current_pos_ms : current_pos_ms + segment_len_ms]
            segments.append(segment)

            current_pos_ms += segment_len_ms

        for i, segment in enumerate(segments):
            normalized_segment = segment.set_frame_rate(TARGET_SR).normalize(
                headroom=-3.0
            )

            speaker_output_dir = os.path.join(output_base_dir, speaker_id)
            os.makedirs(speaker_output_dir, exist_ok=True)

            unique_clip_id = next(clip_counter)
            # Tên file: speakerID_originalBaseName_uniqueID.wav (Ví dụ: 27_27_124992_000000_000002_000001.wav)
            # clip_base_name là "27_124992_000000_000002"
            output_filepath = os.path.join(
                speaker_output_dir, f"{clip_base_name}_{unique_clip_id:06d}.wav"
            )  # Đã bỏ speaker_id lặp lại
            normalized_segment.export(output_filepath, format="wav")

            waveform, sample_rate = torchaudio.load(output_filepath)
            if sample_rate != TARGET_SR:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
                waveform = resampler(waveform)
                torchaudio.save(output_filepath, waveform, TARGET_SR)

            processed_clips_info.append(
                {
                    "path": os.path.relpath(output_filepath, DATASET_OUTPUT_ROOT),
                    "label": "real",
                    "speaker_id": speaker_id,
                    "fake_level": 0,
                    "duration": len(normalized_segment) / 1000.0,
                    "source_dataset": source_dataset_name,
                    "audio_base_id": clip_base_name,  # clip_base_name ở đây là "27_124992_000000_000002"
                }
            )
    except Exception as e:
        logging.error(f"Lỗi khi xử lý {input_path}: {e}")
    return processed_clips_info


# --- Xử lý LibriTTS Audio Files cho từng subset (CÓ THAY ĐỔI NHỎ) ---
def process_libri_tts_subset(
    libri_tts_root,
    output_base_dir_for_set,
    subset_name,
    clip_counter,
    target_real_clips_count=None,
):
    logging.info(
        f"Bắt đầu xử lý LibriTTS subset '{subset_name}' từ {os.path.join(libri_tts_root, subset_name)}"
    )
    all_metadata = []

    real_audio_output_dir = os.path.join(
        output_base_dir_for_set, "real", LIBRI_TTS_DIR_NAME
    )
    os.makedirs(real_audio_output_dir, exist_ok=True)

    subset_actual_path = os.path.join(libri_tts_root, subset_name)

    if not os.path.exists(subset_actual_path):
        logging.warning(
            f"Thư mục LibriTTS subset '{subset_name}' không tồn tại: {subset_actual_path}. Bỏ qua."
        )
        return []

    all_audio_files = []
    # LibriTTS cấu trúc: root/subset_name/speaker_id/chapter_id/speaker_id_chapter_id_segment_id.wav
    for speaker_id in os.listdir(subset_actual_path):
        speaker_dir = os.path.join(subset_actual_path, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue
        for chapter_id in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter_id)
            if not os.path.isdir(chapter_dir):
                continue
            for file_name in os.listdir(chapter_dir):
                if file_name.endswith(".wav"):
                    # clip_base_name: "27_124992_000000_000002" (tên file gốc không bao gồm đuôi)
                    clip_base_name = file_name.replace(".wav", "")
                    # KHÔNG CẦN sửa clip_base_name để bỏ speaker_id lặp lại ở đây
                    # Vì tên file gốc đã là speaker_id_chapter_id_segment_id.wav
                    all_audio_files.append(
                        (
                            os.path.join(chapter_dir, file_name),
                            speaker_id,
                            clip_base_name,
                        )
                    )

    random.shuffle(all_audio_files)
    logging.info(
        f"Tìm thấy tổng cộng {len(all_audio_files)} file audio từ subset '{subset_name}'."
    )

    processed_count = 0
    for input_filepath, speaker_id, clip_base_name in tqdm(
        all_audio_files, desc=f"Processing {subset_name} files"
    ):
        if (
            target_real_clips_count is not None
            and processed_count >= target_real_clips_count
        ):
            logging.info(
                f"Đã đạt số lượng clip real mục tiêu ({target_real_clips_count}) cho subset '{subset_name}'. Dừng xử lý."
            )
            break

        processed_clips = process_audio_clip(
            input_filepath,
            real_audio_output_dir,
            speaker_id,
            clip_base_name,
            "LibriTTS",
            clip_counter,
        )
        all_metadata.extend(processed_clips)
        processed_count += len(processed_clips)

    logging.info(f"Đã tạo {processed_count} clip real từ subset '{subset_name}'.")
    return all_metadata


# --- Hàm đọc transcript từ file .normalized.txt (GIỮ NGUYÊN) ---
# Hàm này sẽ được gọi trực tiếp trong main loop
def load_transcript_from_normalized_txt(wav_path):
    """
    Đọc transcript từ file .normalized.txt dựa trên đường dẫn file .wav.
    Ví dụ: Từ 'path/to/1089_134686_000001_000001.wav', đọc file 'path/to/1089_134686_000001_000001.normalized.txt'.
    """
    try:
        txt_path = wav_path.replace(".wav", ".normalized.txt")
        if not os.path.exists(txt_path):
            # logging.warning(f"File .normalized.txt không tồn tại: {txt_path}") # Tắt cảnh báo quá nhiều
            return None

        with open(txt_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
        return transcript
    except Exception as e:
        logging.error(f"Lỗi khi đọc transcript từ {txt_path}: {e}")
        return None


# --- Hàm quản lý ID duy nhất cho clip (GIỮ NGUYÊN) ---
class ClipIdCounter:
    def __init__(self):
        self._counter = 0

    def __next__(self):
        self._counter += 1
        return self._counter

    def get_current(self):
        return self._counter


if __name__ == "__main__":
    libri_tts_raw_path = os.path.join(RAW_DATA_ROOT, LIBRI_TTS_DIR_NAME)

    TARGET_REAL_CLIPS_PER_SUBSET = {
        "train-clean-100": 15000,
        "train-clean-360": 15000,
        "dev-clean": None,
        "test-clean": None,
    }

    all_real_speaker_ids_by_set = {}

    for libri_tts_subset_name, dataset_set_type in LIBRI_TTS_SUBSET_MAPPING.items():
        logging.info(
            f"\n--- Xử lý tập '{dataset_set_type}' từ LibriTTS subset '{libri_tts_subset_name}' ---"
        )

        current_clip_counter = ClipIdCounter()

        output_dir_for_set = os.path.join(DATASET_OUTPUT_ROOT, dataset_set_type)
        os.makedirs(output_dir_for_set, exist_ok=True)

        real_metadata_for_subset = process_libri_tts_subset(
            libri_tts_raw_path,
            output_dir_for_set,
            libri_tts_subset_name,
            current_clip_counter,
            target_real_clips_count=TARGET_REAL_CLIPS_PER_SUBSET.get(
                libri_tts_subset_name
            ),
        )

        df_real_metadata_for_subset = pd.DataFrame(real_metadata_for_subset)

        if df_real_metadata_for_subset.empty:
            logging.warning(
                f"Không có clip real nào được tạo cho tập '{dataset_set_type}' từ subset '{libri_tts_subset_name}'. Bỏ qua."
            )
            continue

        real_metadata_filepath = os.path.join(
            output_dir_for_set, f"real_audio_metadata_{dataset_set_type}.csv"
        )
        df_real_metadata_for_subset.to_csv(real_metadata_filepath, index=False)
        logging.info(
            f"Metadata real audio cho tập '{dataset_set_type}' đã lưu tại: {real_metadata_filepath}"
        )

        current_real_speaker_ids = (
            df_real_metadata_for_subset["speaker_id"].unique().tolist()
        )
        if dataset_set_type in all_real_speaker_ids_by_set:
            all_real_speaker_ids_by_set[dataset_set_type].extend(
                current_real_speaker_ids
            )
            all_real_speaker_ids_by_set[dataset_set_type] = list(
                set(all_real_speaker_ids_by_set[dataset_set_type])
            )
        else:
            all_real_speaker_ids_by_set[dataset_set_type] = current_real_speaker_ids

        logging.info(
            f"Tổng số speaker real độc đáo cho tập '{dataset_set_type}': {len(all_real_speaker_ids_by_set[dataset_set_type])}"
        )

        # Thêm transcript từ file .normalized.txt (ĐÃ CẬP NHẬT LOGIC LẤY PATH GỐC)
        transcripts = []
        # Duyệt qua từng bản ghi trong metadata đã tạo (df_real_metadata_for_subset)
        # để tìm đường dẫn audio GỐC và từ đó tìm file .normalized.txt

        # Bằng cách này, chúng ta đảm bảo chỉ lấy transcript cho những audio_base_id
        # mà chúng ta thực sự đã tạo ra clip real 5-10s
        unique_audio_base_ids_in_processed_clips = df_real_metadata_for_subset[
            ["speaker_id", "audio_base_id"]
        ].drop_duplicates()

        for _, row in tqdm(
            unique_audio_base_ids_in_processed_clips.iterrows(),
            total=len(unique_audio_base_ids_in_processed_clips),
            desc=f"Loading transcripts for {dataset_set_type}",
        ):
            audio_base_id = row["audio_base_id"]  # Ví dụ: 27_124992_000000_000002
            speaker_id = row["speaker_id"]  # Ví dụ: 27

            # Tách chapter_id từ audio_base_id
            # 27_124992_000000_000002 -> chapter_id = 124992 (phần tử thứ 2)
            parts = audio_base_id.split("_")
            if len(parts) >= 2:
                chapter_id = parts[1]
            else:
                logging.warning(
                    f"Không thể trích xuất chapter_id từ audio_base_id: {audio_base_id}. Bỏ qua."
                )
                continue

            # Tái tạo đường dẫn file .wav GỐC trong thư mục RAW_DATA_ROOT
            # Ví dụ: raw_data/LibriTTS/train-clean-100/27/124992/27_124992_000000_000002.wav
            wav_path_in_raw_data = os.path.join(
                libri_tts_raw_path,
                libri_tts_subset_name,  # train-clean-100
                speaker_id,  # 27
                chapter_id,  # 124992
                f"{audio_base_id}.wav",  # Tên file gốc: 27_124992_000000_000002.wav
            )

            transcript = load_transcript_from_normalized_txt(wav_path_in_raw_data)
            if transcript is not None:
                transcripts.append(
                    {
                        "audio_base_id": audio_base_id,  # Đây là ID gốc của transcript
                        "text": transcript,
                        "speaker_id": speaker_id,
                    }
                )

        df_transcripts_for_subset = pd.DataFrame(transcripts)

        if df_transcripts_for_subset.empty:
            logging.warning(
                f"Không có transcripts nào được tải cho tập '{dataset_set_type}' từ subset '{libri_tts_subset_name}'. Bỏ qua."
            )
            continue

        logging.debug(f"Số bản ghi metadata: {len(df_real_metadata_for_subset)}")
        logging.debug(f"Số bản ghi transcripts: {len(df_transcripts_for_subset)}")
        logging.debug(
            f"Mẫu audio_base_id (metadata): {df_real_metadata_for_subset['audio_base_id'].head().tolist()}"
        )
        logging.debug(
            f"Mẫu audio_base_id (transcripts): {df_transcripts_for_subset['audio_base_id'].head().tolist()}"
        )

        # Merge metadata và transcripts
        # Merge dựa trên speaker_id và audio_base_id (mà giờ là text_id)
        df_real_audio_paths_for_fake = pd.merge(
            df_real_metadata_for_subset,
            df_transcripts_for_subset,
            on=["speaker_id", "audio_base_id"],  # Match cả speaker_id và audio_base_id
            how="inner",
        )

        logging.debug(
            f"Số bản ghi sau merge (trước drop_duplicates): {len(df_real_audio_paths_for_fake)}"
        )

        # Chúng ta chỉ cần một dòng duy nhất cho mỗi 'audio_base_id' (tức mỗi transcript gốc)
        # để đưa vào các mô hình tạo fake.
        # Giữ lại các cột cần thiết và đổi tên 'audio_base_id' thành 'text_id'
        df_real_audio_paths_for_fake = df_real_audio_paths_for_fake.drop_duplicates(
            subset=["audio_base_id"]
        )
        df_real_audio_paths_for_fake.rename(
            columns={"audio_base_id": "text_id"}, inplace=True
        )
        df_real_audio_paths_for_fake = df_real_audio_paths_for_fake[
            ["path", "speaker_id", "text", "text_id"]
        ]

        fake_paths_filepath = os.path.join(
            output_dir_for_set, f"real_audio_paths_for_fake_{dataset_set_type}.csv"
        )
        df_real_audio_paths_for_fake.to_csv(fake_paths_filepath, index=False)
        logging.info(
            f"Đã chuẩn bị {len(df_real_audio_paths_for_fake)} đường dẫn audio real và transcripts để tạo fake audio cho tập '{dataset_set_type}'."
        )
        logging.info(f"File này đã lưu tại: {fake_paths_filepath}")

    for set_type, speaker_ids in all_real_speaker_ids_by_set.items():
        with open(
            os.path.join(
                DATASET_OUTPUT_ROOT, set_type, f"real_speaker_ids_{set_type}.json"
            ),
            "w",
        ) as f:
            json.dump(speaker_ids, f, indent=4)
        logging.info(
            f"Danh sách speaker IDs cho tập '{set_type}' đã lưu tại: {os.path.join(DATASET_OUTPUT_ROOT, set_type, f'real_speaker_ids_{set_type}.json')}"
        )

    logging.info("Quy trình chuẩn bị audio real hoàn tất.")
