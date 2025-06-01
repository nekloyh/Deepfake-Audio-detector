import os
import torchaudio.transforms as T
import logging
from pydub import AudioSegment
import torch  # Import torch

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cấu hình chung (phải khớp với prepare_real_audio.py)
TARGET_SR = 16000  # Tốc độ lấy mẫu mục tiêu


def process_and_save_fake_clip(
    waveform,
    sample_rate,
    output_base_dir,
    speaker_id,
    text_id,  # ID của văn bản gốc (thường là audio_base_id của real audio)
    clip_id_counter,  # Đối tượng đếm ID duy nhất cho clip fake
    source_dataset_name,  # Tên dataset tạo fake (ví dụ: 'LibriTTS')
    synthesis_tool,  # Tên công cụ/mô hình tạo fake (ví dụ: 'YourTTS', 'Vits')
    reference_path=None,  # Đường dẫn tương đối đến audio real gốc dùng làm reference (tùy chọn)
    fake_level=1
):
    """
    Xử lý (resample nếu cần) và lưu clip audio fake.
    Trả về metadata của clip đã lưu.
    """
    processed_clips_info = []
    try:
        # Đảm bảo waveform là torch.Tensor và đúng định dạng
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform).float()  # Chuyển sang float tensor

        # Nếu waveform có batch dimension, loại bỏ nó (TTS thường trả về [1, samples] hoặc [samples])
        if waveform.ndim > 1 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)  # Loại bỏ batch dimension

        # Resample về TARGET_SR nếu cần
        if sample_rate != TARGET_SR:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
            waveform = resampler(waveform)
            sample_rate = TARGET_SR  # Cập nhật sample_rate sau khi resample

        # Chuyển torchaudio tensor sang numpy array, sau đó sang AudioSegment
        audio_np = waveform.numpy()

        audio_segment = AudioSegment(
            audio_np.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_np.dtype.itemsize,
            channels=1,  # Luôn là mono
        )

        # Chuẩn hóa âm lượng tương tự như real audio
        target_dBFS = -3.0
        change_in_dBFS = target_dBFS - audio_segment.dBFS
        normalized_audio_segment = audio_segment.apply_gain(change_in_dBFS)

        # Tạo thư mục output cho speaker trong cấu trúc output_base_dir/speaker_id/
        speaker_output_dir = os.path.join(
            output_base_dir, str(speaker_id)
        )  # Đảm bảo speaker_id là string
        os.makedirs(speaker_output_dir, exist_ok=True)

        # Tạo ID duy nhất cho clip fake
        unique_fake_clip_id = next(clip_id_counter)
        # Tên file: speakerID_textID_uniqueID.wav
        # text_id ở đây chính là audio_base_id gốc từ real audio
        output_filepath = os.path.join(
            speaker_output_dir, f"{speaker_id}_{text_id}_{unique_fake_clip_id:06d}.wav"
        )
        normalized_audio_segment.export(output_filepath, format="wav")

        processed_clips_info.append(
            {
                "path": os.path.relpath(
                    output_filepath, os.path.dirname(os.path.dirname(output_base_dir))
                ),  # Đường dẫn tương đối từ 'dataset'
                "label": "fake",
                "speaker_id": speaker_id,
                "fake_level": fake_level,  # Mức độ giả mạo: 1 (TTS cơ bản), 2 (Voice Cloning)
                "duration": len(normalized_audio_segment) / 1000.0,
                "source_dataset": source_dataset_name,
                "synthesis_tool": synthesis_tool,
                "text_id": text_id,  # Giữ lại text_id để truy vết đến transcript gốc
                "reference_path": reference_path
                if reference_path
                else None,  # Lưu lại đường dẫn ref audio nếu có
            }
        )
    except Exception as e:
        logging.error(f"Lỗi khi xử lý và lưu fake audio clip: {e}")
    return processed_clips_info


# Hàm quản lý ID duy nhất cho clip (Giống trong prepare_real_audio)
class ClipIdCounter:
    def __init__(self):
        self._counter = 0

    def __next__(self):
        self._counter += 1
        return self._counter

    def get_current(self):
        return self._counter
