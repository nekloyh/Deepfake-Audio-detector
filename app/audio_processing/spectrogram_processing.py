# spectrogram_processing.py
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import pyloudnorm as pyln
from typing import Optional
from app.config import settings

try:
    from app.config import settings
except ImportError:
    class Settings:
        TARGET_SAMPLE_RATE = 16000
        N_MELS = 128
        N_FFT = 2048
        HOP_LENGTH = 512
        IMAGE_SIZE = 224
        LOUDNESS_LUFS = -23.0
        PIXEL_MEAN = -0.0137
        PIXEL_STD = 0.7317
    settings = Settings()

def create_mel_spectrogram(
    audio_waveform: np.ndarray,
    sr: int = settings.TARGET_SAMPLE_RATE,
    n_mels: int = settings.N_MELS,
    n_fft: int = settings.N_FFT,
    hop_length: int = settings.HOP_LENGTH,
    target_lufs: float = settings.LOUDNESS_LUFS
) -> Optional[np.ndarray]:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_waveform)
    audio_waveform = pyln.normalize.loudness(audio_waveform, loudness, target_lufs)
    if audio_waveform is None or len(audio_waveform) == 0:
        print("Invalid audio waveform: empty or None")
        return None

    # Kiểm tra năng lượng của waveform
    if np.abs(audio_waveform).max() < 1e-5 or np.sum(audio_waveform ** 2) < 1e-6:
        print("Audio waveform has very low energy, likely silent")
        return None

    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_waveform,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0.0,
            fmax=8000.0
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)
        # Kiểm tra độ biến thiên của spectrogram
        # if np.std(log_mel_spectrogram) < 1e-3:
        #     print("Spectrogram has low variance, may lack discriminative features")
        #     return None
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error creating Mel-spectrogram: {e}")
        return None


def preprocess_spectrogram_to_tensor(
    log_mel_spectrogram: np.ndarray,
    image_size: int = settings.IMAGE_SIZE,
    mean: float = settings.PIXEL_MEAN,  # Type will be float from settings
    std: float = settings.PIXEL_STD,  # Type will be float from settings
) -> Optional[torch.Tensor]:
    if log_mel_spectrogram is None or log_mel_spectrogram.size == 0:
        return None
    try:
        mel_spec_tensor = (
            torch.from_numpy(log_mel_spectrogram).float().unsqueeze(0).unsqueeze(0)
        )
        mel_spec_scaled = F.interpolate(
            mel_spec_tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        mel_spec_scaled = mel_spec_scaled.squeeze(0)  # Shape: [1, 224, 224]
        
        # Chuẩn hóa cục bộ 
        mean_val = mel_spec_scaled.mean()
        std_val = mel_spec_scaled.std() + 1e-6
        mel_spec_normalized = (mel_spec_scaled - mean_val) / std_val
        
        # Điều chỉnh nhẹ dựa trên PIXEL_MEAN và PIXEL_STD
        # Nhân với hệ số nhỏ để tránh lệch quá xa
        # Lệch FAKE tăng factor
        # Lệch REAL giảm factor
        std_normalized_factor = 0.19
        mean_normalized_factor = 45.0
        mel_spec_normalized = mel_spec_normalized * (std * std_normalized_factor) + (mean * mean_normalized_factor)
        return mel_spec_normalized
    except Exception as e:
        print(f"Error preprocessing spectrogram to tensor: {e}")
        return None