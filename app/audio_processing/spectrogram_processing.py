# spectrogram_processing.py
import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
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
        PIXEL_MEAN = -0.0137
        PIXEL_STD = 0.7317
    settings = Settings()

def create_mel_spectrogram(
    audio_waveform: np.ndarray,
    sr: int = settings.TARGET_SAMPLE_RATE,
    n_mels: int = settings.N_MELS,
    n_fft: int = settings.N_FFT,
    hop_length: int = settings.HOP_LENGTH,
) -> Optional[np.ndarray]:
    if audio_waveform is None or len(audio_waveform) == 0:
        print("Invalid audio waveform: empty or None")
        return None

    # Kiểm tra năng lượng của waveform
    energy = np.sum(audio_waveform ** 2)
    if energy < 1e-6:
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
        if np.std(log_mel_spectrogram) < 1e-3:
            print("Spectrogram has low variance, may lack discriminative features")
            return None
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
    """
        Normalizes the spectrogram, resizes it, and transforms it to a single-channel PyTorch tensor.
        
        Args:
            log_mel_spectrogram (np.ndarray): Log Mel spectrogram.
            image_size (int): Target square image size (e.g., 224 for 224x224).
            mean (float): Mean value for normalization (single channel).
            std (float): Standard deviation value for normalization (single channel).
        
        Returns:
            torch.Tensor: Preprocessed spectrogram as a single-channel PyTorch tensor ([1, H, W]).
                        Returns None if input spectrogram is invalid.
        """
    if log_mel_spectrogram is None or log_mel_spectrogram.size == 0:
        return None

    try:
        # Convert to tensor and resize
        mel_spec_tensor = torch.from_numpy(log_mel_spectrogram).float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, n_mels, frames]
        mel_spec_scaled = F.interpolate(
            mel_spec_tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False
        )  # Shape: [1, 1, 224, 224]

        # Per-sample normalization
        mel_spec_scaled = mel_spec_scaled.squeeze(0)  # Shape: [1, 224, 224]
        mean_val = mel_spec_scaled.mean()
        std_val = mel_spec_scaled.std() + 1e-6  # Avoid division by zero
        mel_spec_normalized = (mel_spec_scaled - mean_val) / std_val  # Shape: [1, 224, 224]

        # Apply image normalization for single channel
        normalize = transforms.Normalize(mean=[mean], std=[std])  # Single-channel normalization
        image_tensor = normalize(mel_spec_normalized)  # Shape: [1, 224, 224]

        return image_tensor
    except Exception as e:
        print(f"Error preprocessing spectrogram to tensor: {e}")
        return None