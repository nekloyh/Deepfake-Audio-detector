# spectrogram_processing.py
import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from app.config import settings

# SpectrogramConfig class removed


def create_mel_spectrogram(
    audio_waveform: np.ndarray,
    sr: int = settings.TARGET_SAMPLE_RATE,
    n_mels: int = settings.N_MELS,
    n_fft: int = settings.N_FFT,
    hop_length: int = settings.HOP_LENGTH,
) -> np.ndarray:
    """
    Creates a Mel spectrogram from an audio waveform.
    Matches the core logic from 7_convert_mel-spectrograms.ipynb.

    Args:
        audio_waveform (np.ndarray): Audio waveform.
        sr (int): Sample rate.
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: Log Mel spectrogram (dB scale).
    """
    if audio_waveform is None or len(audio_waveform) == 0:
        return None

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram


def preprocess_spectrogram_to_tensor(
    log_mel_spectrogram: np.ndarray,
    image_size: int = settings.IMAGE_SIZE,
    mean: list = settings.PIXEL_MEAN,  # Type will be List[float] from settings
    std: list = settings.PIXEL_STD,  # Type will be List[float] from settings
) -> torch.Tensor:
    """
    Normalizes the spectrogram, converts it to an image, resizes, and transforms to a PyTorch tensor.
    Matches image processing steps for ViT input.

    Args:
        log_mel_spectrogram (np.ndarray): Log Mel spectrogram.
        image_size (int): Target square image size (e.g., 224 for 224x224).
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        torch.Tensor: Preprocessed spectrogram as a PyTorch tensor (C, H, W).
                      Returns None if input spectrogram is invalid.
    """
    if log_mel_spectrogram is None:
        return None

    # Normalize spectrogram to [0, 255] and convert to uint8 for PIL Image
    min_val = log_mel_spectrogram.min()
    max_val = log_mel_spectrogram.max()
    if max_val == min_val:  # Handle cases of flat spectrograms
        # Create a black image if all values are the same
        normalized_spectrogram = np.zeros_like(log_mel_spectrogram, dtype=np.uint8)
    else:
        normalized_spectrogram = (
            (log_mel_spectrogram - min_val) / (max_val - min_val) * 255
        ).astype(np.uint8)

    # Convert to PIL Image and ensure 3 channels (RGB)
    image = Image.fromarray(normalized_spectrogram)
    image = image.convert("RGB")

    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    image_tensor = transform(image)
    return image_tensor


if __name__ == "__main__":
    print("--- Testing spectrogram_processing.py ---")
    # Create a dummy audio waveform (e.g., a 3-second segment)
    sr = settings.TARGET_SAMPLE_RATE
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    dummy_waveform = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.2 * np.random.randn(
        len(t)
    )  # 1kHz sine wave + noise

    print(f"Dummy waveform shape: {dummy_waveform.shape}")

    # Step 1: Create Mel spectrogram
    mel_spec = create_mel_spectrogram(dummy_waveform)
    if mel_spec is not None:
        print(f"Mel spectrogram shape: {mel_spec.shape}")

        # Step 2: Preprocess spectrogram to tensor
        input_tensor = preprocess_spectrogram_to_tensor(mel_spec)
        if input_tensor is not None:
            print(f"Preprocessed tensor shape: {input_tensor.shape}")
            print(f"Preprocessed tensor data type: {input_tensor.dtype}")
            # You can optionally save the processed image for visual inspection
            # processed_image = transforms.ToPILImage()(input_tensor)
            # processed_image.save("processed_spectrogram_test.png")
            # print("Processed spectrogram image saved as processed_spectrogram_test.png")
        else:
            print("Failed to preprocess spectrogram to tensor.")
    else:
        print("Failed to create Mel spectrogram.")
