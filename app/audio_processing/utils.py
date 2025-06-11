import numpy as np
import librosa
from app.config import settings  # Import the global settings


def load_audio(
    audio_path: str, target_sr: int = settings.TARGET_SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    """
    Loads an audio file, resamples to target_sr, and converts to mono.

    Args:
        audio_path (str): Path to the audio file or file-like object.
        target_sr (int): The target sample rate.

    Returns:
        tuple[np.ndarray, int]: A tuple containing the waveform as a NumPy array
                                and the actual sample rate after loading (should be target_sr).
    """
    try:
        # Librosa can handle file paths or already opened file-like objects
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return waveform, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise


def split_into_chunks(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_duration_sec: float = settings.CHUNK_DURATION_SECONDS,
) -> list[np.ndarray]:
    """
    Splits the waveform into chunks of specified duration.

    Args:
        waveform (np.ndarray): The input audio waveform.
        sample_rate (int): The sample rate of the waveform.
        chunk_duration_sec (float): The duration of each chunk in seconds.

    Returns:
        list[np.ndarray]: A list of NumPy arrays, where each array is an audio chunk.
    """
    samples_per_chunk = int(sample_rate * chunk_duration_sec)
    if samples_per_chunk == 0:  # Should not happen with reasonable settings
        print(
            "Warning: samples_per_chunk is 0. Check sample_rate and chunk_duration_sec."
        )
        return [waveform] if len(waveform) > 0 else []

    num_chunks = int(np.ceil(len(waveform) / samples_per_chunk))
    chunks = []

    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = waveform[start:end]

        # Pad the last chunk if it's shorter
        if len(chunk) < samples_per_chunk:
            padding = samples_per_chunk - len(chunk)
            # Pad with zeros (silence)
            chunk = np.pad(chunk, (0, padding), "constant", constant_values=0.0)

        chunks.append(chunk)
    return chunks


def waveform_to_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = settings.TARGET_SAMPLE_RATE,
    n_mels: int = settings.N_MELS,
    n_fft: int = settings.N_FFT,
    hop_length: int = settings.HOP_LENGTH,
    target_width: int = settings.SPECTROGRAM_WIDTH,
    min_db_level: float = settings.MIN_DB_LEVEL,
    top_db: float = settings.TOP_DB,
) -> np.ndarray:
    """
    Converts a single audio waveform chunk to a Mel-spectrogram, normalizes, and resizes it.

    Args:
        waveform (np.ndarray): Input audio waveform chunk.
        sr (int): Sample rate of the waveform.
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        target_width (int): Target width of the spectrogram (time axis).
        min_db_level (float): Minimum dB level for normalization reference.
        top_db (float): top_db for librosa.power_to_db, effectively clipping louder sounds.

    Returns:
        np.ndarray: Processed Mel-spectrogram with shape (n_mels, target_width), dtype float32.
    """
    # 1. Mel Spectrogram Calculation using Librosa
    mel_spec_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,  # Default
        fmax=sr / 2.0,  # Default
        power=2.0,  # Compute power spectrogram (amplitude squared)
    )

    # 2. Amplitude to dB
    # ref=np.max sets the loudest part of the signal to 0dB (or close to it).
    # top_db means that any signal components quieter than (max_signal - top_db) will be floored.
    # E.g. if max is 0dB and top_db is 80, values range from -80dB to 0dB.
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max, top_db=top_db)

    # 3. Normalization to [0, 1] range
    # Values in mel_spec_db are expected to be roughly in [max_val - top_db, max_val].
    # After ref=np.max, max_val is close to 0. So, range is approx [-top_db, 0].
    # We want to map [min_db_level, some_max_db_near_zero] to [0, 1].
    # The provided min_db_level is the floor.
    spec_db_clipped = np.maximum(mel_spec_db, min_db_level)

    # Normalize: (value - min) / (max - min)
    # Here, min is min_db_level. Max is effectively 0 dB or slightly higher if signal is loud.
    # We use -min_db_level as the range, assuming max_db is effectively 0 for normalization scaling.
    # (value - min_db_level) / (0 - min_db_level)
    spec_norm = (spec_db_clipped - min_db_level) / (-min_db_level)
    spec_norm = np.clip(spec_norm, 0, 1)  # Ensure values are strictly within [0, 1]

    # 4. Pad or Truncate width (time axis)
    current_width = spec_norm.shape[1]
    if current_width < target_width:
        pad_amount = target_width - current_width
        # Pad with the normalized equivalent of min_db_level (which is 0 after normalization)
        spec_padded = np.pad(
            spec_norm, ((0, 0), (0, pad_amount)), "constant", constant_values=0.0
        )
    elif current_width > target_width:
        # Truncate from the end
        spec_padded = spec_norm[:, :target_width]
    else:
        spec_padded = spec_norm

    if spec_padded.shape != (n_mels, target_width):
        print(
            f"Warning: Spectrogram shape after padding/truncation is {spec_padded.shape}, expected {(n_mels, target_width)}. This might occur with very short audio inputs leading to few frames."
        )
        # Fallback: if there are too few frames, pad again. This shouldn't be common.
        # This usually indicates an issue with chunking or extremely short audio not handled well by FFT.
        # For now, we assume the previous padding is generally correct.
        # If spec_padded.shape[1] < target_width, it implies an issue in logic or input.
        # A robust fix might involve reshaping with fixed output size if librosa utils were used,
        # or ensuring `y` to melspectrogram has enough samples.
        # For now, we'll rely on the padding done.

    return spec_padded.astype(np.float32)  # Ensure float32 for ONNX models


# Example Usage (for testing this file directly):
if __name__ == "__main__":
    print(
        f"Audio Utils - Using Settings: SR={settings.TARGET_SAMPLE_RATE}, ChunkSec={settings.CHUNK_DURATION_SECONDS}, N_MELS={settings.N_MELS}, SpecWidth={settings.SPECTROGRAM_WIDTH}, N_FFT={settings.N_FFT}, HOP_LENGTH={settings.HOP_LENGTH}, MIN_DB={settings.MIN_DB_LEVEL}, TOP_DB={settings.TOP_DB}"
    )

    # Create a dummy audio signal for testing
    dummy_sr = settings.TARGET_SAMPLE_RATE
    duration = (
        settings.CHUNK_DURATION_SECONDS * 2.5
    )  # Make it a bit longer than two chunks

    # Use a simple sine wave for predictability, chirp can have rapid changes
    freq = 440  # A4 note
    dummy_waveform = 0.5 * np.sin(
        2
        * np.pi
        * freq
        * np.linspace(0, duration, int(dummy_sr * duration), endpoint=False)
    )

    print(
        f"\nOriginal dummy waveform shape: {dummy_waveform.shape}, SR: {dummy_sr}, Duration: {duration}s"
    )

    # Test split_into_chunks
    chunks = split_into_chunks(
        dummy_waveform, dummy_sr
    )  # Uses CHUNK_DURATION_SECONDS from settings
    print(f"Number of chunks: {len(chunks)}")
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    if chunks:
        expected_samples_per_chunk = int(dummy_sr * settings.CHUNK_DURATION_SECONDS)
        print(f"Expected samples per chunk: {expected_samples_per_chunk}")
        for i, chunk in enumerate(chunks):
            print(f"Shape of chunk {i}: {chunk.shape}")
            assert chunk.shape[0] == expected_samples_per_chunk, (
                f"Chunk {i} length mismatch"
            )

    # Test waveform_to_mel_spectrogram on the first chunk
    if chunks:
        first_chunk = chunks[0]
        print(f"\nProcessing first chunk with shape {first_chunk.shape}...")
        mel_spec = waveform_to_mel_spectrogram(
            first_chunk, sr=dummy_sr
        )  # Uses other relevant settings

        print(f"Mel spectrogram shape: {mel_spec.shape}")
        assert mel_spec.shape == (settings.N_MELS, settings.SPECTROGRAM_WIDTH), (
            "Mel spectrogram shape mismatch"
        )
        print(f"Mel spectrogram dtype: {mel_spec.dtype}")
        assert mel_spec.dtype == np.float32, "Mel spectrogram dtype mismatch"

        min_val, max_val = np.min(mel_spec), np.max(mel_spec)
        print(f"Mel spectrogram min/max values: {min_val}, {max_val}")
        assert min_val >= 0.0 and max_val <= 1.000001, (
            f"Mel spectrogram normalization out of [0,1] range: min={min_val}, max={max_val}"
        )  # Add small epsilon for float comparisons

        if np.all(mel_spec == 0):
            print(
                "Warning: Resulting spectrogram is all zeros. This might be okay for silence, but check dummy signal if unexpected."
            )

        print("First chunk spectrogram processing seems OK.")
    else:
        print("No chunks to test spectrogram conversion.")

    # Test with a very short audio (shorter than one chunk duration)
    short_duration = settings.CHUNK_DURATION_SECONDS / 2.0
    short_waveform = 0.5 * np.sin(
        2
        * np.pi
        * freq
        * np.linspace(0, short_duration, int(dummy_sr * short_duration), endpoint=False)
    )
    print(f"\nTesting with short audio ({short_duration}s)...")

    chunks_short = split_into_chunks(short_waveform, dummy_sr)
    print(f"Number of chunks from short audio: {len(chunks_short)}")
    assert len(chunks_short) == 1, (
        f"Expected 1 chunk for short audio, got {len(chunks_short)}"
    )

    if chunks_short:
        print(f"Shape of the chunk (short audio): {chunks_short[0].shape}")
        assert chunks_short[0].shape[0] == int(
            dummy_sr * settings.CHUNK_DURATION_SECONDS
        ), "Short audio chunk length mismatch (should be padded)"

        mel_spec_short = waveform_to_mel_spectrogram(chunks_short[0], sr=dummy_sr)
        print(f"Mel spectrogram shape (short audio): {mel_spec_short.shape}")
        assert mel_spec_short.shape == (settings.N_MELS, settings.SPECTROGRAM_WIDTH), (
            "Short audio Mel spectrogram shape mismatch"
        )
        min_val_s, max_val_s = np.min(mel_spec_short), np.max(mel_spec_short)
        print(f"Mel spectrogram min/max values (short audio): {min_val_s}, {max_val_s}")
        assert min_val_s >= 0.0 and max_val_s <= 1.000001, (
            f"Short audio Mel spectrogram normalization out of [0,1] range: min={min_val_s}, max={max_val_s}"
        )

        print("Short audio processing seems OK.")
    else:
        print("No chunks from short audio to test.")

    print("\nAll local tests for app.audio_processing.utils passed.")

print("app.audio_processing.utils loaded and tested if run directly.")
