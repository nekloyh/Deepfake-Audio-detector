import numpy as np
import librosa
from app.config import settings  # Import the global settings


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


# Example Usage (for testing this file directly):
if __name__ == "__main__":
    print(
        f"Audio Utils - Using Settings: SR={settings.TARGET_SAMPLE_RATE}, ChunkSec={settings.CHUNK_DURATION_SECONDS}"
    )

    # Create a dummy audio signal for testing
    dummy_sr = settings.TARGET_SAMPLE_RATE
    duration = (
        settings.CHUNK_DURATION_SECONDS * 2.5
    )  # Make it a bit longer than two chunks

    # Use a simple sine wave for predictability
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
        print("split_into_chunks seems OK.")
    else:
        print("No chunks to test.")

    # Test with a very short audio (shorter than one chunk duration)
    short_duration = settings.CHUNK_DURATION_SECONDS / 2.0
    short_waveform = 0.5 * np.sin(
        2
        * np.pi
        * freq
        * np.linspace(0, short_duration, int(dummy_sr * short_duration), endpoint=False)
    )
    print(f"\nTesting split_into_chunks with short audio ({short_duration}s)...")

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
        print("split_into_chunks with short audio seems OK.")
    else:
        print("No chunks from short audio to test.")

    # Note: waveform_to_mel_spectrogram tests were removed as the function was removed earlier.

    print(
        "\nRelevant local tests for app.audio_processing.utils (split_into_chunks) passed."
    )

print("app.audio_processing.utils loaded.")
