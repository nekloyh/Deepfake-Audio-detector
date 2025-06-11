# audio_segmentation.py
import os
import sys
import librosa
import numpy as np
import warnings
from app.config import settings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# AudioConfig class removed


def segment_audio(
    audio_path: str,
    target_sr: int = settings.TARGET_SAMPLE_RATE,
    segment_duration: float = settings.CHUNK_DURATION_SECONDS,
    overlap_duration: float = settings.SEGMENT_OVERLAP_SECONDS,
) -> list[np.ndarray]:
    """
    Loads an audio file and segments it into fixed-duration chunks.
    Each chunk is either padded with zeros or truncated to `segment_duration`.

    Args:
        audio_path (str): Path to the input audio file (.wav, .flac).
        target_sr (int): Target sample rate for loading.
        segment_duration (float): Target duration of each segment in seconds.
        overlap_duration (float): Overlap between consecutive segments in seconds.

    Returns:
        list[np.ndarray]: A list of numpy arrays, where each array is a segment
                          of `segment_duration` length, sampled at `target_sr`.
                          Returns an empty list if loading fails.
    """
    try:
        y, sr = librosa.load(audio_path, sr=target_sr)

        # Convert segment duration and overlap duration to samples
        segment_samples = int(segment_duration * target_sr)
        overlap_samples = int(overlap_duration * target_sr)
        step_samples = segment_samples - overlap_samples

        total_samples = len(y)
        segments = []

        if total_samples < segment_samples:
            # If audio is shorter than a single segment, pad it
            padded_y = np.pad(y, (0, segment_samples - total_samples), "constant")
            segments.append(padded_y)
        else:
            # Segment the audio
            for i in range(0, total_samples, step_samples):
                segment = y[i : i + segment_samples]
                if len(segment) < segment_samples:
                    # Pad the last segment if it's shorter than target_duration
                    segment = np.pad(
                        segment, (0, segment_samples - len(segment)), "constant"
                    )
                segments.append(segment)

        return segments

    except Exception as e:
        print(f"Error loading or segmenting audio {audio_path}: {e}")
        return []


if __name__ == "__main__":
    # Example Usage:
    print("--- Testing audio_segmentation.py ---")
    try:
        import soundfile as sf

        # Create a dummy 20-second WAV file for testing
        dummy_sr = settings.TARGET_SAMPLE_RATE
        dummy_duration = 20  # seconds
        t = np.linspace(
            0, dummy_duration, int(dummy_sr * dummy_duration), endpoint=False
        )
        dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn(
            int(dummy_sr * dummy_duration)
        )  # Add some noise
        dummy_audio_path = "test_audio_20s.wav"
        sf.write(dummy_audio_path, dummy_audio, dummy_sr)
        print(f"Dummy 20s audio saved to {dummy_audio_path}")

        segments = segment_audio(dummy_audio_path)

        print(f"\nOriginal audio length: {dummy_duration} seconds")
        print(f"Number of segments created: {len(segments)}")
        if segments:
            print(f"Length of first segment (samples): {len(segments[0])}")
            print(
                f"Length of first segment (seconds): {len(segments[0]) / settings.TARGET_SAMPLE_RATE}"
            )
        else:
            print("No segments created.")

        # Test with a shorter file
        dummy_audio_short_path = "test_audio_1_5s.wav"
        dummy_duration_short = 1.5
        t_short = np.linspace(
            0,
            dummy_duration_short,
            int(
                dummy_sr * dummy_duration_short
            ),  # Assuming dummy_sr here, as dummy_sr_short is not defined
            endpoint=False,
        )
        dummy_audio_short = 0.3 * np.sin(2 * np.pi * 880 * t_short)
        sf.write(
            dummy_audio_short_path, dummy_audio_short, dummy_sr
        )  # Assuming dummy_sr here
        print(f"\nDummy 1.5s audio saved to {dummy_audio_short_path}")

        segments_short = segment_audio(dummy_audio_short_path)
        print(f"Number of segments created for short audio: {len(segments_short)}")
        if segments_short:
            print(f"Length of short segment (samples): {len(segments_short[0])}")
            print(
                f"Length of short segment (seconds): {len(segments_short[0]) / settings.TARGET_SAMPLE_RATE}"
            )

    except ImportError:
        print(
            "soundfile not installed. Please install it (`pip install soundfile`) to run the example."
        )
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
    finally:
        import os

        if os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)
        if os.path.exists(dummy_audio_short_path):
            os.remove(dummy_audio_short_path)
        print("\nCleaned up dummy audio files.")
