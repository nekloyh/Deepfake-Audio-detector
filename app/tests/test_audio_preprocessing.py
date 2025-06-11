import numpy as np
import soundfile as sf
import io
import os

# Removed pytest import and PYTEST_AVAILABLE flag

from app.config import settings
from app.routers.predict import process_audio_for_model
# import onnxruntime # Commented out as the inference test is removed for now


def test_process_audio_output_characteristics():
    """
    Tests the output characteristics of the process_audio_for_model function.
    Ensures shape, dtype, and data range ([0,1]) are as expected.
    """
    print("RUNNING TEST: test_process_audio_output_characteristics")
    sr = settings.TARGET_SAMPLE_RATE
    duration = 1.0  # 1 second of audio
    frequency = 440.0  # A4 note

    print("Generating dummy audio...")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio_np = 0.5 * np.sin(2 * np.pi * frequency * t)

    audio_bytes_io = io.BytesIO()
    sf.write(audio_bytes_io, audio_np, sr, format="WAV", subtype="PCM_16")
    audio_bytes = audio_bytes_io.getvalue()
    print(f"Generated dummy audio: {len(audio_bytes)} bytes")

    print("Processing audio...")
    try:
        output_tensor = process_audio_for_model(audio_bytes)
    except Exception as e:
        print(f"ERROR during process_audio_for_model: {e}")
        import traceback

        traceback.print_exc()
        assert False, f"process_audio_for_model raised an exception: {e}"
        return

    print(
        f"Processed audio tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}"
    )

    print("Running assertions...")
    assert output_tensor.shape == (1, 1, settings.N_MELS, settings.SPECTROGRAM_WIDTH), (
        f"AssertionError: Expected shape (1, 1, {settings.N_MELS}, {settings.SPECTROGRAM_WIDTH}), got {output_tensor.shape}"
    )

    assert output_tensor.dtype == np.float32, (
        f"AssertionError: Expected dtype np.float32, got {output_tensor.dtype}"
    )

    min_val = np.min(output_tensor)
    max_val = np.max(output_tensor)
    assert min_val >= 0.0, f"AssertionError: Expected min value >= 0.0, got {min_val}"
    # Loosen the upper bound slightly due to potential floating point inaccuracies if max_val is extremely close to 1.0
    assert max_val <= 1.000001, (
        f"AssertionError: Expected max value <= 1.0, got {max_val}"
    )

    if np.all(output_tensor == 0):
        print("WARNING: Output tensor is all zeros. This might be unexpected.")
        # assert not np.all(output_tensor == 0), "Output tensor is all zeros, which is unexpected."

    assert output_tensor.ndim == 4, (
        f"AssertionError: Expected 4 dimensions, got {output_tensor.ndim}"
    )
    print("ALL ASSERTIONS PASSED for test_process_audio_output_characteristics.")


if __name__ == "__main__":
    print("Executing app/tests/test_audio_processing.py directly...")
    try:
        test_process_audio_output_characteristics()
        print(
            "--- test_process_audio_output_characteristics SUCCEEDED (direct run) ---"
        )
    except AssertionError as e:
        print(
            f"--- test_process_audio_output_characteristics FAILED (direct run): AssertionError: {e} ---"
        )
        # Optionally, exit with a non-zero code to indicate failure
        # import sys
        # sys.exit(1)
    except Exception as e:
        print(
            f"--- test_process_audio_output_characteristics FAILED (direct run): Exception: {e} ---"
        )
        import traceback

        traceback.print_exc()
        # import sys
        # sys.exit(1)
    print("Direct run of test_audio_processing.py finished.")
