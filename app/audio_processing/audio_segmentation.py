# audio_segmentation.py
import os
import sys
import librosa
import numpy as np
import warnings
from pydub import AudioSegment, silence
from app.config import settings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# AudioConfig class removed
def detect_silence_ratio(audio_segment: AudioSegment, silence_thresh: int = -40, min_silence_len: int = 300) -> float:
    silences = silence.detect_silence(
        audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    total_silence = sum((end - start) for start, end in silences)
    return total_silence / len(audio_segment)

def segment_audio(
    audio_path: str,
    target_sr: int = settings.TARGET_SAMPLE_RATE,
    segment_duration: float = settings.CHUNK_DURATION_SECONDS,
    overlap_duration: float = settings.SEGMENT_OVERLAP_SECONDS,
) -> list[np.ndarray]:
    try:
        y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        total_samples = len(y)
        segment_samples = int(segment_duration * target_sr)
        overlap_samples = int(overlap_duration * target_sr)
        step_samples = segment_samples - overlap_samples
        segments = []

        audio = (
            AudioSegment.from_file(audio_path).set_frame_rate(target_sr).set_channels(1)
        )
        audio_len_ms = len(audio)
        target_len_ms = int(segment_duration * 1000)

        silent_segments = 0
        for i in range(0, total_samples, step_samples):
            segment_start = i
            segment_end = min(i + segment_samples, total_samples)
            segment = y[segment_start:segment_end]

            start_ms = int(segment_start * 1000 / target_sr)
            end_ms = min(start_ms + target_len_ms, audio_len_ms)
            segment_audio = audio[start_ms:end_ms]

            if len(segment) < segment_samples:
                segment = np.pad(
                    segment, (0, segment_samples - len(segment)), "constant"
                )
                segment_audio = segment_audio + AudioSegment.silent(
                    duration=target_len_ms - len(segment_audio)
                )

            silence_ratio = detect_silence_ratio(
                segment_audio, silence_thresh=-50, min_silence_len=500
            )
            print(f"Segment {i // step_samples + 1}: silence_ratio={silence_ratio:.2f}")
            if silence_ratio <= 0.7:
                segments.append(segment)
            else:
                silent_segments += 1

        print(
            f"Total segments: {len(segments)}, silent segments skipped: {silent_segments}"
        )
        return segments
    except Exception as e:
        print(f"Error loading or segmenting audio {audio_path}: {e}")
        return []