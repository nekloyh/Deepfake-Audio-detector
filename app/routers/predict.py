# app/routers/predict.py

from fastapi import APIRouter, File, UploadFile, HTTPException, status, Request
from typing import Dict, Any
import torch
import io
import numpy as np
import librosa
import soundfile as sf
import numpy as np  # Already imported, but ensure it's available for type hints if strict

# Import settings from app.config
from ..config import settings
from ..audio_processing.utils import split_into_chunks  # Added import

router = APIRouter()


def process_audio_for_model(audio_chunk: np.ndarray, original_sr: int) -> np.ndarray:
    """
    Processes an audio chunk (NumPy array) into a normalized Mel spectrogram.
    Uses settings from app.config for audio processing parameters.
    The audio chunk is resampled, padded/trimmed to CHUNK_DURATION_SECONDS (if not already),
    converted to a Mel spectrogram, converted to dB, normalized to [0,1],
    and padded/trimmed to (N_MELS, SPECTROGRAM_WIDTH).
    The result is prepared with batch and channel dimensions for ONNX inference.
    """
    print(
        f"Starting audio processing for chunk... Original SR: {original_sr}, Chunk Samples: {len(audio_chunk)}"
    )
    try:
        audio = audio_chunk
        sr = original_sr

        # Step 1: Resample audio to the target sample rate if it differs.
        # Models are typically trained on audio with a specific sample rate (settings.TARGET_SAMPLE_RATE).
        if sr != settings.TARGET_SAMPLE_RATE:
            print(
                f"Resampling audio chunk from {sr} Hz to {settings.TARGET_SAMPLE_RATE} Hz..."
            )
            audio = librosa.resample(
                y=audio, orig_sr=sr, target_sr=settings.TARGET_SAMPLE_RATE
            )
            sr = settings.TARGET_SAMPLE_RATE  # Update sample rate to target
            print(f"Audio chunk resampled: New Samples={len(audio)}")

        # Step 2: Pad or trim audio to a consistent fixed duration (settings.CHUNK_DURATION_SECONDS).
        # This ensures that the resulting spectrogram has a consistent width (time dimension).
        # This step is crucial as split_into_chunks provides chunks of CHUNK_DURATION_SECONDS based on original_sr.
        # After resampling, the number of samples for that duration might change if original_sr != TARGET_SAMPLE_RATE.
        target_length_samples = int(
            settings.TARGET_SAMPLE_RATE * settings.CHUNK_DURATION_SECONDS
        )

        if len(audio) < target_length_samples:
            # Pad with zeros (silence) at the end if the audio is shorter.
            print(
                f"Padding audio chunk from {len(audio)} to {target_length_samples} samples."
            )
            audio = np.pad(
                audio,
                (0, target_length_samples - len(audio)),
                "constant",
                constant_values=0.0,
            )
        elif len(audio) > target_length_samples:
            # Trim from the end if the audio is longer.
            print(
                f"Trimming audio chunk from {len(audio)} to {target_length_samples} samples."
            )
            audio = audio[:target_length_samples]
        print(f"Audio chunk length after padding/trimming: {len(audio)} samples.")

        # Step 3: Compute Mel spectrogram from the processed audio waveform.
        # Parameters like n_fft, hop_length, and n_mels (number of Mel bands) are crucial
        # for consistency with the model's training data.
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=settings.N_FFT,
            hop_length=settings.HOP_LENGTH,
            n_mels=settings.N_MELS,
        )
        print(
            f"Mel spectrogram computed. Shape: {mel_spectrogram.shape}"
        )  # (n_mels, num_time_frames)

        # Step 5: Convert power spectrogram to decibels (dB).
        # `ref=np.max` sets the loudest part of the signal to approximately 0dB.
        # `top_db` defines the dynamic range; values quieter than `(ref - top_db)` are clipped.
        mel_spectrogram_db = librosa.power_to_db(
            mel_spectrogram,
            ref=np.max,
            amin=1e-10,  # Small constant to avoid log(0) errors.
            top_db=settings.TOP_DB,
        )
        print(
            f"Mel spectrogram converted to dB. Shape: {mel_spectrogram_db.shape}, Min: {mel_spectrogram_db.min():.2f} dB, Max: {mel_spectrogram_db.max():.2f} dB"
        )

        # Step 6: Normalize the dB spectrogram to the [0, 1] range.
        # This specific normalization maps the dB range [settings.MIN_DB_LEVEL, 0dB] to [0,1].
        # - 0dB is assumed as the effective maximum after `librosa.power_to_db` with `ref=np.max`.
        # - Values below MIN_DB_LEVEL are clipped to MIN_DB_LEVEL before scaling.
        # - Values originally above 0dB would also be scaled relative to this range and then clipped to 1.0.
        spec_db_clipped = np.maximum(mel_spectrogram_db, settings.MIN_DB_LEVEL)

        # Perform scaling: (value - min_db) / (effective_max_db - min_db)
        # Here, effective_max_db is 0. So, divisor is (0 - settings.MIN_DB_LEVEL) = -settings.MIN_DB_LEVEL.
        normalized_spectrogram = (spec_db_clipped - settings.MIN_DB_LEVEL) / (
            -settings.MIN_DB_LEVEL
        )

        # Clip the values to ensure they are strictly within the [0, 1] range.
        # This is crucial if the ONNX model strictly expects inputs in this precise range.
        normalized_spectrogram = np.clip(normalized_spectrogram, 0, 1)
        print(
            f"Normalized spectrogram to [0,1] range. Shape: {normalized_spectrogram.shape}, Min: {normalized_spectrogram.min():.2f}, Max: {normalized_spectrogram.max():.2f}"
        )
        # NOTE: This [0,1] normalization is a common practice. However, due to environmental limitations
        # during development (inability to directly inspect ONNX model's training preprocessing),
        # it's an assumption that the model expects this specific range. If model performance is suboptimal,
        # this normalization step (especially the range) should be revisited and verified against the
        # model's actual training data preprocessing.

        # Step 7: Pad or truncate the spectrogram width (time axis) to SPECTROGRAM_WIDTH.
        # The height (N_MELS) is determined by librosa.feature.melspectrogram (using settings.N_MELS)
        # and should match the model's expected height. No explicit height resizing is performed here.
        current_height, current_width = normalized_spectrogram.shape
        target_height = settings.N_MELS  # Expected height from config
        target_width = settings.SPECTROGRAM_WIDTH  # Expected width from config

        if current_height != target_height:
            # This case should ideally not occur if n_mels parameter to melspectrogram is correct.
            # It indicates a potential configuration mismatch or unexpected librosa output.
            # Resizing height here might distort features and is generally avoided post-spectrogram generation.
            print(
                f"CRITICAL: Spectrogram height {current_height} does not match target N_MELS {target_height}. Check N_MELS setting ({settings.N_MELS}) and audio processing."
            )
            # For robustness, one might fall back to resizing, but it's better to ensure correct generation:
            # e.g., using Pillow/cv2: img = Image.fromarray((normalized_spectrogram * 255).astype(np.uint8)).resize((target_width, target_height)) ... then back to np.array / 255.0

        # Pad width if current_width is less than target_width
        if current_width < target_width:
            pad_amount = target_width - current_width
            # Pad with 0.0, which corresponds to MIN_DB_LEVEL in the normalized [0,1] space (silence).
            processed_spectrogram = np.pad(
                normalized_spectrogram,
                ((0, 0), (0, pad_amount)),
                "constant",
                constant_values=0.0,
            )
            print(
                f"Padded spectrogram width from {current_width} to {target_width}. New shape: {processed_spectrogram.shape}"
            )
        # Truncate width if current_width is greater than target_width
        elif current_width > target_width:
            processed_spectrogram = normalized_spectrogram[
                :, :target_width
            ]  # Truncate from the end
            print(
                f"Truncated spectrogram width from {current_width} to {target_width}. New shape: {processed_spectrogram.shape}"
            )
        else:
            processed_spectrogram = (
                normalized_spectrogram  # No change needed if width matches
            )
            print(
                "Spectrogram width matches target; no width padding/truncation needed."
            )

        # Final safeguard: check actual dimensions against target.
        # This helps catch issues if height was mismatched and not handled, or padding logic had an error.
        if processed_spectrogram.shape != (target_height, target_width):
            raise ValueError(
                f"Spectrogram shape {processed_spectrogram.shape} after processing does not match target ({target_height}, {target_width})."
            )

        # Step 8: Add batch and channel dimensions for ONNX model input.
        # Common shape for CNNs: (batch_size, channels, height, width).
        # For a single grayscale spectrogram, channels = 1, batch_size = 1.
        input_tensor = processed_spectrogram[np.newaxis, np.newaxis, :, :]
        print(f"Final input tensor shape for ONNX model: {input_tensor.shape}")

        # Ensure the final tensor is float32, as typically required by ONNX models.
        return input_tensor.astype(np.float32)

    except sf.LibsndfileError as lse:
        print(f"Error during audio file reading: {lse}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading audio file: {lse}. Ensure it's a valid audio format like WAV, FLAC, etc.",
        )
    except ValueError as ve:  # Catch ValueError specifically if raised by shape checks
        print(f"ValueError during audio processing: {ve}")
        import traceback

        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing error (ValueError): {ve}",
        )
    except Exception as e:
        print(f"An unexpected error occurred during audio processing: {e}")
        import traceback

        print(traceback.format_exc())  # Print full traceback for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during audio processing: {e}",
        )


# The rest of the router code remains the same
@router.post("/predict_audio")
async def predict_audio(
    request: Request,  # Inject the Request object
    audio_file: UploadFile = File(...),
    model_name: str = "cnn_small",  # Default model is cnn_small
) -> list:  # Changed return type to list for chunked predictions
    """
    Receives an audio file, splits it into chunks, processes each chunk,
    and makes a prediction using the specified PyTorch model.
    The model can be selected via the 'model_name' query parameter.
    Defaults to 'cnn_small' if no model_name is provided.
    Returns a list of prediction results, one for each chunk.
    """
    pytorch_models_dict: Dict[str, torch.nn.Module] = request.app.state.pytorch_models

    if not pytorch_models_dict:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyTorch models not loaded or available. Server configuration issue.",
        )

    if model_name not in pytorch_models_dict:
        valid_models = [
            name for name, m in pytorch_models_dict.items() if m is not None
        ]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found. Available models: {valid_models}",
        )

    model = pytorch_models_dict.get(model_name)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' is configured but was not loaded successfully. Check server logs.",
        )

    audio_bytes = await audio_file.read()
    predictions_list = []

    try:
        # Convert audio_bytes to waveform and sample rate
        original_waveform, original_sr = sf.read(io.BytesIO(audio_bytes))

        # Ensure waveform is mono
        if original_waveform.ndim > 1 and original_waveform.shape[1] > 1:
            print(
                f"Original waveform is stereo (shape: {original_waveform.shape}), converting to mono."
            )
            original_waveform = np.mean(original_waveform, axis=1)

        # Split audio into chunks
        # split_into_chunks expects sample_rate and chunk_duration_sec
        # It returns chunks of original_waveform, each approximately chunk_duration_sec long,
        # potentially padded. These chunks are at original_sr.
        chunks = split_into_chunks(
            original_waveform, original_sr, settings.CHUNK_DURATION_SECONDS
        )
        print(f"Audio split into {len(chunks)} chunks.")

        for i, chunk_waveform in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)}")
            # process_audio_for_model expects the chunk waveform and its original sample rate.
            # It will then resample to TARGET_SAMPLE_RATE and process.
            input_data = process_audio_for_model(
                chunk_waveform, original_sr
            )  # Pass original_sr

            input_tensor = torch.from_numpy(input_data)

            try:
                device = next(model.parameters()).device
            except StopIteration:
                print(
                    "Warning: Could not determine model device from parameters. Assuming CPU."
                )
                device = torch.device("cpu")

            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                outputs = model(input_tensor)

            prediction = outputs.cpu().detach().numpy()

            if prediction.shape[-1] > 1:
                predicted_index = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_index])
                result_label = settings.LABELS.get(predicted_index, "unknown")
            else:
                confidence = float(prediction[0][0])
                if confidence > 0.5:
                    result_label = settings.LABELS.get(
                        settings.REAL_LABEL_INDEX, "real"
                    )
                else:
                    result_label = settings.LABELS.get(
                        settings.FAKE_LABEL_INDEX, "deepfake"
                    )

            predictions_list.append(
                {
                    "filename": audio_file.filename,
                    "chunk_index": i,
                    "num_chunks": len(chunks),
                    "prediction": result_label,
                    "confidence": confidence,
                    "model_used": model_name,
                    "raw_model_output": prediction.tolist(),
                }
            )

        return predictions_list

    except sf.LibsndfileError as lse:
        # This exception should be caught by the process_audio_for_model if it happens during sf.read
        # However, sf.read is now called before process_audio_for_model. So, catch here.
        print(f"Error during audio file reading (soundfile): {lse}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading audio file with soundfile: {lse}. Ensure it's a valid audio format.",
        )
    except ValueError as ve:
        # This can be raised by process_audio_for_model or other numpy/torch operations
        print(f"ValueError during prediction processing: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        import traceback

        print(f"Prediction failed with an unexpected error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}. Check server logs for details.",
        )
