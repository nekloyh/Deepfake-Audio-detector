import os
import shutil
import uuid
import torch
import numpy as np
from typing import List, Tuple, Literal, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form # For type hinting app.state if needed

from app.audio_processing.audio_segmentation import segment_audio
from app.audio_processing.spectrogram_processing import (
    create_mel_spectrogram,
    preprocess_spectrogram_to_tensor,
)
from app.config import settings # Required for normalization

router = APIRouter()

# Define UPLOAD_DIR relative to this file's location (app/routers/prediction_pipeline.py)
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
print(f"Upload directory initialized at: {UPLOAD_DIR}")


def get_pytorch_models(request: Request) -> Dict[str, torch.nn.Module]:
    """
    Retrieves PyTorch models from app state.
    """
    if hasattr(request.app.state, "pytorch_models"):
        return request.app.state.pytorch_models
    return {}


def predict_single_segment(
    model: torch.nn.Module, segment_waveform: np.ndarray, device: torch.device
) -> torch.Tensor | None:
    print(f"Processing segment with waveform shape: {segment_waveform.shape}")
    log_mel_spectrogram = create_mel_spectrogram(
        segment_waveform,
        sr=settings.TARGET_SAMPLE_RATE,
        n_mels=settings.N_MELS,
        n_fft=settings.N_FFT,
        hop_length=settings.HOP_LENGTH,
    )
    if log_mel_spectrogram is None:
        print("Spectrogram creation failed for a segment.")
        return None
    print(f"Log Mel spectrogram shape: {log_mel_spectrogram.shape}")

    mean = (
        settings.PIXEL_MEAN
        if isinstance(settings.PIXEL_MEAN, float)
        else settings.PIXEL_MEAN
    )
    std = (
        settings.PIXEL_STD
        if isinstance(settings.PIXEL_STD, float)
        else settings.PIXEL_STD
    )
    input_tensor = preprocess_spectrogram_to_tensor(
        log_mel_spectrogram,
        image_size=settings.IMAGE_SIZE,
        mean=mean,
        std=std,
    )
    if input_tensor is None:
        print("Spectrogram preprocessing failed for a segment.")
        return None
    print(f"Input tensor shape: {input_tensor.shape}")

    input_tensor = input_tensor.unsqueeze(0).to(device)
    print(f"Input tensor shape after unsqueeze: {input_tensor.shape}")

    try:
        with torch.no_grad():
            output_logits = model(input_tensor)
        print(f"Output logits shape: {output_logits.shape}, values: {output_logits}")
        return output_logits.cpu()
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

def aggregate_predictions(
    all_logits: List[torch.Tensor],
    aggregation_method: Literal[
        "mean_logits", "mean_probs", "majority_vote"
    ] = "mean_probs",
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """
    Aggregates predictions from multiple segments.
    """
    class_names = settings.LABELS
    if not all_logits:
        num_classes = len(class_names)
        return (np.zeros(num_classes), np.zeros(num_classes), "no_valid_logits", -1)

    stacked_logits_all = torch.stack(all_logits, dim=0).squeeze(1)
    print(
        f"Aggregation method: {aggregation_method}, stacked logits shape: {stacked_logits_all.shape}"
    )

    if aggregation_method == "mean_logits":
        mean_logits_tensor = torch.mean(stacked_logits_all, dim=0)
        final_probabilities_tensor = torch.softmax(mean_logits_tensor, dim=0)
        final_probabilities = final_probabilities_tensor.numpy()
        final_logits = mean_logits_tensor.numpy()
    elif aggregation_method == "mean_probs":
        all_probabilities_tensor = torch.softmax(stacked_logits_all, dim=1)
        mean_probabilities_tensor = torch.mean(all_probabilities_tensor, dim=0)
        final_probabilities = mean_probabilities_tensor.numpy()
        final_logits = np.log(final_probabilities + 1e-9)
    elif aggregation_method == "majority_vote":
        predicted_indices_per_segment = torch.argmax(
            torch.softmax(stacked_logits_all, dim=1), dim=1
        ).numpy()
        if predicted_indices_per_segment.size == 0:
            num_classes = len(class_names)
            return (np.zeros(num_classes), np.zeros(num_classes), "no_valid_logits", -1)
        counts = np.bincount(predicted_indices_per_segment, minlength=len(class_names))
        total_segments = counts.sum()
        final_probabilities = counts / total_segments
        final_logits = np.log(final_probabilities + 1e-9)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    # Validate probabilities
    if (
        np.any(final_probabilities < 0)
        or np.any(final_probabilities > 1)
        or not np.isclose(np.sum(final_probabilities), 1.0, atol=1e-5)
    ):
        print(f"Invalid probabilities detected: {final_probabilities}")
        num_classes = len(class_names)
        return (
            np.zeros(num_classes),
            np.zeros(num_classes),
            "invalid_probabilities",
            -1,
        )

    predicted_class_index = int(np.argmax(final_probabilities))
    predicted_class_name = class_names.get(predicted_class_index, "unknown")

    return (
        final_probabilities,
        final_logits,
        predicted_class_name,
        predicted_class_index,
    )
    

def predict_audio_file(
    model: torch.nn.Module,
    audio_path: str,
    aggregation_method: str = "mean_probs",
) -> Tuple[np.ndarray, np.ndarray, str, int, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Đảm bảo mô hình ở chế độ đánh giá
    print(f"Using device: {device} for prediction for audio: {audio_path}")

    audio_segments = segment_audio(
        audio_path,
        target_sr=settings.TARGET_SAMPLE_RATE,
        segment_duration=settings.CHUNK_DURATION_SECONDS,
        overlap_duration=getattr(settings, "SEGMENT_OVERLAP_SECONDS", 0.0),
    )
    print(f"Created {len(audio_segments)} segments from {audio_path}.")

    all_logits: List[torch.Tensor] = []
    valid_segments_count = 0
    for i, segment in enumerate(audio_segments):
        print(f"Processing segment {i + 1}/{len(audio_segments)}")
        logits = predict_single_segment(model, segment, device)
        if logits is not None:
            all_logits.append(logits)
            valid_segments_count += 1
            print(f"Segment {i + 1} logits: {logits.numpy()}")
        else:
            print(f"Skipping segment {i + 1} due to processing error.")

    if not all_logits:
        num_classes = len(settings.LABELS)
        return (
            np.zeros(num_classes),
            np.zeros(num_classes),
            "no_valid_segments",
            -1,
            valid_segments_count,
        )

    final_probabilities, final_logits, predicted_class_name, predicted_class_index = (
        aggregate_predictions(all_logits, aggregation_method)
    )
    print(f"Final probabilities: {final_probabilities}")
    return (
        final_probabilities,
        final_logits,
        predicted_class_name,
        predicted_class_index,
        valid_segments_count,
    )

@router.post("/predict/")
async def predict_endpoint(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form("vit_small"),
    aggregation_method: str = Form("mean_probs"),
):
    """
    FastAPI endpoint for audio file prediction.
    """
    pytorch_models = get_pytorch_models(request)

    if not pytorch_models:
        print("Error: PyTorch models not loaded in app.state.")
        raise HTTPException(status_code=503, detail="Models not loaded or unavailable.")

    selected_model = pytorch_models.get(model_name)
    if selected_model is None:
        if model_name in pytorch_models:
            print(f"Error: Model '{model_name}' was configured but failed to load.")
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' failed to load during server startup.",
            )
        else:
            print(
                f"Error: Model '{model_name}' not found. Available: {list(pytorch_models.keys())}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not found. Available models: {list(pytorch_models.keys())}",
            )

    file_id = str(uuid.uuid4())
    original_filename = file.filename if file.filename else "unknown_file"
    file_extension = os.path.splitext(original_filename)[1]
    temp_audio_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

    print(f"Saving uploaded file '{original_filename}' to '{temp_audio_path}'")
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Uploaded file '{original_filename}' saved successfully")

        probabilities, logits, class_name, class_idx, segments_processed = (
            predict_audio_file(
                model=selected_model,
                audio_path=temp_audio_path,
                aggregation_method=aggregation_method,
            )
        )

        probabilities_list = probabilities.tolist()
        # Validate probabilities
        if not probabilities_list or any(p < 0 or p > 1 for p in probabilities_list):
            print(f"Invalid probabilities: {probabilities_list}")
            probabilities_list = [0.0] * len(settings.LABELS)
            # class_name = "invalid_probabilities"
            class_idx = -1
            confidence = 0.0
        else:
            confidence = max(probabilities_list, default=0.0) * 100  # Percentage

        if segments_processed == 0:
            print(f"No valid segments processed for {original_filename}")
            return {
                "filename": original_filename,
                "model_used": model_name,
                "predicted_class_index": -1,
                "predicted_class_name": "no_valid_segments",
                "probabilities": probabilities_list,
                "confidence": 0.0,
                "aggregation_method": aggregation_method,
                "segments_processed": segments_processed,
                "error": "No valid audio segments could be processed. Check audio file format or content.",
            }

        if class_idx == -1:
            print(f"Prediction inconclusive for {original_filename}")
            predicted_class_name = "inconclusive"
        else:
            predicted_class_name = settings.LABELS.get(class_idx, "unknown")

        print(
            f"Prediction result: class={predicted_class_name}, probabilities={probabilities_list}, confidence={confidence}%"
        )
        return {
            "filename": original_filename,
            "model_used": model_name,
            "predicted_class_index": class_idx,
            "predicted_class_name": predicted_class_name,
            "probabilities": probabilities_list,
            "confidence": confidence,
            "aggregation_method": aggregation_method,
            "segments_processed": segments_processed,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unhandled error during prediction for {original_filename}: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing audio file: {str(e)}"
        )
    finally:
        if hasattr(file, "file") and file.file and not file.file.closed:
            file.file.close()
            print(f"Closed file stream for {original_filename}")
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"Cleaned up temporary file: {temp_audio_path}")
            except OSError as e_os:
                print(f"Error deleting temporary file {temp_audio_path}: {e_os}")