import os
import shutil
import uuid
import torch
import numpy as np
from typing import List, Tuple, Literal, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form
from starlette.datastructures import State  # For type hinting app.state if needed

from app.audio_processing.audio_segmentation import segment_audio

from app.audio_processing.spectrogram_processing import (
    create_mel_spectrogram,
    preprocess_spectrogram_to_tensor,
)

from app.config import settings

router = APIRouter()

# Correctly define UPLOAD_DIR relative to this file's location (app/routers/predition_pipeline.py)
# It should point to app/uploads/
# __file__ is app/routers/predition_pipeline.py
# os.path.dirname(__file__) is app/routers/
# os.path.join(os.path.dirname(__file__), '..', 'uploads') is app/uploads/
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
print(f"Upload directory initialized at: {UPLOAD_DIR}")


def get_pytorch_models(request: Request) -> Dict[str, torch.nn.Module]:
    if hasattr(request.app.state, "pytorch_models"):
        return request.app.state.pytorch_models
    return {}


def predict_single_segment(
    model: torch.nn.Module, segment_waveform: np.ndarray, device: torch.device
) -> torch.Tensor | None:
    log_mel_spectrogram = create_mel_spectrogram(
        segment_waveform,
        sr=settings.TARGET_SAMPLE_RATE,
        n_fft=settings.N_FFT,
        hop_length=settings.HOP_LENGTH,
        n_mels=settings.N_MELS,
    )
    if log_mel_spectrogram is None:
        print("Spectrogram creation failed for a segment.")
        return None

    input_tensor = preprocess_spectrogram_to_tensor(log_mel_spectrogram)
    if input_tensor is None:
        print("Spectrogram preprocessing failed for a segment.")
        return None

    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output_logits = model(input_tensor)
    return output_logits.cpu()


def aggregate_predictions(
    all_logits: List[torch.Tensor],  # List of tensors, each [1, num_classes]
    aggregation_method: Literal[
        "mean_logits", "mean_probs", "majority_vote"
    ] = "mean_probs",
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    class_names = settings.LABELS

    if not all_logits:
        num_classes = len(class_names)
        return (np.zeros(num_classes), np.zeros(num_classes), "unknown", -1)

    # Stack along a new dimension (dim=0) to get shape [num_segments, 1, num_classes]
    # Then squeeze out the dimension of size 1 to get [num_segments, num_classes]
    stacked_logits_all = torch.stack(all_logits, dim=0).squeeze(1)

    if aggregation_method == "mean_logits":
        mean_logits_tensor = torch.mean(
            stacked_logits_all, dim=0
        )  # Shape: [num_classes]
        final_probabilities_tensor = torch.softmax(mean_logits_tensor, dim=0)
        final_probabilities = final_probabilities_tensor.numpy()
        final_logits = mean_logits_tensor.numpy()
    elif aggregation_method == "mean_probs":
        all_probabilities_tensor = torch.softmax(
            stacked_logits_all, dim=1
        )  # Softmax along class dimension
        mean_probabilities_tensor = torch.mean(
            all_probabilities_tensor, dim=0
        )  # Shape: [num_classes]
        final_probabilities = mean_probabilities_tensor.numpy()
        final_logits = np.log(final_probabilities + 1e-9)
    elif aggregation_method == "majority_vote":
        predicted_indices_per_segment = torch.argmax(
            torch.softmax(stacked_logits_all, dim=1), dim=1
        ).numpy()

        if predicted_indices_per_segment.size == 0:
            num_classes = len(class_names)
            return (np.zeros(num_classes), np.zeros(num_classes), "unknown", -1)

        counts = np.bincount(predicted_indices_per_segment, minlength=len(class_names))
        predicted_class_index = np.argmax(counts)

        total_segments = len(predicted_indices_per_segment)
        final_probabilities = (
            counts / total_segments
            if total_segments > 0
            else np.zeros(len(class_names))
        )
        final_logits = np.log(final_probabilities + 1e-9)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    if aggregation_method != "majority_vote":
        predicted_class_index = np.argmax(final_probabilities)

    predicted_class_index = int(predicted_class_index)
    predicted_class_name = class_names.get(predicted_class_index, "Unknown Label")

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
    print(f"Using device: {device} for prediction for audio: {audio_path}")
    model.to(device)

    print(
        f"Segmenting audio file: {audio_path} into {settings.CHUNK_DURATION_SECONDS}s chunks..."
    )
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
        logits = predict_single_segment(model, segment, device)
        if (
            logits is not None
        ):  # logits from predict_single_segment are [1, num_classes]
            all_logits.append(logits)
            valid_segments_count += 1
        else:
            print(f"Skipping segment {i} of {audio_path} due to processing error.")

    num_segments_predicted = valid_segments_count

    if not all_logits:
        print(
            f"No valid segments processed for {audio_path}. Cannot make a prediction."
        )
        num_classes = len(settings.LABELS)
        return (
            np.zeros(num_classes),
            np.zeros(num_classes),
            "unknown",
            -1,
            num_segments_predicted,
        )

    print(
        f"Aggregating {len(all_logits)} segment predictions for {audio_path} using '{aggregation_method}' method..."
    )
    final_probabilities, final_logits, predicted_class_name, predicted_class_index = (
        aggregate_predictions(all_logits, aggregation_method)
    )

    return (
        final_probabilities,
        final_logits,
        predicted_class_name,
        predicted_class_index,
        num_segments_predicted,
    )


@router.post("/predict/")
async def predict_endpoint(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form("vit_small"),
    aggregation_method: str = Form("mean_probs"),
):
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

    print(
        f"Attempting to save uploaded file '{original_filename}' to '{temp_audio_path}'"
    )

    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(
            f"Uploaded file '{original_filename}' saved successfully to '{temp_audio_path}'"
        )

        probabilities, logits, class_name, class_idx, segments_processed = (
            predict_audio_file(
                model=selected_model,
                audio_path=temp_audio_path,
                aggregation_method=aggregation_method,
            )
        )

        probabilities_list = probabilities.tolist()

        if class_idx == -1 and segments_processed == 0:
            display_class_name = "No valid audio segments processed"
        elif class_idx == -1:
            display_class_name = "Prediction inconclusive"
        else:
            display_class_name = settings.LABELS.get(
                class_idx, "Unknown Label (from settings)"
            )

        return {
            "filename": original_filename,
            "model_used": model_name,
            "predicted_class_index": class_idx,
            "predicted_class_name": display_class_name,
            "probabilities": probabilities_list,
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
        if (
            hasattr(file, "file")
            and file.file
            and hasattr(file.file, "close")
            and not file.file.closed
        ):
            file.file.close()
            print(f"Closed file stream for {original_filename}")

        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"Cleaned up temporary file: {temp_audio_path}")
            except OSError as e_os:
                print(f"Error deleting temporary file {temp_audio_path}: {e_os}")
