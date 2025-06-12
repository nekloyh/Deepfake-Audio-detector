import os
import shutil
import uuid
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Request,
    Form,
)

from app.audio_processing.audio_segmentation import segment_audio
from app.audio_processing.spectrogram_processing import (
    create_mel_spectrogram,
    preprocess_spectrogram_to_tensor,
)
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def calculate_adaptive_threshold(
    all_probabilities_tensor: torch.Tensor,
    base_threshold: float = 0.6,
    min_threshold: float = 0.3,
    max_threshold: float = 0.8,
) -> float:
    """
    Calculate adaptive confidence threshold based on probability distribution.
    Helps reduce bias by adjusting threshold based on actual model confidence.
    """
    if all_probabilities_tensor.numel() == 0:
        return base_threshold

    max_probs = torch.max(all_probabilities_tensor, dim=1)[0]
    mean_confidence = torch.mean(max_probs).item()
    std_confidence = torch.std(max_probs).item()

    # Adaptive threshold: lower if overall confidence is low, higher if very confident
    adaptive_threshold = mean_confidence - (0.5 * std_confidence)

    # Clamp to reasonable bounds
    adaptive_threshold = max(min_threshold, min(max_threshold, adaptive_threshold))

    logger.info(
        f"Adaptive threshold calculated: {adaptive_threshold:.3f} "
        f"(mean_conf: {mean_confidence:.3f}, std: {std_confidence:.3f})"
    )

    return adaptive_threshold


def apply_class_balancing(
    probabilities: np.ndarray,
    method: str = "equal_weight",
    class_weights: Optional[np.ndarray] = None,
    real_bias_factor: float = settings.REAL_BIAS_FACTOR,
) -> np.ndarray:
    """
    Apply class balancing with optional bias towards REAL class.

    Args:
        probabilities: Raw probability predictions
        method: Balancing method ("equal_weight", "inverse_freq", "custom", "real_bias")
        class_weights: Custom weights for each class (if method="custom")
        real_bias_factor: Multiplier for REAL class probability (>1.0 favors REAL, <1.0 favors FAKE)
    """
    if method == "equal_weight":
        # Equal weighting - no bias towards any class
        return probabilities

    elif method == "real_bias":
        # Apply bias towards REAL class
        # Assuming REAL is index 1 and FAKE is index 0 (adjust based on your LABELS config)
        real_class_idx = None
        fake_class_idx = None

        # Find REAL and FAKE indices from settings.LABELS
        for idx, label in settings.LABELS.items():
            if label.lower() in ["real", "authentic", "genuine"]:
                real_class_idx = idx
            elif label.lower() in ["fake", "deepfake", "synthetic"]:
                fake_class_idx = idx

        if real_class_idx is not None:
            weights = np.ones(len(probabilities))
            weights[real_class_idx] = real_bias_factor

            logger.info(
                f"Applying REAL bias: factor={real_bias_factor}, "
                f"REAL_idx={real_class_idx}, FAKE_idx={fake_class_idx}"
            )
        else:
            logger.warning("Could not identify REAL class index, using equal weights")
            weights = np.ones(len(probabilities))

    elif method == "inverse_freq":
        # Weight inversely proportional to class frequency (helps with imbalanced data)
        # This would need training data statistics - using equal for now
        weights = np.ones(len(probabilities))

    elif method == "custom" and class_weights is not None:
        weights = class_weights

    else:
        # Default: no balancing
        return probabilities

    # Apply weights
    if "weights" in locals():
        balanced_probs = probabilities * weights
        # Re-normalize to ensure probabilities sum to 1
        balanced_probs = balanced_probs / (np.sum(balanced_probs) + 1e-9)

        logger.info(
            f"Applied class balancing: {method}, "
            f"original_max: {np.max(probabilities):.3f}, "
            f"balanced_max: {np.max(balanced_probs):.3f}, "
            f"original_probs: {probabilities}, balanced_probs: {balanced_probs}"
        )

        return balanced_probs

    return probabilities


def remove_outlier_predictions(
    all_logits: List[torch.Tensor], method: str = "iqr", threshold: float = 2.0
) -> List[torch.Tensor]:
    """
    Remove outlier predictions to improve robustness and reduce bias from anomalous segments.
    """
    if len(all_logits) < 3:  # Need at least 3 predictions for outlier detection
        return all_logits

    # Convert to tensor for easier computation
    stacked_logits = torch.stack([logit.squeeze() for logit in all_logits])

    if method == "iqr":
        # Use Interquartile Range method
        q1 = torch.quantile(stacked_logits, 0.25, dim=0)
        q3 = torch.quantile(stacked_logits, 0.75, dim=0)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Check which predictions are within bounds
        valid_mask = torch.all(
            (stacked_logits >= lower_bound) & (stacked_logits <= upper_bound), dim=1
        )

    elif method == "zscore":
        # Use Z-score method
        mean_logits = torch.mean(stacked_logits, dim=0)
        std_logits = torch.std(stacked_logits, dim=0)

        z_scores = torch.abs((stacked_logits - mean_logits) / (std_logits + 1e-9))
        valid_mask = torch.all(z_scores <= threshold, dim=1)

    else:
        # No outlier removal
        return all_logits

    # Filter outliers
    filtered_logits = [all_logits[i] for i in range(len(all_logits)) if valid_mask[i]]

    removed_count = len(all_logits) - len(filtered_logits)
    if removed_count > 0:
        logger.info(
            f"Removed {removed_count}/{len(all_logits)} outlier predictions using {method} method"
        )

    return (
        filtered_logits if filtered_logits else all_logits
    )  # Fallback to original if all removed


def predict_single_segment(
    model: torch.nn.Module, segment_waveform: np.ndarray, device: torch.device
) -> Optional[torch.Tensor]:
    """
    Predict single audio segment with improved error handling.
    """
    try:
        print(f"Processing segment with waveform shape: {segment_waveform.shape}")

        log_mel_spectrogram = create_mel_spectrogram(
            segment_waveform,
            sr=settings.TARGET_SAMPLE_RATE,
            n_mels=settings.N_MELS,
            n_fft=settings.N_FFT,
            hop_length=settings.HOP_LENGTH,
            target_lufs=settings.LOUDNESS_LUFS,
        )

        if log_mel_spectrogram is None:
            logger.warning("Spectrogram creation failed for a segment.")
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
            log_mel_spectrogram, image_size=settings.IMAGE_SIZE, mean=mean, std=std
        )

        if input_tensor is None:
            logger.warning("Spectrogram preprocessing failed for a segment.")
            return None

        print(f"Input tensor shape: {input_tensor.shape}")

        input_tensor = input_tensor.unsqueeze(0).to(device)
        print(f"Input tensor shape after unsqueeze: {input_tensor.shape}")

        with torch.no_grad():
            output_logits = model(input_tensor)

        # Validate output
        if torch.isnan(output_logits).any() or torch.isinf(output_logits).any():
            logger.warning("Invalid logits detected (NaN or Inf), skipping segment")
            return None

        print(f"Output logits shape: {output_logits.shape}, values: {output_logits}")
        return output_logits.cpu()

    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        return None


def aggregate_predictions(
    all_logits: List[torch.Tensor],
    aggregation_method: str = settings.AGGREGATION_METHOD,
    base_confidence_threshold: float = 0.6,
    enable_outlier_removal: bool = True,
    enable_class_balancing: bool = True,
    real_bias_factor: float = settings.REAL_BIAS_FACTOR,
    bias_method: str = settings.BIAS_METHOD,
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """
    Enhanced aggregation with bias reduction techniques.
    """
    class_names = settings.LABELS

    if not all_logits:
        num_classes = len(class_names)
        logger.warning("No valid logits provided for aggregation.")
        return (np.zeros(num_classes), np.zeros(num_classes), "no_valid_logits", -1)

    # Remove outliers to improve robustness
    if enable_outlier_removal and len(all_logits) >= 3:
        all_logits = remove_outlier_predictions(all_logits, method="iqr")

    if not all_logits:  # Check again after outlier removal
        num_classes = len(class_names)
        logger.warning("All predictions were filtered as outliers.")
        return (np.zeros(num_classes), np.zeros(num_classes), "all_outliers", -1)

    stacked_logits_all = torch.stack(all_logits, dim=0).squeeze(1)
    logger.info(
        f"Aggregation method: {aggregation_method}, stacked logits shape: {stacked_logits_all.shape}"
    )

    all_probabilities_tensor = torch.softmax(stacked_logits_all, dim=1)

    # Log segment probabilities for debugging
    for i, probs in enumerate(all_probabilities_tensor):
        logger.info(f"Segment {i + 1} probabilities: {probs.numpy()}")

    # Calculate adaptive threshold for methods that use confidence filtering
    if aggregation_method in ["majority_vote", "confidence_weighted_probs"]:
        confidence_threshold = calculate_adaptive_threshold(
            all_probabilities_tensor, base_confidence_threshold
        )
    else:
        confidence_threshold = base_confidence_threshold

    # Apply different aggregation methods
    if aggregation_method == "mean_logits":
        mean_logits_tensor = torch.mean(stacked_logits_all, dim=0)
        final_probabilities_tensor = torch.softmax(mean_logits_tensor, dim=0)
        final_probabilities = final_probabilities_tensor.numpy()
        final_logits = mean_logits_tensor.numpy()

    elif aggregation_method == "mean_probs":
        mean_probabilities_tensor = torch.mean(all_probabilities_tensor, dim=0)
        final_probabilities = mean_probabilities_tensor.numpy()
        final_logits = np.log(final_probabilities + 1e-9)

    elif aggregation_method == "median_probs":
        # Add median aggregation for more robustness
        median_probabilities_tensor = torch.median(all_probabilities_tensor, dim=0)[0]
        final_probabilities = median_probabilities_tensor.numpy()
        final_logits = np.log(final_probabilities + 1e-9)

    elif aggregation_method == "majority_vote":
        max_probs, predicted_indices = torch.max(all_probabilities_tensor, dim=1)
        valid_mask = max_probs > confidence_threshold

        if not torch.any(valid_mask):
            logger.warning(
                f"No segments with confidence above adaptive threshold {confidence_threshold:.3f}."
            )
            # Fallback: use segments with top 50% confidence
            top_k = max(1, len(max_probs) // 2)
            _, top_indices = torch.topk(max_probs, top_k)
            valid_mask = torch.zeros_like(max_probs, dtype=torch.bool)
            valid_mask[top_indices] = True
            logger.info(f"Fallback: using top {top_k} confident segments")

        valid_indices = predicted_indices[valid_mask].numpy()
        counts = np.bincount(valid_indices, minlength=len(class_names))
        total_segments = counts.sum()
        final_probabilities = counts / max(total_segments, 1)  # Avoid division by zero
        final_logits = np.log(final_probabilities + 1e-9)

    elif aggregation_method == "confidence_weighted_probs":
        max_probs, _ = torch.max(all_probabilities_tensor, dim=1)
        valid_mask = max_probs > confidence_threshold

        if not torch.any(valid_mask):
            logger.warning(
                f"No segments with confidence above adaptive threshold {confidence_threshold:.3f}."
            )
            # Fallback: use all segments with equal weights
            valid_mask = torch.ones_like(max_probs, dtype=torch.bool)
            logger.info("Fallback: using all segments with equal weights")

        valid_probs = all_probabilities_tensor[valid_mask]
        valid_confidences = max_probs[valid_mask]

        # Use confidence as weights
        weights = valid_confidences / torch.sum(valid_confidences)
        final_probabilities_tensor = torch.sum(
            valid_probs * weights.unsqueeze(1), dim=0
        )
        final_probabilities = final_probabilities_tensor.numpy()
        final_logits = np.log(final_probabilities + 1e-9)

    elif aggregation_method == "robust_mean":
        # New robust aggregation method
        # Remove top and bottom 10% of predictions by confidence, then average
        max_probs, _ = torch.max(all_probabilities_tensor, dim=1)

        if len(max_probs) >= 5:  # Only apply if we have enough samples
            k_remove = max(1, len(max_probs) // 10)  # Remove 10%
            _, sorted_indices = torch.sort(max_probs)

            # Keep middle 80% of predictions
            keep_indices = (
                sorted_indices[k_remove:-k_remove]
                if k_remove < len(max_probs) // 2
                else sorted_indices
            )
            trimmed_probs = all_probabilities_tensor[keep_indices]

            final_probabilities_tensor = torch.mean(trimmed_probs, dim=0)
            final_probabilities = final_probabilities_tensor.numpy()
            final_logits = np.log(final_probabilities + 1e-9)

            logger.info(
                f"Robust mean: used {len(keep_indices)}/{len(max_probs)} predictions"
            )
        else:
            # Fallback to simple mean for small samples
            final_probabilities_tensor = torch.mean(all_probabilities_tensor, dim=0)
            final_probabilities = final_probabilities_tensor.numpy()
            final_logits = np.log(final_probabilities + 1e-9)

    else:
        logger.error(f"Invalid aggregation method: {aggregation_method}")
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    # Apply class balancing if enabled
    if enable_class_balancing:
        final_probabilities = apply_class_balancing(
            final_probabilities, method=bias_method, real_bias_factor=real_bias_factor
        )
        final_logits = np.log(final_probabilities + 1e-9)

    # Validate final probabilities
    if (
        np.any(final_probabilities < 0)
        or np.any(final_probabilities > 1)
        or not np.isclose(np.sum(final_probabilities), 1.0, atol=1e-5)
    ):
        logger.warning(f"Invalid probabilities detected: {final_probabilities}")
        # Normalize probabilities
        if np.sum(final_probabilities) > 0:
            final_probabilities = final_probabilities / np.sum(final_probabilities)
        else:
            num_classes = len(class_names)
            final_probabilities = (
                np.ones(num_classes) / num_classes
            )  # Uniform distribution
        final_logits = np.log(final_probabilities + 1e-9)
        logger.info(f"Normalized probabilities: {final_probabilities}")

    predicted_class_index = int(np.argmax(final_probabilities))
    predicted_class_name = class_names.get(predicted_class_index, "unknown")

    logger.info(
        f"Aggregation complete: class={predicted_class_name}, "
        f"probabilities={final_probabilities}, confidence={np.max(final_probabilities):.3f}"
    )

    return (
        final_probabilities,
        final_logits,
        predicted_class_name,
        predicted_class_index,
    )


def assess_prediction_reliability(
    valid_segments_count: int,
    total_segments: int,
    final_probabilities: np.ndarray,
    aggregation_method: str,
) -> Tuple[str, float]:
    """
    Assess the reliability of the prediction to provide fair confidence estimates.
    """
    if valid_segments_count == 0:
        return "no_data", 0.0

    segment_ratio = valid_segments_count / max(total_segments, 1)
    max_prob = np.max(final_probabilities)

    # Reliability categories
    if segment_ratio < 0.3:
        reliability = "low"
        confidence_multiplier = 0.5
    elif segment_ratio < 0.7:
        reliability = "medium"
        confidence_multiplier = 0.8
    else:
        reliability = "high"
        confidence_multiplier = 1.0

    # Adjust confidence based on aggregation method
    if aggregation_method in ["majority_vote", "robust_mean"]:
        confidence_multiplier *= 0.9  # Slightly more conservative
    elif aggregation_method == "confidence_weighted_probs":
        confidence_multiplier *= 1.1  # Slightly more confident

    adjusted_confidence = max_prob * confidence_multiplier

    logger.info(
        f"Prediction reliability: {reliability} "
        f"(segments: {valid_segments_count}/{total_segments}, "
        f"raw_conf: {max_prob:.3f}, adj_conf: {adjusted_confidence:.3f})"
    )

    return reliability, adjusted_confidence


def predict_audio_file(
    model: torch.nn.Module,
    audio_path: str,
    aggregation_method: str = settings.AGGREGATION_METHOD,
    enable_bias_reduction: bool = True,
    real_bias_factor: float = settings.REAL_BIAS_FACTOR,
    bias_method: str = settings.BIAS_METHOD,
) -> Tuple[np.ndarray, np.ndarray, str, int, int, List[np.ndarray], str]:
    """
    Enhanced audio file prediction with bias reduction techniques.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Using device: {device} for prediction for audio: {audio_path}")

    # Audio segmentation
    audio_segments = segment_audio(
        audio_path,
        target_sr=settings.TARGET_SAMPLE_RATE,
        segment_duration=settings.CHUNK_DURATION_SECONDS,
        overlap_duration=getattr(settings, "SEGMENT_OVERLAP_SECONDS", 0.0),
    )
    print(f"Created {len(audio_segments)} segments from {audio_path}.")

    # Process segments
    all_logits: List[torch.Tensor] = []
    valid_segments_count = 0

    for i, segment in enumerate(audio_segments):
        print(f"Processing segment {i + 1}/{len(audio_segments)}")
        logits = predict_single_segment(model, segment, device)

        if logits is not None:
            print(
                f"Segment {i + 1} logits shape: {logits.shape}, values: {logits.numpy()}"
            )
            all_logits.append(logits)
            valid_segments_count += 1
        else:
            print(f"Skipping segment {i + 1} due to processing error.")

    # Handle case with no valid segments
    if not all_logits:
        num_classes = len(settings.LABELS)
        print("No valid logits collected.")
        return (
            np.zeros(num_classes),
            np.zeros(num_classes),
            "no_valid_segments",
            -1,
            valid_segments_count,
            [],
            "no_data",
        )

    # Prepare raw logits for output
    raw_logits = []
    for logit in all_logits:
        squeezed_logit = logit.squeeze().numpy()
        print(
            f"Logit shape after squeeze: {squeezed_logit.shape}, values: {squeezed_logit}"
        )
        raw_logits.append(squeezed_logit)
    print(f"Collected {len(raw_logits)} raw logits for raw_output.")

    # Aggregate predictions with bias reduction
    final_probabilities, final_logits, predicted_class_name, predicted_class_index = (
        aggregate_predictions(
            all_logits,
            aggregation_method,
            enable_outlier_removal=enable_bias_reduction,
            enable_class_balancing=enable_bias_reduction,
            real_bias_factor=real_bias_factor,
            bias_method=bias_method,
        )
    )

    # Assess prediction reliability
    reliability, _ = assess_prediction_reliability(
        valid_segments_count,
        len(audio_segments),
        final_probabilities,
        aggregation_method,
    )

    print(f"Final probabilities: {final_probabilities}")
    return (
        final_probabilities,
        final_logits,
        predicted_class_name,
        predicted_class_index,
        valid_segments_count,
        raw_logits,
        reliability,
    )


@router.post("/predict/")
async def predict_endpoint(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form("vit_small"),
    aggregation_method: str = Form(settings.AGGREGATION_METHOD),
    enable_bias_reduction: bool = Form(True),
    real_bias_factor: float = Form(settings.REAL_BIAS_FACTOR),
    bias_method: str = Form(settings.BIAS_METHOD),
):
    """
    Enhanced prediction endpoint with bias reduction features.
    """
    pytorch_models = get_pytorch_models(request)
    if not pytorch_models:
        print("Error: PyTorch models not loaded in app.state.")
        raise HTTPException(status_code=503, detail="Models not loaded or unavailable.")

    selected_model = pytorch_models.get(model_name.lower())
    if selected_model is None:
        print(
            f"Error: Model '{model_name}' (searched as '{model_name.lower()}') not found or failed to load. "
            f"Available keys: {list(pytorch_models.keys())}"
        )

        if model_name.lower() in pytorch_models:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' was configured but appears to have failed to load correctly during server startup.",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not found. Available models are typically referenced by their lowercase names in the configuration (e.g., cnn_small, vit_large). Please check the available model identifiers.",
            )

    # Validate bias parameters
    if real_bias_factor <= 0:
        logger.warning(f"Invalid real_bias_factor: {real_bias_factor}, using 1.0")
        real_bias_factor = 1.0

    valid_bias_methods = ["equal_weight", "real_bias", "inverse_freq", "custom"]
    if bias_method not in valid_bias_methods:
        logger.warning(f"Unknown bias method '{bias_method}', using 'equal_weight'")
        bias_method = "equal_weight"

    # Validate aggregation method
    valid_methods = [
        "mean_logits",
        "mean_probs",
        "median_probs",
        "majority_vote",
        "confidence_weighted_probs",
        "robust_mean",
    ]
    if aggregation_method not in valid_methods:
        logger.warning(
            f"Unknown aggregation method '{aggregation_method}', using 'mean_probs'"
        )
        aggregation_method = "mean_probs"

    file_id = str(uuid.uuid4())
    original_filename = file.filename if file.filename else "unknown_file"
    file_extension = os.path.splitext(original_filename)[1]
    temp_audio_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

    print(f"Saving uploaded file '{original_filename}' to '{temp_audio_path}'")

    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Uploaded file '{original_filename}' saved successfully")

        # Enhanced prediction with bias reduction
        (
            probabilities,
            logits,
            class_name,
            class_idx,
            segments_processed,
            raw_logits,
            reliability,
        ) = predict_audio_file(
            model=selected_model,
            audio_path=temp_audio_path,
            aggregation_method=aggregation_method,
            enable_bias_reduction=enable_bias_reduction,
            real_bias_factor=real_bias_factor,
            bias_method=bias_method,
        )

        probabilities_list = probabilities.tolist()

        # Process raw_logits for JSON serialization
        try:
            processed_raw_output = []
            for logit_array in raw_logits:
                logit_list = logit_array.tolist()
                current_processed_list = []
                for x in logit_list:
                    if np.isnan(x) or np.isinf(x):
                        current_processed_list.append(str(x))
                    else:
                        current_processed_list.append(x)
                processed_raw_output.append(current_processed_list)
            raw_output = processed_raw_output
            print(f"Raw output content (first 2 processed): {raw_output[:2]}")
        except Exception as e:
            print(f"Error serializing raw_logits: {e}")
            raw_output = []
        print(f"Raw output prepared with {len(raw_output)} entries.")

        # Enhanced validation and confidence calculation
        if (
            not probabilities_list
            or any(
                np.isnan(p) or np.isinf(p) or p < 0 or p > 1 for p in probabilities_list
            )
            or not np.isclose(sum(probabilities_list), 1.0, atol=1e-5)
        ):
            print(f"Invalid probabilities: {probabilities_list}")
            probabilities_list = [1.0 / len(settings.LABELS)] * len(
                settings.LABELS
            )  # Uniform distribution
            class_idx = -1
            confidence = 0.0
            reliability = "invalid"
        else:
            raw_confidence = max(probabilities_list, default=0.0) * 100
            # Adjust confidence based on reliability
            reliability_multipliers = {
                "high": 1.0,
                "medium": 0.8,
                "low": 0.6,
                "no_data": 0.0,
                "invalid": 0.0,
            }
            confidence = raw_confidence * reliability_multipliers.get(reliability, 0.5)

        # Handle edge cases
        if segments_processed == 0:
            print(f"No valid segments processed for {original_filename}")
            return {
                "filename": original_filename,
                "model_used": model_name,
                "predicted_class_index": -1,
                "predicted_class_name": "no_valid_segments",
                "probabilities": [1.0 / len(settings.LABELS)] * len(settings.LABELS),
                "confidence": 0.0,
                "aggregation_method": aggregation_method,
                "segments_processed": segments_processed,
                "reliability": "no_data",
                "bias_reduction_enabled": enable_bias_reduction,
                "raw_output": [],
                "error": "No valid audio segments could be processed. Check audio file format or content.",
            }

        if class_idx == -1:
            print(f"Prediction inconclusive for {original_filename}")
            predicted_class_name = "inconclusive"
        else:
            predicted_class_name = settings.LABELS.get(class_idx, "unknown")

        print(
            f"Prediction result: class={predicted_class_name}, "
            f"probabilities={probabilities_list}, confidence={confidence:.1f}%, "
            f"reliability={reliability}"
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
            "reliability": reliability,
            "bias_reduction_enabled": enable_bias_reduction,
            "real_bias_factor": real_bias_factor,
            "bias_method": bias_method,
            "raw_output": raw_output,
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
        # Cleanup
        if hasattr(file, "file") and file.file and not file.file.closed:
            file.file.close()
            print(f"Closed file stream for {original_filename}")
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"Cleaned up temporary file: {temp_audio_path}")
            except OSError as e_os:
                print(f"Error deleting temporary file {temp_audio_path}: {e_os}")
