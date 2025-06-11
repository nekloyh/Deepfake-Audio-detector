# prediction_pipeline.py
import torch
import numpy as np
from typing import List, Tuple, Literal

# Import classes and functions from the other two files
from audio_segmentation import segment_audio, AudioConfig
from spectrogram_processing import (
    create_mel_spectrogram,
    preprocess_spectrogram_to_tensor,
    SpectrogramConfig,
)

# Assume you have your ViT model definition in a separate module
# For example, if your ViT_Small class is in 'models/vit_model.py'
try:
    from models.vit_model import ViT_Small  # Adjust this import path as needed
except ImportError:
    print("Warning: Could not import ViT_Small from models/vit_model.py.")
    print("Please ensure your ViT_Small class is accessible or adjust the import path.")

    # Fallback/dummy class if ViT_Small isn't available for testing purposes
    class ViT_Small(torch.nn.Module):
        def __init__(
            self,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            emb_dropout,
        ):
            super().__init__()
            print(
                "Using dummy ViT_Small for demonstration. Model will not function correctly without actual implementation."
            )
            self.linear = torch.nn.Linear(dim, num_classes)  # Dummy layer

        def forward(self, x):
            # Dummy forward pass that returns random logits
            return torch.randn(x.shape[0], 2)  # Assuming 2 classes (real/fake)


class PredictionConfig:
    """
    Configuration for the prediction process.
    """

    model_path = (
        "path/to/your_trained_model_small.pth"  # !! IMPORTANT: Update this path !!
    )
    class_names = ["real", "fake"]  # Update with your actual class names and order

    # Model parameters from your vit-trainer.ipynb (training_params_small or similar)
    # These MUST match the parameters used to train your .pth model
    vit_model_params = {
        "image_size": SpectrogramConfig.image_size,  # Should be 224
        "patch_size": 16,  # VERIFY from your vit-trainer.ipynb
        "num_classes": len(class_names),
        "dim": 768,  # VERIFY
        "depth": 12,  # VERIFY
        "heads": 12,  # VERIFY
        "mlp_dim": 3072,  # VERIFY
        "dropout": 0.1,  # VERIFY
        "emb_dropout": 0.1,  # VERIFY
    }


def load_model(
    model_path: str, model_params: dict, device: torch.device
) -> torch.nn.Module:
    """
    Loads the trained ViT model.
    """
    model = ViT_Small(**model_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    model.to(device)
    return model


def predict_single_segment(
    model: torch.nn.Module, segment_waveform: np.ndarray, device: torch.device
) -> torch.Tensor:
    """
    Processes a single audio segment and makes a prediction.
    Returns raw logits for the segment.
    """
    log_mel_spectrogram = create_mel_spectrogram(segment_waveform)
    if log_mel_spectrogram is None:
        return None  # Indicate failure for this segment

    input_tensor = preprocess_spectrogram_to_tensor(log_mel_spectrogram)
    if input_tensor is None:
        return None  # Indicate failure for this segment

    # Add batch dimension and move to device
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output_logits = model(input_tensor)  # Get logits
    return output_logits.cpu()  # Move back to CPU for aggregation


def aggregate_predictions(
    all_logits: List[torch.Tensor],
    aggregation_method: Literal[
        "mean_logits", "mean_probs", "majority_vote"
    ] = "mean_probs",
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """
    Aggregates predictions from multiple segments into a single final prediction.

    Args:
        all_logits (List[torch.Tensor]): List of raw logits from each segment.
        aggregation_method (str): How to aggregate:
            - 'mean_logits': Average the logits across segments, then apply softmax.
            - 'mean_probs': Apply softmax to each segment's logits, then average probabilities.
            - 'majority_vote': Predict class for each segment, then take the most frequent class.

    Returns:
        Tuple[np.ndarray, np.ndarray, str, int]:
            - final_probabilities (np.ndarray): Probabilities for each class for the entire audio.
            - final_logits (np.ndarray): Aggregated logits.
            - predicted_class_name (str): Name of the predicted class.
            - predicted_class_index (int): Index of the predicted class.
    """
    if not all_logits:
        # Handle case where no valid segments were processed
        num_classes = len(PredictionConfig.class_names)
        return (np.zeros(num_classes), np.zeros(num_classes), "unknown", -1)

    if aggregation_method == "mean_logits":
        # Stack logits and average
        stacked_logits = torch.stack(all_logits)
        mean_logits = torch.mean(stacked_logits, dim=0)
        final_probabilities = torch.softmax(mean_logits, dim=1).squeeze(0).numpy()
        final_logits = mean_logits.squeeze(0).numpy()
    elif aggregation_method == "mean_probs":
        # Apply softmax to each, then average probabilities
        all_probabilities = [torch.softmax(lg, dim=1) for lg in all_logits]
        stacked_probabilities = torch.stack(all_probabilities)
        mean_probabilities = torch.mean(stacked_probabilities, dim=0)
        final_probabilities = mean_probabilities.squeeze(0).numpy()
        # To get equivalent logits for mean_probs, it's not straightforward without inverse softmax
        # For simplicity, we can just take the log of the probabilities, or skip reporting logits
        final_logits = np.log(
            final_probabilities + 1e-9
        )  # Add small epsilon to avoid log(0)
    elif aggregation_method == "majority_vote":
        # Predict class for each segment
        predicted_classes_per_segment = [
            torch.argmax(torch.softmax(lg, dim=1), dim=1).item() for lg in all_logits
        ]
        if not predicted_classes_per_segment:
            num_classes = len(PredictionConfig.class_names)
            return (np.zeros(num_classes), np.zeros(num_classes), "unknown", -1)

        # Count occurrences of each class
        counts = np.bincount(
            predicted_classes_per_segment, minlength=len(PredictionConfig.class_names)
        )
        predicted_class_index = np.argmax(counts)
        predicted_class_name = PredictionConfig.class_names[predicted_class_index]

        # For 'majority_vote', calculating a single 'probability' is tricky.
        # We can report the proportion of votes for the winning class.
        total_segments = len(predicted_classes_per_segment)
        if total_segments > 0:
            final_probabilities = counts / total_segments
        else:
            final_probabilities = np.zeros(
                len(PredictionConfig.class_names)
            )  # No segments, all zeros

        # Logits are not directly meaningful for majority vote aggregation
        final_logits = np.log(
            final_probabilities + 1e-9
        )  # Pseudo-logits for consistency
        return (
            final_probabilities,
            final_logits,
            predicted_class_name,
            predicted_class_index,
        )
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    predicted_class_index = np.argmax(final_probabilities)
    predicted_class_name = PredictionConfig.class_names[predicted_class_index]

    return (
        final_probabilities,
        final_logits,
        predicted_class_name,
        predicted_class_index,
    )


def predict_audio_file(
    audio_path: str, aggregation_method: str = "mean_probs"
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """
    Main function to predict the class of an entire audio file.

    Args:
        audio_path (str): Path to the input audio file (.wav, .flac).
        aggregation_method (str): Method to aggregate segment predictions.
                                  Options: 'mean_logits', 'mean_probs', 'majority_vote'.

    Returns:
        Tuple[np.ndarray, np.ndarray, str, int]:
            - final_probabilities (np.ndarray): Probabilities for each class for the entire audio.
            - final_logits (np.ndarray): Aggregated logits.
            - predicted_class_name (str): Name of the predicted class.
            - predicted_class_index (int): Index of the predicted class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model once
    model = load_model(
        PredictionConfig.model_path, PredictionConfig.vit_model_params, device
    )
    print(f"Model loaded from {PredictionConfig.model_path}")

    # Segment the audio file
    print(
        f"Segmenting audio file: {audio_path} into {AudioConfig.segment_duration}s chunks..."
    )
    audio_segments = segment_audio(audio_path)
    print(f"Created {len(audio_segments)} segments.")

    all_logits: List[torch.Tensor] = []
    for i, segment in enumerate(audio_segments):
        # Predict for each segment
        logits = predict_single_segment(model, segment, device)
        if logits is not None:
            all_logits.append(logits)
        else:
            print(f"Skipping segment {i} due to processing error.")

    if not all_logits:
        print("No valid segments processed. Cannot make a prediction.")
        num_classes = len(PredictionConfig.class_names)
        return (np.zeros(num_classes), np.zeros(num_classes), "unknown", -1)

    # Aggregate predictions
    print(
        f"Aggregating {len(all_logits)} segment predictions using '{aggregation_method}' method..."
    )
    final_probabilities, final_logits, predicted_class_name, predicted_class_index = (
        aggregate_predictions(all_logits, aggregation_method)
    )

    return (
        final_probabilities,
        final_logits,
        predicted_class_name,
        predicted_class_index,
    )


if __name__ == "__main__":
    import os
    import soundfile as sf

    # Create a dummy 20-second WAV file for testing
    print("Creating a dummy 20-second WAV file for testing...")
    dummy_sr = AudioConfig.sample_rate
    dummy_duration = 20  # seconds
    t = np.linspace(0, dummy_duration, int(dummy_sr * dummy_duration), endpoint=False)
    dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn(
        int(dummy_sr * dummy_duration)
    )
    dummy_audio_path = "test_audio_20s_for_prediction.wav"
    sf.write(dummy_audio_path, dummy_audio, dummy_sr)
    print(f"Dummy 20s audio saved to {dummy_audio_path}")

    # --- IMPORTANT ---
    # Before running this example, you MUST:
    # 1. Place a dummy or actual ViT_Small definition in 'models/vit_model.py'
    #    (or adjust the import in prediction_pipeline.py)
    # 2. Update PredictionConfig.model_path to a valid .pth file
    #    (or comment out load_model and use a dummy model for testing)
    # 3. Update PredictionConfig.vit_model_params with your actual model's architecture
    #    (these are crucial for the model instantiation)

    print("\n--- Running prediction pipeline ---")
    # Example using 'mean_probs' aggregation
    final_probs, final_logits, predicted_name, predicted_idx = predict_audio_file(
        dummy_audio_path, aggregation_method="mean_probs"
    )

    print(f"\nFinal Predicted Probabilities: {final_probs}")
    print(f"Final Predicted Logits (approx): {final_logits}")
    print(f"Predicted Class: {predicted_name} (Index: {predicted_idx})")

    # Example using 'majority_vote' aggregation
    print("\n--- Running prediction pipeline with majority_vote ---")
    final_probs_mv, final_logits_mv, predicted_name_mv, predicted_idx_mv = (
        predict_audio_file(dummy_audio_path, aggregation_method="majority_vote")
    )
    print(f"\nFinal Predicted Probabilities (Majority Vote): {final_probs_mv}")
    print(f"Final Predicted Logits (Majority Vote, approx): {final_logits_mv}")
    print(
        f"Predicted Class (Majority Vote): {predicted_name_mv} (Index: {predicted_idx_mv})"
    )

    # Clean up dummy file
    if os.path.exists(dummy_audio_path):
        os.remove(dummy_audio_path)
        print(f"\nCleaned up {dummy_audio_path}")
