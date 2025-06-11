import click
import os
import torch
import numpy as np
import soundfile as sf
import sys  # For potentially modifying path if needed, though not strictly here.

# Ensure 'app' directory is in Python's path if cli.py is in root
# This allows 'from app.config import settings' to work.
# One way: sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Or ensure PYTHONPATH is set correctly when running. For now, assume it works.

from app.config import settings
from app.model_definitions import CNN_Audio, ViT_Audio
from app.audio_processing.utils import split_into_chunks
from app.routers.predict import process_audio_for_model  # For audio preprocessing

# Global variable to store loaded models to avoid reloading
loaded_models_cli = {}


def load_single_model_for_cli(model_name_to_load: str):
    """
    Synchronous function to load a single specified PyTorch model for CLI usage.
    Adapted from app.main.load_models.
    """
    if (
        model_name_to_load in loaded_models_cli
        and loaded_models_cli[model_name_to_load] is not None
    ):
        # click.echo(f"Model {model_name_to_load} already loaded.", err=True, color='yellow') # Optional: for debugging
        return loaded_models_cli[model_name_to_load]

    click.echo(
        f"Attempting to load PyTorch model for CLI: {model_name_to_load}...", err=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}", err=True)

    model_path = None
    model_architecture_config = {}

    # Determine model file path and architecture configurations
    # These should mirror the logic in app/main.py's load_models for consistency in parameters
    # For ViT, ensure parameters like 'dim', 'heads', 'mlp_dim' are correctly derived/mapped for ViT_Audio class

    base_model_params = {
        "img_size": settings.N_MELS,
        "in_channels": 1,
        "num_classes": len(settings.LABELS),
    }

    if model_name_to_load == "cnn_small":
        model_path = os.path.join(settings.MODEL_DIR, settings.CNN_SMALL_MODEL_NAME)
        model_architecture_config = {
            "type": "cnn",
            "params": {
                **base_model_params,
                "cnn_conv_channels": [32, 64, 128],
                "cnn_pool_after_conv": [True, True, True],
                "linear_output_units_1st_fc": 192,
                # CNN_Audio uses a default dropout if not specified.
            },
        }
    elif model_name_to_load == "cnn_large":
        model_path = os.path.join(settings.MODEL_DIR, settings.CNN_LARGE_MODEL_NAME)
        model_architecture_config = {
            "type": "cnn",
            "params": {
                **base_model_params,
                "cnn_conv_channels": [64, 128, 256, 512, 512],
                "cnn_pool_after_conv": [True, True, True, True, False],
                "linear_output_units_1st_fc": 192,
            },
        }
    elif model_name_to_load == "vit_small":
        model_path = os.path.join(settings.MODEL_DIR, settings.VIT_SMALL_MODEL_NAME)
        vit_specific_params = {
            "patch_size": 16,
            "embed_dim": 192,  # This will be mapped to 'dim'
            "depth": 4,
            "num_heads": 4,  # This will be mapped to 'heads'
            "mlp_ratio": 2.0,  # Used to calculate mlp_dim
            "dropout": 0.1,
        }
        model_architecture_config = {
            "type": "vit",
            "params": {
                **base_model_params,
                "patch_size": vit_specific_params["patch_size"],
                "dim": vit_specific_params["embed_dim"],
                "depth": vit_specific_params["depth"],
                "heads": vit_specific_params["num_heads"],
                "mlp_dim": int(
                    vit_specific_params["embed_dim"] * vit_specific_params["mlp_ratio"]
                ),
                "dropout": vit_specific_params["dropout"],
            },
        }
    elif model_name_to_load == "vit_large":
        model_path = os.path.join(settings.MODEL_DIR, settings.VIT_LARGE_MODEL_NAME)
        vit_specific_params = {  # Assuming some params are shared with vit_small as per app/main.py logic
            "patch_size": 16,
            "embed_dim": 384,  # Larger embed_dim
            "depth": 6,  # Assuming same depth
            "num_heads": 8,  # More heads
            "mlp_ratio": 2.0,  # Assuming same mlp_ratio
            "dropout": 0.1,
        }
        model_architecture_config = {
            "type": "vit",
            "params": {
                **base_model_params,
                "patch_size": vit_specific_params["patch_size"],
                "dim": vit_specific_params["embed_dim"],
                "depth": vit_specific_params["depth"],
                "heads": vit_specific_params["num_heads"],
                "mlp_dim": int(
                    vit_specific_params["embed_dim"] * vit_specific_params["mlp_ratio"]
                ),
                "dropout": vit_specific_params["dropout"],
            },
        }
    else:
        click.echo(
            f"Error: Model name '{model_name_to_load}' is not recognized or configured in cli.py.",
            err=True,
        )
        return None

    if not model_path or not os.path.exists(model_path):
        abs_model_path = os.path.abspath(model_path) if model_path else "Not specified"
        click.echo(f"Error: Model file not found at {abs_model_path}", err=True)
        click.echo(
            f"Please check MODEL_DIR in .env or app/config.py, and model filenames.",
            err=True,
        )
        loaded_models_cli[model_name_to_load] = None
        return None

    try:
        click.echo(
            f"Loading PyTorch model: {model_name_to_load} from {os.path.abspath(model_path)}",
            err=True,
        )

        model = None
        if model_architecture_config["type"] == "cnn":
            model = CNN_Audio(**model_architecture_config["params"])
        elif model_architecture_config["type"] == "vit":
            model = ViT_Audio(**model_architecture_config["params"])

        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        loaded_models_cli[model_name_to_load] = model
        click.echo(
            f"Successfully loaded PyTorch model for CLI: {model_name_to_load}", err=True
        )
        return model

    except Exception as e:
        click.echo(
            f"Error loading PyTorch model {model_name_to_load} from {model_path}: {e}",
            err=True,
        )
        # import traceback # Uncomment for detailed stack trace during debugging
        # traceback.print_exc()
        loaded_models_cli[model_name_to_load] = None
        return None


@click.command()
@click.argument(
    "audio_file_path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "--model_name",
    default=settings.CNN_SMALL_MODEL_NAME.split(".")[0]
    if settings.CNN_SMALL_MODEL_NAME
    else "cnn_small",  # Default to cnn_small or first part of its filename
    help="Model to use for prediction.",
    type=click.Choice(
        ["cnn_small", "cnn_large", "vit_small", "vit_large"], case_sensitive=False
    ),
)
def predict(audio_file_path: str, model_name: str):
    """
    Predicts whether an AUDIO_FILE_PATH is real or fake using the specified audio model.
    Example: python cli.py path/to/your/audio.wav --model_name cnn_large
    """
    click.echo(f"Starting prediction for: {os.path.abspath(audio_file_path)}")
    click.echo(f"Using model: {model_name}")

    # 1. Load Model
    actual_model_name_key = model_name  # This should be 'cnn_small', 'vit_large' etc.
    model = load_single_model_for_cli(actual_model_name_key)
    if model is None:
        click.echo("Model could not be loaded. Exiting.", err=True)
        return

    try:
        # 2. Read and Process Audio
        original_waveform, original_sr = sf.read(audio_file_path, dtype="float32")
        click.echo(
            f"Audio loaded: {len(original_waveform)} samples, SR: {original_sr}Hz"
        )

        if original_waveform.ndim > 1 and original_waveform.shape[1] > 1:
            click.echo("Audio is stereo, converting to mono.")
            original_waveform = np.mean(original_waveform, axis=1)

        if len(original_waveform) == 0:
            click.echo(
                "Error: Audio file is empty or could not be read properly.", err=True
            )
            return

        chunks = split_into_chunks(
            original_waveform, original_sr, settings.CHUNK_DURATION_SECONDS
        )
        click.echo(
            f"Audio split into {len(chunks)} chunks of approx. {settings.CHUNK_DURATION_SECONDS}s each."
        )

        overall_predictions = []
        for i, chunk_waveform in enumerate(chunks):
            click.echo(f"  Processing chunk {i + 1}/{len(chunks)}...")

            input_data_np = process_audio_for_model(chunk_waveform, original_sr)
            input_tensor = torch.from_numpy(input_data_np)

            try:
                device = next(model.parameters()).device
            except StopIteration:  # Should not happen if model loaded
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = input_tensor.to(device)

            # 3. Perform Inference
            with torch.no_grad():
                outputs = model(input_tensor)

            prediction_scores = outputs.cpu().detach().numpy()[0]

            # 4. Interpret Results (aligning with app/routers/predict.py)
            # The model output can be either 2 scores (logits/probabilities for real/fake) or 1 score.
            # settings.LABELS = {0: "real", 1: "fake"}
            # settings.REAL_LABEL_INDEX = 0, settings.FAKE_LABEL_INDEX = 1

            result_label = "N/A"
            confidence = 0.0

            if (
                prediction_scores.size == 2
            ):  # Two output scores (e.g., logits for class 0 and class 1)
                probs = torch.softmax(
                    torch.from_numpy(prediction_scores), dim=-1
                ).numpy()
                predicted_index = np.argmax(probs)
                confidence = float(probs[predicted_index])
                result_label = settings.LABELS.get(predicted_index, "unknown_index")
            elif prediction_scores.size == 1:  # Single output score
                # This case depends on what the single score represents (e.g., P(real) or P(fake) or a logit)
                # The logic in app/routers/predict.py for single output:
                #   confidence_val = float(prediction[0][0])
                #   if confidence_val > 0.5: result_label = settings.LABELS.get(settings.REAL_LABEL_INDEX, "real")
                #   else: result_label = settings.LABELS.get(settings.FAKE_LABEL_INDEX, "deepfake")
                # This implies the single score is P(real) or a score for 'real' class.
                score_for_real = float(prediction_scores[0])
                if score_for_real > 0.5:  # Thresholding at 0.5
                    result_label = settings.LABELS.get(
                        settings.REAL_LABEL_INDEX, "real"
                    )
                    confidence = score_for_real
                else:
                    result_label = settings.LABELS.get(
                        settings.FAKE_LABEL_INDEX, "fake"
                    )  # "fake" or "deepfake"
                    confidence = 1.0 - score_for_real  # Confidence in the "fake" label
            else:
                click.echo(
                    f"    Warning: Unexpected model output shape for score interpretation: {prediction_scores.shape}",
                    err=True,
                )
                result_label = "error_shape"
                confidence = 0.0

            chunk_result = {
                "file": os.path.basename(audio_file_path),
                "chunk_index": i + 1,
                "num_chunks": len(chunks),
                "prediction": result_label,
                "confidence": f"{confidence:.4f}",
                "model_used": actual_model_name_key,
            }
            overall_predictions.append(chunk_result)
            click.echo(
                f"    Chunk {i + 1}: Label: {result_label}, Confidence: {confidence:.4f}"
            )

        # Print a summary table at the end (optional, but good for CLI)
        click.echo("--- Prediction Summary ---")
        # Header
        click.echo(f"{'Chunk':<7} | {'Prediction':<10} | {'Confidence':<10}")
        click.echo("-" * 30)
        for res in overall_predictions:
            click.echo(
                f"{res['chunk_index']:<7} | {res['prediction']:<10} | {res['confidence']:<10}"
            )

    except sf.LibsndfileError as lse:
        click.echo(
            f"Error reading audio file '{audio_file_path}': {lse}. Ensure it's a valid audio format.",
            err=True,
        )
    except FileNotFoundError:  # Should be caught by click.Path, but as a fallback.
        click.echo(f"Error: Audio file not found at {audio_file_path}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred during prediction: {e}", err=True)
        # import traceback # Uncomment for detailed stack trace
        # traceback.print_exc()


if __name__ == "__main__":
    # This makes the script executable and engages click.
    # To run: python cli.py your_audio.wav --model_name cnn_small
    predict()
