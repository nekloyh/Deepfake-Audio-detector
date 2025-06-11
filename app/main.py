import os
import torch
import numpy as np
from dataclasses import dataclass, field
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import the router
from .routers import predict as predict_router
from .model_definitions import Config, CNN_Audio

# --- FastAPI App Initialization ---
app = FastAPI()

# Mount static files (CSS, JS) to be served from the '/static' path.
app.mount("/static", StaticFiles(directory="app/static"), name="static")
# Initialize Jinja2 templates for rendering HTML.
templates = Jinja2Templates(directory="app/templates")

# --- PyTorch Model Loading ---
# Import settings from the application's configuration module.
from .config import settings

# Determine model directory from settings.
MODEL_DIR = settings.MODEL_DIR
# `pytorch_models` will store loaded PyTorch model objects.
# It's populated by `load_models` during app startup and attached to `app.state`.
# Keys are model identifiers (e.g., "cnn_small"), values are PyTorch model objects.
pytorch_models: dict = {}


@app.on_event("startup")
async def load_models():
    """
    Event handler for application startup.
    Loads PyTorch models specified in the application settings (config.py).
    The loaded models are stored in the `pytorch_models` dictionary, which is then
    attached to `app.state.pytorch_models`, making them accessible from request handlers.
    Models are identified by keys like "cnn_small", "cnn_large".
    """
    print(
        f"Attempting to load PyTorch models from directory: {os.path.abspath(settings.MODEL_DIR)}"
    )
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Ensure model directory exists, create if not. This helps if deploying to a new environment.
    os.makedirs(settings.MODEL_DIR, exist_ok=True)

    # Dynamically builds a dictionary of model names (keys) to their file paths (values)
    # based on attributes defined in the `settings` object. This approach allows models
    # to be easily added or removed via `config.py` without changing this loading logic.
    current_model_paths = {}
    # Check for CNN models defined in settings
    if hasattr(settings, "CNN_SMALL_MODEL_NAME") and settings.CNN_SMALL_MODEL_NAME:
        current_model_paths["cnn_small"] = os.path.join(
            MODEL_DIR, settings.CNN_SMALL_MODEL_NAME
        )
    if hasattr(settings, "CNN_LARGE_MODEL_NAME") and settings.CNN_LARGE_MODEL_NAME:
        current_model_paths["cnn_large"] = os.path.join(
            MODEL_DIR, settings.CNN_LARGE_MODEL_NAME
        )

    # Example for ViT models (or other types): these will only be added if
    # their respective _MODEL_NAME attributes are defined and non-empty in settings.
    if hasattr(settings, "VIT_SMALL_MODEL_NAME") and settings.VIT_SMALL_MODEL_NAME:
        current_model_paths["vit_small"] = os.path.join(
            MODEL_DIR, settings.VIT_SMALL_MODEL_NAME
        )
    if hasattr(settings, "VIT_LARGE_MODEL_NAME") and settings.VIT_LARGE_MODEL_NAME:
        current_model_paths["vit_large"] = os.path.join(
            MODEL_DIR, settings.VIT_LARGE_MODEL_NAME
        )

    if not current_model_paths:
        print(
            "Warning: No model names configured in settings. No PyTorch models will be loaded."
        )
        app.state.pytorch_models = pytorch_models
        return

    # Add numpy to safe globals for torch.load if models were saved with numpy scalars
    # This is a workaround for a change in PyTorch 1.9+ that restricts loading pickles with numpy scalars
    # torch.serialization.add_safe_globals([np._core.multiarray.scalar]) # Example, adjust if needed

    for model_name, model_path in current_model_paths.items():
        if not os.path.exists(model_path):
            print(
                f"Warning: Model file not found at {model_path} (abs: {os.path.abspath(model_path)})"
            )
            pytorch_models[model_name] = None
            continue
        try:
            print(
                f"Loading PyTorch model: {model_name} from {model_path} (abs: {os.path.abspath(model_path)})"
            )

            # Create model configuration
            model_config_params = {
                "img_size": 224,  # Assuming square inputs, N_MELS would be used for height if different
                "in_channels": 1,
                "num_classes": 2,  # For binary classification real/fake
                # Other default parameters from Config can be set here if needed
            }
            if model_name == "cnn_small":
                model_config_params.update(
                    {
                        "cnn_conv_channels": [32, 64, 128],
                        "cnn_pool_after_conv": [True, True, True],
                        "linear_output_units_1st_fc": 192,
                    }
                )
            elif model_name == "cnn_large":
                model_config_params.update(
                    {
                        "cnn_conv_channels": [64, 128, 256, 512, 512],
                        "cnn_pool_after_conv": [True, True, True, True, False],
                        "linear_output_units_1st_fc": 192,  # Or other appropriate value
                    }
                )
            else:
                print(
                    f"Warning: Model name {model_name} not recognized for specific config. Using default CNN params."
                )
                # Fallback or error, here we might use some default config or skip
                model_config_params.update(
                    {  # Defaulting to a smallish config
                        "cnn_conv_channels": [32, 64],
                        "cnn_pool_after_conv": [True, True],
                        "linear_output_units_1st_fc": 128,
                    }
                )

            # Instantiate the model
            # Note: The Config class defined above is a general config, not directly the model's hyperparameter config.
            # We are directly passing parameters to CNN_Audio constructor.
            model = CNN_Audio(
                img_size=model_config_params["img_size"],
                in_channels=model_config_params["in_channels"],
                num_classes=model_config_params["num_classes"],
                linear_output_units_1st_fc=model_config_params[
                    "linear_output_units_1st_fc"
                ],
                cnn_conv_channels=model_config_params["cnn_conv_channels"],
                cnn_pool_after_conv=model_config_params["cnn_pool_after_conv"],
                # dropout can be taken from settings or a default
            )

            # Load the checkpoint
            # Set weights_only=False if the checkpoint contains more than just model state_dict (e.g., optimizer state)
            # However, for inference, typically only model_state_dict is needed.
            # If .pth was saved with torch.save(model.state_dict(), PATH), then load directly.
            # If .pth was saved with torch.save({'model_state_dict': model.state_dict(), ...}, PATH), then access the key.
            # The provided conversion script uses the latter.
            checkpoint = torch.load(
                model_path, map_location=device
            )  # , weights_only=True potentially if only state_dict

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # This case handles checkpoints that are just the state_dict
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()

            pytorch_models[model_name] = model
            print(f"Successfully loaded PyTorch model: {model_name}")

        except Exception as e:
            print(f"Error loading PyTorch model {model_name} from {model_path}: {e}")
            pytorch_models[model_name] = None

    # Post-loading checks and warnings.
    if not pytorch_models:
        print("CRITICAL WARNING: No models were configured or attempted to load.")
    elif all(model is None for model in pytorch_models.values()):
        print(
            "CRITICAL WARNING: All configured PyTorch models failed to load. Predictions will fail."
        )
    elif any(model is None for model in pytorch_models.values()):
        failed_models = [
            name for name, model in pytorch_models.items() if model is None
        ]
        print(
            f"Warning: Some PyTorch models could not be loaded: {failed_models}. Check paths and file integrity."
        )
    else:
        print("All configured PyTorch models loaded successfully.")

    # IMPORTANT: Attach the loaded pytorch_models to the app's state
    app.state.pytorch_models = pytorch_models


# --- Endpoints ---
# Serves the main HTML page from the templates directory.
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main index.html page."""
    return templates.TemplateResponse("index.html", {"request": request})


# Include the router for prediction endpoints (e.g., /predict_audio)
app.include_router(predict_router.router)


# Standard Python entry point for running the Uvicorn server.
if __name__ == "__main__":
    import uvicorn

    print(
        f"Starting Uvicorn server for FastAPI app (app.main) on {settings.HOST}:{settings.PORT}..."
    )
    uvicorn.run(
        "app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG
    )
