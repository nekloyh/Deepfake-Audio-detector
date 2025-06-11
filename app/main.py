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


# --- PyTorch Model and Config Definitions ---
# Define Config class (copied from models/convert_cnn_small_to_onnx.py)
@dataclass
class Config:
    # Data processing parameters
    SEED: int = 42
    SR: int = 16000
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    N_MELS: int = 128
    FMIN: float = 0.0
    FMAX: float = 8000.0
    NUM_TIME_MASKS: int = 2
    NUM_FREQ_MASKS: int = 2
    TIME_MASK_MAX_WIDTH: int = 30
    FREQ_MASK_MAX_WIDTH: int = 15
    MASK_REPLACEMENT_VALUE: float = -80.0
    NORM_EPSILON: float = 1e-6
    LOUDNESS_LUFS: float = -23.0
    USE_GLOBAL_NORMALIZATION: bool = True
    USE_RANDOM_CROPPING: bool = True
    CACHE_DIR_BASE: str = "/kaggle/input/cnn-3s-dataset"
    DATASET_SUBDIR: str = "cnn_3s_dataset"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    metadata_file: str = "kaggle_metadata.csv"

    # Model architecture
    img_size: int = 224
    num_classes: int = 2
    in_channels: int = 1
    dropout: float = 0.1

    # CNN specific architecture parameters
    cnn_conv_channels: list[int] = field(default_factory=list)
    cnn_pool_after_conv: list[bool] = field(default_factory=list)
    linear_output_units_1st_fc: int = 512

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 20
    weight_decay: float = 1e-4
    num_workers: int = 4

    # Data augmentation
    apply_augmentation: bool = True
    augmentation_prob: float = 0.5
    audio_length_seconds: float = 3.0
    overlap_ratio: float = 0.5

    model_size: str = ""
    dataset_name: str = ""

    def validate(self):
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert len(self.cnn_conv_channels) == len(self.cnn_pool_after_conv), (
            "cnn_conv_channels and cnn_pool_after_conv must have the same length"
        )


# Define CNN_Audio class (copied from models/convert_cnn_small_to_onnx.py)
class CNN_Audio(torch.nn.Module):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: int,
        linear_output_units_1st_fc: int,
        cnn_conv_channels: list[int],
        cnn_pool_after_conv: list[bool],
        dropout: float = 0.1,
    ):
        super(CNN_Audio, self).__init__()
        self.in_channels = in_channels
        self.cnn_conv_channels = cnn_conv_channels
        self.cnn_pool_after_conv = cnn_pool_after_conv
        self.img_size = img_size
        self.dropout = dropout
        self.num_classes = num_classes

        layers = []
        in_dim = self.in_channels

        for i, out_dim in enumerate(self.cnn_conv_channels):
            layers.append(
                torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
            )
            layers.append(torch.nn.BatchNorm2d(out_dim))
            layers.append(torch.nn.ReLU(inplace=True))

            if self.cnn_pool_after_conv[i]:
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

            layers.append(torch.nn.Dropout2d(self.dropout))
            in_dim = out_dim

        self.conv_layers = torch.nn.Sequential(*layers)

        # Calculate output size of conv layers dynamically
        with torch.no_grad():
            dummy_input_for_init = torch.randn(
                1, self.in_channels, self.img_size, self.img_size
            )
            dummy_output_conv = self.conv_layers(dummy_input_for_init)
            # self.flattened_size = dummy_output_conv.view(1, -1).size(1) # Not used directly

        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))

        with torch.no_grad():
            dummy_pooled = self.adaptive_pool(dummy_output_conv)
            self.pooled_size = dummy_pooled.view(1, -1).size(1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.pooled_size, linear_output_units_1st_fc),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                linear_output_units_1st_fc, linear_output_units_1st_fc // 2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(linear_output_units_1st_fc // 2, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


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
