import os
import torch
import numpy as np  # Keep this, it might be needed for torch.serialization.add_safe_globals
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import the router
from .routers import predict as predict_router
from .model_definitions import CNN_Audio, ViT_Audio
from .config import settings

# Optional: Add numpy.core.multiarray.scalar to safe globals if you prefer this over weights_only=False for all loads
# However, for trusted model files, using weights_only=False is often more straightforward.
# If you choose this, uncomment the following two lines:
# import numpy.core.multiarray
# torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
pytorch_models: dict = {}


@app.on_event("startup")
async def load_models():
    print(
        f"Attempting to load PyTorch models from directory: {os.path.abspath(settings.MODEL_DIR)}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(settings.MODEL_DIR, exist_ok=True)

    current_model_paths = {}
    if hasattr(settings, "CNN_SMALL_MODEL_NAME") and settings.CNN_SMALL_MODEL_NAME:
        current_model_paths["cnn_small"] = os.path.join(
            settings.MODEL_DIR, settings.CNN_SMALL_MODEL_NAME
        )
    if hasattr(settings, "CNN_LARGE_MODEL_NAME") and settings.CNN_LARGE_MODEL_NAME:
        current_model_paths["cnn_large"] = os.path.join(
            settings.MODEL_DIR, settings.CNN_LARGE_MODEL_NAME
        )
    if hasattr(settings, "VIT_SMALL_MODEL_NAME") and settings.VIT_SMALL_MODEL_NAME:
        current_model_paths["vit_small"] = os.path.join(
            settings.MODEL_DIR, settings.VIT_SMALL_MODEL_NAME
        )
    if hasattr(settings, "VIT_LARGE_MODEL_NAME") and settings.VIT_LARGE_MODEL_NAME:
        current_model_paths["vit_large"] = os.path.join(
            settings.MODEL_DIR, settings.VIT_LARGE_MODEL_NAME
        )

    if not current_model_paths:
        print(
            "Warning: No model names configured in settings. No PyTorch models will be loaded."
        )
        app.state.pytorch_models = pytorch_models
        return

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

            base_params = {
                "img_size": 224,
                "in_channels": 1,
                "num_classes": 2,
            }

            model = None

            # Specific model configurations
            if model_name == "cnn_small":
                # NOTE: CNN model parameters are hardcoded here. Verify these against the original training configurations if issues arise.
                cnn_params = {
                    **base_params,
                    "cnn_conv_channels": [32, 64, 128],
                    "cnn_pool_after_conv": [True, True, True],
                    "linear_output_units_1st_fc": 192,
                }
                model = CNN_Audio(**cnn_params)
            elif model_name == "cnn_large":
                # NOTE: CNN model parameters are hardcoded here. Verify these against the original training configurations if issues arise.
                cnn_params = {
                    **base_params,
                    "cnn_conv_channels": [64, 128, 256, 512, 512],
                    "cnn_pool_after_conv": [True, True, True, True, False],
                    "linear_output_units_1st_fc": 192,  # Note: Same as cnn_small. Confirm if intended.
                }
                model = CNN_Audio(**cnn_params)
            elif model_name == "vit_small":
                # NOTE: ViT model parameters are hardcoded here. Verify these against the original training configurations if issues arise.
                vit_config = {
                    "patch_size": 16,
                    "embed_dim": 128,
                    "depth": 4,
                    "num_heads": 4,
                    "mlp_ratio": 2.0,
                    "dropout": 0.1,
                }
                actual_vit_params = {
                    **base_params,
                    "patch_size": vit_config["patch_size"],
                    "dim": vit_config["embed_dim"],
                    "depth": vit_config["depth"],
                    "heads": vit_config["num_heads"],
                    "mlp_dim": int(vit_config["embed_dim"] * vit_config["mlp_ratio"]),
                    "dropout": vit_config["dropout"],
                }
                # print(f"Configuring parameters for ViT Small: {vit_config}")
                # print(
                #     f"Instantiating ViT_Audio for {model_name} with mapped params: {actual_vit_params}"
                # )
                model = ViT_Audio(**actual_vit_params)
            elif model_name == "vit_large":
                # NOTE: ViT model parameters are hardcoded here. Verify these against the original training configurations if issues arise.
                vit_config = {
                    "patch_size": 16,
                    "embed_dim": 384,
                    "depth": 6,  # Assuming depth is same as small
                    "num_heads": 8,
                    "mlp_ratio": 2.0,
                    "dropout": 0.1,  # Assuming mlp_ratio, dropout same as small
                }
                actual_vit_params = {
                    **base_params,
                    "patch_size": vit_config["patch_size"],
                    "dim": vit_config["embed_dim"],
                    "depth": vit_config["depth"],
                    "heads": vit_config["num_heads"],
                    "mlp_dim": int(vit_config["embed_dim"] * vit_config["mlp_ratio"]),
                    "dropout": vit_config["dropout"],
                }
                # print(f"Configuring parameters for ViT Large: {vit_config}")
                # print(
                #     f"Instantiating ViT_Audio for {model_name} with mapped params: {actual_vit_params}"
                # )
                model = ViT_Audio(**actual_vit_params)
            elif "cnn" in model_name:  # Fallback for other CNNs AFTER specific names
                print(
                    f"CRITICAL WARNING: Model name {model_name} looks like a CNN but is not explicitly 'cnn_small' or 'cnn_large'."
                )
                print(
                    "Attempting to load with generic small CNN parameters, but this is NOT RECOMMENDED."
                )
                cnn_params = {
                    **base_params,
                    "cnn_conv_channels": [32, 64],
                    "cnn_pool_after_conv": [True, True],
                    "linear_output_units_1st_fc": 128,
                }
                model = CNN_Audio(**cnn_params)
            elif "vit" in model_name:  # Fallback for other ViTs AFTER specific names
                print(
                    f"CRITICAL WARNING: Model name {model_name} looks like a ViT but is not 'vit_small' or 'vit_large'."
                )
                print(
                    "Attempting to load with generic small ViT parameters, but this is NOT RECOMMENDED."
                )
                vit_config = {  # Defaulting to small ViT params
                    "patch_size": 16,
                    "embed_dim": 192,
                    "depth": 12,
                    "num_heads": 3,
                    "mlp_ratio": 4.0,
                    "dropout": 0.1,
                }
                actual_vit_params = {
                    **base_params,
                    "patch_size": vit_config["patch_size"],
                    "dim": vit_config["embed_dim"],
                    "depth": vit_config["depth"],
                    "heads": vit_config["num_heads"],
                    "mlp_dim": int(vit_config["embed_dim"] * vit_config["mlp_ratio"]),
                    "dropout": vit_config["dropout"],
                }
                model = ViT_Audio(**actual_vit_params)
            else:
                print(
                    f"CRITICAL: Model type for {model_name} not recognized for instantiation."
                )
                pytorch_models[model_name] = None
                continue

            # Load checkpoint - MODIFIED HERE
            # Set weights_only=False if you trust the source of the checkpoint.
            # This is common for models saved in older PyTorch versions or containing numpy objects.
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()
            pytorch_models[model_name] = model
            print(f"Successfully loaded PyTorch model: {model_name}")

        except Exception as e:
            print(f"Error loading PyTorch model {model_name} from {model_path}: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging startup errors
            pytorch_models[model_name] = None

    app.state.pytorch_models = pytorch_models
    if not any(pytorch_models.values()):  # Check if all models failed to load
        print(
            "CRITICAL WARNING: All configured PyTorch models failed to load. Predictions will fail."
        )
    elif not all(
        m is not None for m in pytorch_models.values() if m is not None
    ):  # Check if some models failed
        failed_models = [
            name
            for name, model_obj in pytorch_models.items()
            if model_obj is None and name in current_model_paths
        ]
        if failed_models:
            print(
                f"Warning: Some PyTorch models could not be loaded: {failed_models}. Check paths and file integrity."
            )
    else:
        print("All configured PyTorch models loaded successfully.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


app.include_router(predict_router.router)

if __name__ == "__main__":
    import uvicorn

    print(
        f"Starting Uvicorn server for FastAPI app (app.main) on {settings.HOST}:{settings.PORT}..."
    )
    uvicorn.run(
        "app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG
    )
