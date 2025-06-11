import os
import onnxruntime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import the router
from .routers import predict as predict_router

# --- FastAPI App Initialization ---
app = FastAPI()

# Mount static files (CSS, JS) to be served from the '/static' path.
app.mount("/static", StaticFiles(directory="app/static"), name="static")
# Initialize Jinja2 templates for rendering HTML.
templates = Jinja2Templates(directory="app/templates")

# --- ONNX Model Loading ---
# Import settings from the application's configuration module.
from .config import settings

# Determine model directory from settings.
MODEL_DIR = settings.MODEL_DIR
# `onnx_sessions` will store loaded ONNX InferenceSession objects.
# It's populated by `load_models` during app startup and attached to `app.state`.
# Keys are model identifiers (e.g., "cnn_small"), values are InferenceSession objects.
onnx_sessions: dict = {}


@app.on_event("startup")
async def load_models():
    """
    Event handler for application startup.
    Loads ONNX models specified in the application settings (config.py).
    The loaded sessions are stored in the `onnx_sessions` dictionary, which is then
    attached to `app.state.onnx_sessions`, making them accessible from request handlers.
    Models are identified by keys like "cnn_small", "cnn_large".
    """
    print(
        f"Attempting to load ONNX models from directory: {os.path.abspath(settings.MODEL_DIR)}"
    )
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
            "Warning: No model names configured in settings. No ONNX models will be loaded."
        )
        app.state.onnx_sessions = onnx_sessions
        return

    for model_name, model_path in current_model_paths.items():
        if not os.path.exists(model_path):
            print(
                f"Warning: Model file not found at {model_path} (abs: {os.path.abspath(model_path)})"
            )
            onnx_sessions[model_name] = None
            continue
        try:
            print(
                f"Loading ONNX model: {model_name} from {model_path} (abs: {os.path.abspath(model_path)})"
            )
            session = onnxruntime.InferenceSession(model_path)
            onnx_sessions[model_name] = session
            print(f"Successfully loaded ONNX model: {model_name}")

            # Store the session in the dictionary.
            # If loading failed at Inferencesession(), the exception below would catch it.
            # If model file was not found earlier, it's already None.
            onnx_sessions[model_name] = session
            print(f"Successfully loaded ONNX model: {model_name}")

            # Inspect and print model input details (useful for debugging/verification)
            try:
                inputs = session.get_inputs()
                if inputs:
                    print(f"    Model '{model_name}' - Input Details:")
                    for i, input_meta in enumerate(inputs):
                        print(
                            f"        Input {i}: Name='{input_meta.name}', Shape={input_meta.shape}, Type={input_meta.type}"
                        )
                else:
                    print(
                        f"    Model '{model_name}': No inputs found in the model metadata."
                    )
            except Exception as e_inspect:
                print(
                    f"    Error inspecting inputs for model '{model_name}': {e_inspect}"
                )

        except Exception as e:
            print(f"Error loading ONNX model {model_name} from {model_path}: {e}")
            # Store None if loading failed; this allows the app to start but the specific model will be unusable.
            # Endpoints should check for None before trying to use a session.
            onnx_sessions[model_name] = None

    # Post-loading checks and warnings.
    if not onnx_sessions:  # No keys in current_model_paths
        # This case is already handled by the `if not current_model_paths:` check above,
        # but as a safeguard for the logic.
        print("CRITICAL WARNING: No models were configured or attempted to load.")
    elif all(session is None for session in onnx_sessions.values()):
        # This checks if the dictionary is not empty, but all its values are None (all models failed to load)
        print(
            "CRITICAL WARNING: All configured ONNX models failed to load. Predictions will fail."
        )
    elif any(session is None for session in onnx_sessions.values()):
        # Some models loaded, others failed.
        failed_models = [
            name for name, session in onnx_sessions.items() if session is None
        ]
        print(
            f"Warning: Some ONNX models could not be loaded: {failed_models}. Check paths and file integrity."
        )
    else:
        print("All configured ONNX models loaded successfully.")

    # IMPORTANT: Attach the loaded onnx_sessions to the app's state,
    # making them accessible from request handlers (e.g., via request.app.state.onnx_sessions).
    app.state.onnx_sessions = onnx_sessions


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
