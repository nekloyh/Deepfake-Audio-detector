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

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- ONNX Model Loading ---
# Import settings
from .config import settings

# Adjusted to use settings
MODEL_DIR = settings.MODEL_DIR
MODEL_PATHS = {
    "cnn_small": os.path.join(MODEL_DIR, settings.CNN_SMALL_MODEL_NAME),
    "cnn_large": os.path.join(MODEL_DIR, settings.CNN_LARGE_MODEL_NAME),
    # Example for ViT models if added to config:
    # "vit_small": os.path.join(MODEL_DIR, settings.VIT_SMALL_MODEL_NAME),
    # "vit_large": os.path.join(MODEL_DIR, settings.VIT_LARGE_MODEL_NAME),
}
# This dictionary will store the loaded ONNX sessions.
# It will now be attached to app.state
# For now, routers.predict handles its own loading TEMPORARILY.
# TODO: Centralize model loading and access.
onnx_sessions = {}  # This will be populated and then attached to app.state


@app.on_event("startup")
async def load_models():
    """
    Load ONNX models during application startup.
    These sessions can then be used by different parts of the application,
    like the routers, via app.state.
    """
    print(f"Attempting to load models from: {os.path.abspath(settings.MODEL_DIR)}")
    # Create the directory if it doesn't exist
    os.makedirs(settings.MODEL_DIR, exist_ok=True)

    active_model_paths = {
        k: v
        for k, v in MODEL_PATHS.items()
        if settings.VIT_SMALL_MODEL_NAME not in v
        and settings.VIT_LARGE_MODEL_NAME not in v
    }  # Filter out ViT models if not defined in settings, to avoid errors for now

    for model_name, model_path in active_model_paths.items():
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

            # Inspect and print model input details
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
            onnx_sessions[model_name] = None

    if not onnx_sessions or all(session is None for session in onnx_sessions.values()):
        print(
            "CRITICAL WARNING: No ONNX models were successfully loaded. Predictions will fail."
        )
    elif any(session is None for session in onnx_sessions.values()):
        print(
            "Warning: Some ONNX models could not be loaded. Check paths and file integrity."
        )
    else:
        print("All configured ONNX models loaded successfully.")

    # IMPORTANT: Attach the loaded onnx_sessions to the app's state
    app.state.onnx_sessions = onnx_sessions


# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Include the router for prediction endpoints
app.include_router(predict_router.router)


if __name__ == "__main__":
    import uvicorn

    print(
        f"Starting Uvicorn server for FastAPI app (app.main) on {settings.HOST}:{settings.PORT}..."
    )
    uvicorn.run(
        "app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG
    )
