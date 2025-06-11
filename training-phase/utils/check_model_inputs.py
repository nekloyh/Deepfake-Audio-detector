import onnxruntime
import os
from app.config import settings

print("--- Script Start ---")

MODEL_DIR = settings.MODEL_DIR
# Let's test with only one model first to simplify
MODEL_NAMES = {
    "cnn_small": settings.CNN_SMALL_MODEL_NAME,
    # "cnn_large": settings.CNN_LARGE_MODEL_NAME,
}

print(f"Model directory used: {os.path.abspath(MODEL_DIR)}")
print(f"Attempting to load models: {list(MODEL_NAMES.keys())}")

for model_key, model_file_name in MODEL_NAMES.items():
    model_path = os.path.join(MODEL_DIR, model_file_name)
    print(f"\n--- Processing model: {model_key} from {model_path} ---")

    print(f"Checking if model file exists at {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        continue
    print("Model file exists.")

    try:
        print(f"Attempting to load ONNX session for {model_key}...")
        session = onnxruntime.InferenceSession(model_path)
        print(f"ONNX session loaded for {model_key}.")

        inputs = session.get_inputs()
        print(f"Retrieved inputs for {model_key}.")

        if inputs:
            print(f"Model '{model_key}' - Input Details:")
            for i, input_meta in enumerate(inputs):
                print(
                    f"  Input {i}: Name='{input_meta.name}', Shape={input_meta.shape}, Type={input_meta.type}"
                )
        else:
            print(f"Model '{model_key}': No inputs found in the model metadata.")

    except Exception as e:
        print(f"Error loading or inspecting model {model_key}: {e}")
        import traceback

        traceback.print_exc()

print("\n--- Configuration Settings for Comparison ---")
print(f"settings.N_MELS: {settings.N_MELS}")
print(f"settings.SPECTROGRAM_WIDTH: {settings.SPECTROGRAM_WIDTH}")
print(
    f"Expected input shape based on config: (1, 1, {settings.N_MELS}, {settings.SPECTROGRAM_WIDTH})"
)
print(f"Expected data type: float32")
print("--- Script End ---")
