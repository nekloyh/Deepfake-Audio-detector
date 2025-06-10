import os
import io
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import onnxruntime
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import librosa
import soundfile as sf # For robust audio loading

# --- Configuration ---
N_MELS = 128
TARGET_SPEC_WIDTH = 256
SAMPLE_RATE = 16000  # Target sample rate for all audio
N_FFT = 2048 # Original value
HOP_LENGTH = 512 # Original value
MIN_DB_LEVEL = -80.0 # Used for padding and MelSpectrogram normalization

LABELS = {0: "real", 1: "fake"}

# --- FastAPI App Initialization ---
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")
templates = Jinja2Templates(directory="webapp/templates")

# --- ONNX Model Loading ---
MODEL_DIR = "F:\\Deepfake-Audio-Detector\\models\\.onnx" # Assuming onnx_models is one level up from webapp
MODEL_PATHS = {
    "cnn_small": os.path.join(MODEL_DIR, "CNN_Small.onnx"),
    "cnn_large": os.path.join(MODEL_DIR, "CNN_Large.onnx"),
    "vit_small": os.path.join(MODEL_DIR, "ViT_Small.onnx"),
    "vit_large": os.path.join(MODEL_DIR, "ViT_Large.onnx"),
}
onnx_sessions = {}

@app.on_event("startup")
async def load_models():
    print(f"Attempting to load models from: {os.path.abspath(MODEL_DIR)}")
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            onnx_sessions[model_name] = None # Or raise an error
            continue
        try:
            onnx_sessions[model_name] = onnxruntime.InferenceSession(model_path)
            print(f"Successfully loaded ONNX model: {model_name} from {model_path}")
        except Exception as e:
            print(f"Error loading ONNX model {model_name} from {model_path}: {e}")
            onnx_sessions[model_name] = None # Or raise an error
    # Check if any model failed to load
    if any(session is None for session in onnx_sessions.values()):
        print("Warning: Some ONNX models could not be loaded. Check paths and file integrity.")


# --- Helper Functions ---
def preprocess_audio_to_spectrogram(waveform: torch.Tensor, input_sr: int) -> np.ndarray:
    """Converts waveform to a standardized mel spectrogram numpy array."""
    # Ensure waveform is a PyTorch tensor
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)

    # 1. Resample if necessary
    if input_sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=input_sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # 2. Convert to mono if stereo
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    elif waveform.ndim == 1: # Ensure it has a channel dimension
        waveform = waveform.unsqueeze(0)


    # 3. Mel Spectrogram Calculation
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0, # Power = 2 for power spectrogram, 1 for magnitude
        norm='slaney', # Slaney norm is common
        mel_scale="htk" # or "slaney"
    )
    mel_spec = mel_spectrogram_transform(waveform)

    # 4. Amplitude to dB
    amplitude_to_db_transform = T.AmplitudeToDB(stype="power", top_db=80) # top_db=80 is common
    mel_spec_db = amplitude_to_db_transform(mel_spec)
    
    # Normalize to [0, 1] (optional, but can help if models expect it)
    # Or normalize based on training data stats if available
    # For now, let's use min-max based on typical dB range from top_db
    # mel_spec_db = (mel_spec_db - MIN_DB_LEVEL) / (0 - MIN_DB_LEVEL) # Assuming max dB is 0 after top_db

    # 5. Pad or Truncate width
    current_width = mel_spec_db.shape[-1]
    if current_width < TARGET_SPEC_WIDTH:
        pad_amount = TARGET_SPEC_WIDTH - current_width
        # Pad with MIN_DB_LEVEL, consistent with how silence or background is treated.
        mel_spec_db = F.pad(mel_spec_db, (0, pad_amount), "constant", value=MIN_DB_LEVEL)
    elif current_width > TARGET_SPEC_WIDTH:
        # Truncate from the center or start. Let's truncate from start.
        mel_spec_db = mel_spec_db[..., :TARGET_SPEC_WIDTH]

    # 6. Ensure correct shape: (1, N_MELS, TARGET_SPEC_WIDTH) for ONNX model
    # Current shape is likely (1, N_MELS, TARGET_SPEC_WIDTH) if input was mono and processed correctly
    # If it was (N_MELS, TARGET_SPEC_WIDTH), it would need .unsqueeze(0)
    # If it somehow became (1, 1, N_MELS, TARGET_SPEC_WIDTH), it would need .squeeze(0)
    # Let's ensure it's (1, N_MELS, TARGET_SPEC_WIDTH)
    if mel_spec_db.ndim == 2: # (N_MELS, TARGET_SPEC_WIDTH)
        mel_spec_db = mel_spec_db.unsqueeze(0) # Add channel dim -> (1, N_MELS, TARGET_SPEC_WIDTH)
    
    # The ONNX models expect (batch_size, channels, height, width) = (1, 1, N_MELS, TARGET_SPEC_WIDTH)
    # So, we need an additional channel dimension if it's not there from mono processing.
    # Most torchaudio transforms for spectrograms on mono audio return (mel_bins, time_frames).
    # After unsqueeze(0) it becomes (1, mel_bins, time_frames) which is (channels, height, width)
    # This should be what the model expects if "channels" is 1.
    # If the model truly expects (batch, 1, N_MELS, TARGET_SPEC_WIDTH), an additional unsqueeze(0) might be needed
    # Let's assume the model input name 'input' maps to (batch_size, feature_maps/channels, height, width)
    # and our current mel_spec_db is (1, N_MELS, TARGET_SPEC_WIDTH) which is (C, H, W)
    # So, we add a batch dimension:
    mel_spec_db = mel_spec_db.unsqueeze(0) # Add batch dim -> (1, 1, N_MELS, TARGET_SPEC_WIDTH)


    return mel_spec_db.numpy()


# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    contents = await file.read()
    audio_bytes_io = io.BytesIO(contents)
    waveform = None
    sr = None

    try:
        # Try Librosa first (handles more formats gracefully)
        waveform_np, sr_librosa = librosa.load(audio_bytes_io, sr=SAMPLE_RATE, mono=True)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # Add channel dimension
        sr = sr_librosa
        print(f"Successfully loaded audio with Librosa. SR: {sr}, Shape: {waveform.shape}")
    except Exception as e_librosa:
        print(f"Librosa loading failed: {e_librosa}. Trying Torchaudio.")
        audio_bytes_io.seek(0) # Reset buffer for next read
        try:
            waveform_ta, sr_ta = torchaudio.load(audio_bytes_io)
            # Ensure waveform is correctly processed by torchaudio (resample, mono)
            if sr_ta != SAMPLE_RATE:
                resampler = T.Resample(orig_freq=sr_ta, new_freq=SAMPLE_RATE)
                waveform_ta = resampler(waveform_ta)
            if waveform_ta.shape[0] > 1: # Stereo to mono
                waveform_ta = torch.mean(waveform_ta, dim=0, keepdim=True)
            waveform = waveform_ta
            sr = SAMPLE_RATE # sr is now target sample rate
            print(f"Successfully loaded audio with Torchaudio. SR: {sr}, Shape: {waveform.shape}")
        except Exception as e_torchaudio:
            print(f"Torchaudio loading also failed: {e_torchaudio}")
            raise HTTPException(status_code=400, detail=f"Invalid audio file or format. Librosa: {e_librosa}, Torchaudio: {e_torchaudio}")

    if waveform is None or sr is None:
        raise HTTPException(status_code=400, detail="Audio could not be loaded or processed.")

    try:
        # Preprocess the audio to get the mel spectrogram
        # The waveform should be (channels, time) for preprocess_audio_to_spectrogram
        # Librosa load with mono=True gives (time,), so unsqueezed to (1, time)
        # Torchaudio load gives (channels, time)
        if waveform.ndim == 1: # If it's somehow still 1D
             waveform = waveform.unsqueeze(0)

        processed_spectrogram_numpy = preprocess_audio_to_spectrogram(waveform, sr)
        # Expected shape (1, 1, N_MELS, TARGET_SPEC_WIDTH)
        print(f"Processed spectrogram shape: {processed_spectrogram_numpy.shape}")


    except Exception as e_preprocess:
        print(f"Error during preprocessing: {e_preprocess}")
        raise HTTPException(status_code=500, detail=f"Error during audio preprocessing: {e_preprocess}")

    predictions = {}
    for model_name, session in onnx_sessions.items():
        if session:
            try:
                ort_inputs = {'input': processed_spectrogram_numpy}
                ort_outs = session.run(None, ort_inputs) # Output is usually a list of numpy arrays

                # Assuming output is logits, apply softmax
                logits = torch.tensor(ort_outs[0])
                probabilities = torch.softmax(logits, dim=-1).squeeze().numpy() # Squeeze to remove batch dim for single item

                predicted_index = int(np.argmax(probabilities))
                predicted_label = LABELS.get(predicted_index, "Unknown")
                
                predictions[model_name] = {
                    "label": predicted_label,
                    "score_real": float(probabilities[0]), # Assuming class 0 is real
                    "score_fake": float(probabilities[1])  # Assuming class 1 is fake
                }
            except Exception as e_inference:
                print(f"Error during inference for {model_name}: {e_inference}")
                predictions[model_name] = {"label": "Error", "detail": f"Inference error: {e_inference}"}
        else:
            predictions[model_name] = {"label": "Error", "detail": "Model not loaded"}

    return JSONResponse(content={"filename": file.filename, "predictions": predictions})


if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for FastAPI app...")
    # Ensure uvicorn runs main:app from the root directory perspective if webapp is a module
    # or run this script directly from within the webapp directory.
    # For simplicity here, assuming running from webapp directory: uvicorn main:app --reload
    # However, Docker will handle this.
    uvicorn.run(app, host="0.0.0.0", port=8000)
