# app/routers/predict.py

from fastapi import APIRouter, File, UploadFile, HTTPException, status, Request
from typing import Dict, Any
import onnxruntime as rt
import io
import numpy as np
import librosa # New import for audio processing
import soundfile as sf # New import for reading audio files from bytes
from PIL import Image # New import for resizing spectrograms

# Import settings from app.config
from ..config import settings

router = APIRouter()

def process_audio_for_model(audio_data: bytes) -> np.ndarray:
    """
    Processes raw audio bytes into the Mel spectrogram format expected by the ONNX model.
    Uses settings from app.config for audio processing parameters.
    The resulting spectrogram is resized to (SPECTROGRAM_WIDTH, N_MELS) if needed,
    and then prepared with batch and channel dimensions for ONNX inference.
    """
    print("Starting audio processing...")
    try:
        # Load audio from bytes in memory
        # sf.read returns audio data as a NumPy array and the sample rate (sr)
        audio, sr = sf.read(io.BytesIO(audio_data))
        print(f"Original audio loaded: Sample Rate={sr}, Samples={len(audio)}")

        # Resample if the audio's sample rate doesn't match the target
        if sr != settings.TARGET_SAMPLE_RATE:
            print(f"Resampling audio from {sr} Hz to {settings.TARGET_SAMPLE_RATE} Hz...")
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=settings.TARGET_SAMPLE_RATE)
            sr = settings.TARGET_SAMPLE_RATE
            print(f"Audio resampled: New Samples={len(audio)}")

        # Pad or trim audio to a consistent fixed length (e.g., 3 seconds)
        # This is critical for models expecting fixed-size inputs.
        target_length_samples = int(settings.TARGET_SAMPLE_RATE * settings.CHUNK_DURATION_SECONDS)
        
        if len(audio) < target_length_samples:
            # Pad with zeros at the end if the audio is shorter than the target duration
            print(f"Padding audio from {len(audio)} to {target_length_samples} samples.")
            audio = np.pad(audio, (0, target_length_samples - len(audio)), 'constant')
        elif len(audio) > target_length_samples:
            # Trim from the end if the audio is longer than the target duration
            print(f"Trimming audio from {len(audio)} to {target_length_samples} samples.")
            audio = audio[:target_length_samples]
        print(f"Audio length after padding/trimming: {len(audio)} samples.")

        # Compute Mel spectrogram
        # n_mels: number of Mel bands (height of the spectrogram)
        # n_fft: length of the FFT window
        # hop_length: number of samples between successive frames
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=settings.N_FFT,
            hop_length=settings.HOP_LENGTH,
            n_mels=settings.N_MELS
        )
        print(f"Mel spectrogram computed. Shape: {mel_spectrogram.shape}") # (n_mels, num_time_frames)

        # Convert amplitude to decibels (dB)
        # ref: reference amplitude for dB calculation (np.max ensures the max is 0dB)
        # amin: minimum threshold for audio magnitude
        # top_db: upper bound on the dynamic range (e.g., 80dB below the peak)
        mel_spectrogram_db = librosa.power_to_db(
            mel_spectrogram,
            ref=np.max,
            amin=1e-10, # A small epsilon to avoid log(0)
            top_db=settings.TOP_DB
        )
        print(f"Mel spectrogram converted to dB. Shape: {mel_spectrogram_db.shape}")

        # Normalize the spectrogram to a [0, 1] range for robust resizing with PIL
        min_val = mel_spectrogram_db.min()
        max_val = mel_spectrogram_db.max()
        if (max_val - min_val) > 1e-6: # Prevent division by zero if spectrogram is flat
            normalized_spectrogram_for_resize = (mel_spectrogram_db - min_val) / (max_val - min_val)
        else:
            normalized_spectrogram_for_resize = np.zeros_like(mel_spectrogram_db)
        
        # Check if resizing is necessary to match the model's expected input dimensions
        # The model expects a (N_MELS, SPECTROGRAM_WIDTH) input (height, width)
        current_height, current_width = normalized_spectrogram_for_resize.shape
        target_height = settings.N_MELS
        target_width = settings.SPECTROGRAM_WIDTH

        if current_width != target_width or current_height != target_height:
            print(f"Resizing spectrogram from ({current_height}, {current_width}) to ({target_height}, {target_width}).")
            # Convert numpy array to PIL Image. 'L' mode for grayscale.
            # Scale to 0-255 for standard image processing if it's float32 and not already in that range
            img = Image.fromarray((normalized_spectrogram_for_resize * 255).astype(np.uint8)).convert('L')
            
            # Resize. PIL's resize method expects (width, height)
            resized_img = img.resize((target_width, target_height), Image.BICUBIC)
            
            # Convert back to numpy array and scale back to [0, 1]
            processed_input_normalized_0_1 = np.array(resized_img).astype(np.float32) / 255.0
            print(f"Spectrogram resized. New shape: {processed_input_normalized_0_1.shape}")

            # Remap the normalized [0, 1] values back to the original dB range
            # This assumes the model expects inputs in the dB range used for training.
            processed_input = processed_input_normalized_0_1 * (settings.TOP_DB - settings.MIN_DB_LEVEL) + settings.MIN_DB_LEVEL
        else:
            print("Spectrogram dimensions match model input; no resizing needed.")
            # If no resizing, use the original dB spectrogram directly
            processed_input = mel_spectrogram_db

        # Add batch and channel dimensions for ONNX model input
        # Common shapes for CNNs: (batch_size, channels, height, width)
        # For a single grayscale spectrogram, channels = 1.
        # So, the shape becomes (1, 1, N_MELS, SPECTROGRAM_WIDTH)
        input_tensor = processed_input[np.newaxis, np.newaxis, :, :]
        print(f"Final input tensor shape for ONNX model: {input_tensor.shape}")
        
        # Ensure the final tensor is float32, which is typically required by ONNX models
        return input_tensor.astype(np.float32)

    except sf.LibsndfileError as lse:
        print(f"Error during audio file reading: {lse}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading audio file: {lse}. Ensure it's a valid audio format like WAV, FLAC, etc."
        )
    except Exception as e:
        print(f"An unexpected error occurred during audio processing: {e}")
        import traceback
        print(traceback.format_exc()) # Print full traceback for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during audio processing: {e}"
        )

# The rest of the router code remains the same
@router.post("/predict_audio")
async def predict_audio(
    request: Request, # Inject the Request object
    audio_file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Receives an audio file, processes it, and makes a prediction using the ONNX model.
    """
    # Access the onnx_sessions from the application state
    onnx_sessions: Dict[str, rt.InferenceSession] = request.app.state.onnx_sessions

    if not onnx_sessions or all(session is None for session in onnx_sessions.values()):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ONNX models not loaded or available. Server configuration issue. Please check server logs."
        )

    audio_bytes = await audio_file.read()

    # Get a specific model session, e.g., 'cnn_small' or 'cnn_large'
    # You'll need to decide which model to use, or make it configurable via query parameter.
    model_name = "cnn_small" # <--- IMPORTANT: Change this or make it dynamic if you have multiple models
    session = onnx_sessions.get(model_name)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found or not loaded. Check your model files and naming in config.py and server logs."
        )

    try:
        # Preprocess the audio data using the updated function
        input_data = process_audio_for_model(audio_bytes)

        # Get input and output names from the ONNX model
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        outputs = session.run([output_name], {input_name: input_data})
        prediction = outputs[0]
        
        # Post-process the prediction (e.g., softmax, thresholding)
        # Assuming your model outputs probabilities or scores for classes
        # This will depend on what your model outputs.
        
        # Determine the predicted label and confidence
        # Using a simple threshold (0.5) for binary classification
        if prediction.shape[-1] > 1: # If output is multi-class probability (e.g., [prob_real, prob_fake])
            predicted_index = np.argmax(prediction[0]) # Get the index of the highest probability
            confidence = float(prediction[0][predicted_index])
            result_label = settings.LABELS.get(predicted_index, "unknown")
        else: # If output is a single score/probability (e.g., for one class, like deepfake score)
              # Assuming higher value means 'real' or it's a binary score for one class
              # Let's stick to the previous thresholding logic for simplicity and alignment with `prediction[0][0] > 0.5`
            confidence = float(prediction[0][0]) # This is the score for whatever the first output represents
            # Determine label based on confidence and REAL_LABEL_INDEX / FAKE_LABEL_INDEX
            if confidence > 0.5: # If higher confidence means 'real' as in the comment
                result_label = settings.LABELS.get(settings.REAL_LABEL_INDEX, "real")
            else:
                result_label = settings.LABELS.get(settings.FAKE_LABEL_INDEX, "deepfake")

        return {
            "filename": audio_file.filename,
            "prediction": result_label,
            "confidence": confidence,
            "model_used": model_name,
            "raw_model_output": prediction.tolist() # Include raw output for debugging/info
        }

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        # Log the full traceback for debugging purposes
        import traceback
        print(f"Prediction failed with an unexpected error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}. Check server logs for details."
        )

