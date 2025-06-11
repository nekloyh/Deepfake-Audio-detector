## Project Title: Deepfake Audio Detector API

### Overview
Briefly describe the project: A FastAPI application for detecting deepfake audio using PyTorch models. It processes uploaded audio files, splits them into chunks, and predicts if each chunk is real or fake.

### Features
- FastAPI backend
- PyTorch model inference for CNN-based audio classification
- Audio processing (resampling, Mel spectrogram, normalization)
- Audio chunking for handling variable-length inputs
- Configurable via `.env` file and `app/config.py`
- Docker support for deployment

### Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Configuration:**
    Create a `.env` file in the project root directory. You can copy `.env.example` if provided, or create it manually.
    Currently, most configurations are in `app/config.py`. Key settings include:
    - `MODEL_DIR`: Path to the directory containing `.pth` model files. (Default: `training-phase/models/.pth/`)
    - `CNN_SMALL_MODEL_NAME`, `CNN_LARGE_MODEL_NAME`: Specific model filenames.
    - `TARGET_SAMPLE_RATE`, `CHUNK_DURATION_SECONDS`, etc., for audio processing.

    Ensure your PyTorch models (e.g., `best_model_CNN_Small_cnn_3s_dataset_102208.pth`) are located in the directory specified by `MODEL_DIR` (relative to the project root).

### Running the Application

1.  **Start the FastAPI server:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    - `--reload` enables auto-reloading for development. Omit for production.

2.  **Access the API documentation:**
    Open your browser and go to `http://localhost:8000/docs` for Swagger UI or `http://localhost:8000/redoc` for ReDoc.

### Using the API

The primary endpoint for prediction is `/predict_audio`.

**Endpoint:** `POST /predict_audio`
**Description:** Uploads an audio file for deepfake detection.
**Query Parameters:**
  - `model_name` (str, optional): The name of the model to use (e.g., "cnn_small", "cnn_large"). Defaults to "cnn_small".
**Request Body:**
  - `audio_file`: The audio file to be analyzed (e.g., WAV, MP3, FLAC).
**Example using `curl`:**
```bash
curl -X POST -F "audio_file=@/path/to/your/audiofile.wav" "http://localhost:8000/predict_audio?model_name=cnn_small"
```
**Expected Response:**
A JSON list of predictions, one for each 3-second chunk of the audio:
```json
[
  {
    "filename": "audiofile.wav",
    "prediction": "fake", // or "real"
    "confidence": 0.85,
    "model_used": "cnn_small",
    "raw_model_output": [[0.15, 0.85]], // Raw output from the model
    "chunk_index": 0,
    "num_chunks": 2
  },
  {
    "filename": "audiofile.wav",
    "prediction": "real",
    "confidence": 0.70,
    "model_used": "cnn_small",
    "raw_model_output": [[0.70, 0.30]],
    "chunk_index": 1,
    "num_chunks": 2
  }
]
```

### Deployment

**Using Docker:**
A `Dockerfile` is provided in the project.

1.  **Build the Docker image:**
    ```bash
    docker build -t deepfake-audio-detector .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 deepfake-audio-detector
    ```
    Ensure that the `MODEL_DIR` path and model files are correctly accessible within the Docker container context. You might need to adjust `COPY` commands in the `Dockerfile` or use Docker volumes to mount the models. The current `Dockerfile` would need to be inspected to confirm how models are handled.

### Quick Testing
1.  Ensure the server is running.
2.  Use the Swagger UI at `http://localhost:8000/docs` to upload an audio file and test the `/predict_audio` endpoint interactively.
3.  Alternatively, use a `curl` command as shown above with a sample audio file.
4.  Verify that the models load correctly on startup by checking the console output when Uvicorn starts. Look for messages like "Successfully loaded PyTorch model: cnn_small".
5.  Test with both short (< 3 seconds) and long (> 3 seconds) audio files to ensure chunking works as expected.

### Project Structure
- `app/`: Main application folder.
  - `main.py`: FastAPI app initialization, model loading.
  - `config.py`: Application settings.
  - `model_definitions.py`: PyTorch model class definitions.
  - `routers/predict.py`: API endpoint for predictions.
  - `audio_processing/utils.py`: Audio processing utilities.
  - `static/`, `templates/`: For a simple web interface.
- `training-phase/models/.pth/`: Default directory for PyTorch model files.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: For containerization.
- `README.md`: This file.

### Notes
- This project currently ignores ONNX models and focuses on PyTorch (`.pth`) models.
- The audio processing parameters (sample rate, FFT settings, etc.) are defined in `app/config.py` and should match the parameters used during model training.
