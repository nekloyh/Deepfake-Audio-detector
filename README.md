# Deepfake Audio Detection API

This application provides a FastAPI backend and a simple web interface to detect deepfake audio. It processes uploaded audio files, segments them, converts them to Mel-spectrograms, and uses ONNX models for inference.

## Features

*   Deepfake audio detection from uploaded audio files (WAV, MP3, FLAC, etc.).
*   Automatic segmentation of audio files into 3-second chunks.
*   Preprocessing of audio chunks into Mel-spectrograms suitable for model input.
*   Support for multiple ONNX models (e.g., CNN Small/Large, ViT Small/Large).
*   FastAPI backend providing a RESTful API for predictions.
*   Simple web interface for easy audio file uploads and viewing of detection results.
*   Docker support for straightforward containerized deployment.
*   Configuration via environment variables for flexibility.

## Project Structure

```
.
├── app/                  # Main application folder
│   ├── main.py           # FastAPI app initialization, startup events
│   ├── config.py         # Application settings (Pydantic BaseSettings)
│   ├── audio_processing/ # Audio loading, chunking, spectrogram generation
│   │   └── utils.py
│   ├── models_onnx/      # ONNX model files (e.g., CNN_Large.onnx)
│   ├── routers/          # API endpoint definitions
│   │   └── predict.py
│   ├── static/           # CSS, JavaScript files
│   │   ├── style.css
│   │   └── script.js
│   └── templates/        # HTML templates (Jinja2)
│       └── index.html
├── Dockerfile            # For building the Docker image
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables
└── README.md             # This file
```

## Prerequisites

*   Python 3.9 or higher.
*   Docker (optional, for containerized deployment).
*   Access to a terminal or command prompt.
*   Git (for cloning the repository).

## Setup and Installation (Local)

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> 
    ```
    (Replace `<repository_url>` with the actual URL of your Git repository)

2.  **Navigate to the project directory:**
    ```bash
    cd <project_directory>
    ```

3.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up environment variables:**
    *   Copy `.env.example` to a new file named `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Review and update variables in the `.env` file if needed. The `MODEL_DIR` defaults to `app/models_onnx/`. The server `PORT` defaults to 8000.

## Running Locally

1.  **Place ONNX Models:** Ensure your `.onnx` model files are located in the directory specified by the `MODEL_DIR` environment variable (default is `app/models_onnx/`). The application is configured to look for `CNN_Small.onnx` and `CNN_Large.onnx`. If you have ViT models or other models, ensure they are present and update `app/config.py` if necessary to include their filenames.

2.  **Start the FastAPI application:**
    Use Uvicorn to run the application. The host and port can be configured in your `.env` file or default to `0.0.0.0` and `8000`.
    ```bash
    uvicorn app.main:app --reload
    ```
    The `--reload` flag enables auto-reloading for development. For production, remove this flag.
    If your `.env` file specifies a different `HOST` or `PORT`, Uvicorn will use those values when started as above (as `app.main` loads them).

3.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8000` (or the host and port you configured).

## Building and Running with Docker

1.  **Ensure Docker is running.**

2.  **Build the Docker image:**
    From the project root directory (where the `Dockerfile` is located):
    ```bash
    docker build -t deepfake-audio-detector .
    ```

3.  **Run the Docker container:**
    This command runs the container and maps port 8000 of the container to port 8000 on your host machine.
    ```bash
    docker run -p 8000:8000 deepfake-audio-detector
    ```
    The application inside the container will use the `MODEL_DIR` as configured (default `app/models_onnx`), which are copied into the image during the build.

4.  **Using Custom Models or Overriding Bundled Models with Docker Volumes:**
    If you want to use models from your local machine without rebuilding the image, or if `MODEL_DIR` in your `.env` (passed to Docker) points to a path that should be mounted:
    ```bash
    docker run -p 8000:8000 \
           -v /path/to/your/local/models_onnx_folder:/app/app/models_onnx \
           -e MODEL_DIR="app/models_onnx" \
           deepfake-audio-detector
    ```
    *   Replace `/path/to/your/local/models_onnx_folder` with the actual path to your models on your host machine.
    *   The `-v` flag mounts your local models directory to `/app/app/models_onnx` inside the container.
    *   The `-e MODEL_DIR="app/models_onnx"` ensures the application inside Docker uses this internal path. This should match the `COPY` destination in the Dockerfile for models if you intend to override them.

## API Usage

The application provides a REST API for programmatic access.

*   **Endpoint:** `POST /predict/`
*   **Request Type:** `multipart/form-data`
*   **Form Field:** `file` (containing the audio file)
*   **Accepted Audio Formats:** Any format supported by Librosa (e.g., WAV, MP3, FLAC).

*   **Example with `curl`:**
    ```bash
    curl -X POST -F "file=@/path/to/your/audio.wav" http://localhost:8000/predict/
    ```

*   **JSON Response Structure:**
    The API returns a JSON object containing the filename and predictions from each loaded model for each audio chunk.
    ```json
    {
      "filename": "your_audio.wav",
      "predictions": {
        "cnn_large": [
          {
            "chunk_index": 0,
            "label": "fake",  // or "real"
            "score_real": 0.123,
            "score_fake": 0.877
          },
          {
            "chunk_index": 1,
            "label": "real",
            "score_real": 0.950,
            "score_fake": 0.050
          }
          // ... more chunks for cnn_large model
        ],
        "cnn_small": [
          // ... predictions for cnn_small model
        ]
        // ... other models if loaded ...
      }
    }
    ```

## Web Interface

A simple web interface is provided for manual testing and demonstration.

1.  Navigate to the root URL of the application (e.g., `http://localhost:8000`).
2.  Use the file input field to select an audio file from your computer.
3.  Click the "Detect Deepfake" button.
4.  The prediction results will be displayed on the page in JSON format.

## ONNX Models

*   The application uses ONNX (Open Neural Network Exchange) models for inference.
*   Place your `.onnx` model files in the directory specified by the `MODEL_DIR` environment variable (defaults to `app/models_onnx/` within the project structure).
*   **Input Shape:** Models are expected to be compatible with Mel-spectrogram inputs derived from 3-second audio chunks. The required input shape for the models is `(1, 1, N_MELS, SPECTROGRAM_WIDTH)`, which defaults to `(1, 1, 224, 224)` based on current `app/config.py` settings.
*   **Model Configuration:** The application attempts to load models listed in `MODEL_PATHS` in `app/config.py`. By default, these are `CNN_Small.onnx` and `CNN_Large.onnx`. (Support for `ViT_Small.onnx` and `ViT_Large.onnx` is commented out but can be enabled by uncommenting in `app/config.py` and ensuring the model files exist).
*   **Input Tensor Name:** Models are assumed to use `'input'` as their input tensor name. This is verified during startup and logged to the console.

## Environment Variables

The application can be configured using environment variables. These are typically defined in a `.env` file in the project root for local development or set directly in the deployment environment. Refer to `.env.example` for a comprehensive list.

Key variables:
*   `HOST`: The IP address the server will listen on (default: `"0.0.0.0"`).
*   `PORT`: The port the server will listen on (default: `8000`).
*   `MODEL_DIR`: Path to the directory containing your ONNX models (default: `"app/models_onnx"`).
*   `DEBUG`: Set to `True` to enable debug mode features, such as Uvicorn's auto-reload (default: `False`).
*   `APP_NAME`: Name of the application (default: `"Deepfake Audio Detector"`).

Other parameters related to audio processing (e.g., `TARGET_SAMPLE_RATE`, `N_MELS`) are currently set as defaults in `app/config.py` but can be modified there or made configurable via environment variables if needed.
