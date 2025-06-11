# Deepfake Audio Detector API

## Project Description

This project provides a FastAPI-based API for detecting deepfake audio. It uses PyTorch models (CNN and Vision Transformer architectures) to classify audio segments as either "real" or "fake". The application processes uploaded audio files, splits them into chunks, and performs inference on each chunk.

## Directory Structure

Here's an overview of the key directories and files within the `app/` directory:

-   `app/main.py`: The main FastAPI application entry point, handles model loading and request routing.
-   `app/config.py`: Contains application settings, including model paths, audio processing parameters, and server configuration.
-   `app/model_definitions.py`: Defines the PyTorch model architectures (e.g., `CNN_Audio`, `ViT_Audio`).
-   `app/models/`: Directory where trained `.pth` model files are stored.
-   `app/routers/predict.py`: Contains the API endpoint logic for audio prediction.
-   `app/audio_processing/utils.py`: Utility functions for audio processing tasks like chunking.
-   `app/static/`: Static files (CSS, JavaScript) for the web interface.
-   `app/templates/`: HTML templates for the web interface.
-   `requirements.txt`: Lists Python dependencies.
-   `Dockerfile`: (If present) Instructions for building a Docker image for the application.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you have an `environment.yml` for Conda, provide instructions for that as well or instead.)*

## Configuration

Application behavior can be configured via `app/config.py` or by setting environment variables. Key configurations include:

-   `MODEL_DIR`: Directory where models are stored (default: `app/models/`).
-   `CNN_SMALL_MODEL_NAME`, `CNN_LARGE_MODEL_NAME`, `VIT_SMALL_MODEL_NAME`, `VIT_LARGE_MODEL_NAME`: Filenames of the respective models.
-   `TARGET_SAMPLE_RATE`, `CHUNK_DURATION_SECONDS`, `N_MELS`, `SPECTROGRAM_WIDTH`: Audio processing parameters.
-   `HOST`, `PORT`, `DEBUG`: Server settings.

Environment variables (defined in a `.env` file or set in the system) will override defaults in `config.py`. Example `.env` file:
```
DEBUG=False
PORT=8000
MODEL_DIR="app/models/"
CNN_SMALL_MODEL_NAME="best_model_CNN_Small_cnn_3s_dataset_102208.pth"
# Add other model names as needed
```

## Running the Application

To run the FastAPI application locally:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

-   `--reload`: Enables auto-reload during development. Omit for production.
-   `--host 0.0.0.0`: Makes the server accessible from other devices on the network.
-   `--port 8000`: Specifies the port.

You should see output indicating the Uvicorn server is running, and the application will be accessible at `http://localhost:8000` (or `http://<your_server_ip>:8000`).

## API Endpoints

### POST /predict_audio

Predicts if an uploaded audio file is real or a deepfake.

-   **Request:** `multipart/form-data`
    -   `audio_file`: The audio file to analyze (e.g., WAV, MP3, FLAC).
    -   `model_name` (query parameter, optional): The name of the model to use for prediction. Defaults to `cnn_small`.
        Available models are listed in the "Available Models" section.

-   **Response:** `application/json`
    A list of prediction results, one for each processed audio chunk.
    ```json
    [
        {
            "filename": "example.wav",
            "chunk_index": 0,
            "num_chunks": 2,
            "prediction": "fake",
            "confidence": 0.85,
            "model_used": "cnn_small",
            "raw_model_output": [[-1.75, 1.75]]
        },
        {
            "filename": "example.wav",
            "chunk_index": 1,
            "num_chunks": 2,
            "prediction": "real",
            "confidence": 0.65,
            "model_used": "cnn_small",
            "raw_model_output": [[0.65, -0.65]]
        }
    ]
    ```

-   **Example using `curl`:**
    ```bash
    curl -X POST -F "audio_file=@/path/to/your/audio.wav" "http://localhost:8000/predict_audio?model_name=cnn_small"
    ```

## Available Models

You can specify the `model_name` query parameter in the `/predict_audio` endpoint. The following models are configured:

-   `cnn_small`: Small CNN model.
-   `cnn_large`: Large CNN model.
-   `vit_small`: Small Vision Transformer model. *(Note: The ViT model integration uses a placeholder architecture. Its performance depends on the compatibility of the saved `.pth` file with this placeholder. Actual ViT architecture details might be needed for optimal performance.)*
-   `vit_large`: Large Vision Transformer model. *(Note: Same as above regarding placeholder architecture.)*

The specific `.pth` filenames for these models are set in `app/config.py`.

## Deployment (Conceptual)

### Docker

If a `Dockerfile` is provided in the project root, you can build and run the application as a Docker container:

1.  **Build the Docker image:**
    ```bash
    docker build -t deepfake-audio-detector .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -d -p 8000:8000 --name deepfake-app deepfake-audio-detector
    ```
    *(Adjust port mapping and container name as needed. You might need to pass environment variables for configuration using `-e` flags or an env file.)*

### General FastAPI Deployment

For production, it's common to run FastAPI applications with Gunicorn and Uvicorn workers. A reverse proxy like Nginx or Traefik can be used for SSL termination, load balancing, and serving static files.

Example (running with Gunicorn):
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app -b 0.0.0.0:8000
```
-   `-w 4`: Number of worker processes. Adjust based on your server's CPU cores.
-   `-k uvicorn.workers.UvicornWorker`: Specifies Uvicorn for handling requests.

Refer to the [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/) for more detailed information.

## Quick Test / Example

1.  Ensure the application is running (see "Running the Application").
2.  Open your web browser and navigate to `http://localhost:8000`. You should see a simple interface to upload an audio file.
3.  Alternatively, use the `curl` command provided in the "API Endpoints" section with a sample audio file.
    - Create a dummy WAV file or use an existing one.
    - Execute:
      ```bash
      curl -X POST -F "audio_file=@/path/to/your/sample.wav" "http://localhost:8000/predict_audio?model_name=cnn_small"
      ```
    - Check the JSON response for predictions.

```
