# Deepfake Audio Detection - User Guide

## 1. Project Overview

This project is a Deepfake Audio Detection system that can identify whether an audio recording is real or synthetically generated. It utilizes advanced machine learning models (CNN and ViT architectures) to perform audio classification. The application can be run as a web service with a user-friendly interface or as a command-line tool.

## 2. Project Structure

Here's an outline of the key directories and files within the project:

```
.
├── app/                        # Main application source code
│   ├── __init__.py
│   ├── audio_processing/       # Audio segmentation and spectrogram utilities
│   │   ├── __init__.py
│   │   ├── audio_segmentation.py
│   │   └── spectrogram_processing.py
│   ├── config.py               # Application configuration
│   ├── main.py                 # FastAPI application entry point
│   ├── model_definitions.py    # PyTorch model class definitions
│   ├── models/                 # Pre-trained model files (.pth)
│   │   ├── best_model_CNN_Large_...pth
│   │   └── ... (other .pth files)
│   ├── routers/                # FastAPI prediction endpoints
│   │   ├── __init__.py
│   │   └── prediction_pipeline.py # Prediction endpoint logic
│   ├── static/                 # CSS and JavaScript for the web interface
│   ├── templates/              # HTML templates
│   └── tests/                  # Unit and integration tests (currently empty)
│       └── __init__.py
├── cli.py                      # Command-Line Interface script
├── Dockerfile                  # For building a Docker container
├── GUIDE.md                    # Detailed user guide
├── workflow.md                 # Description of the application workflow
├── deploy-env.yml              # Conda environment for deploy
├── audio-env.yml               # Conda environment for train(alternative)
├── .env                        # Example environment file (create your own from this or use .env.example)
└── README.md                   # Project overview (this file should be similar to README.md)
```

## 3. Installation

Follow these steps to set up the project environment:

### 3.1. Prerequisites

-   Python 3.8+
-   Git

### 3.2. Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Set up a Python virtual environment (recommended):**
    *   Using `venv`:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    *   Or using Conda:
        ```bash
        conda create -n deepfake_audio_env python=3.9 # Or your preferred Python version
        conda activate deepfake_audio_env
        ```

3.  **Install dependencies:**
    ```bash
    conda env update -f deploy-env.yml --prune
    ```
    (Note: The `environment.yml` is an alternative for Conda users but `requirements.txt` is the primary source of dependencies).

## 4. Configuration

Application behavior is configured through `app/config.py` (Pydantic settings), which loads settings from environment variables. A `.env` file is used to manage these variables locally.

### 4.1. `.env` File

1.  Create a `.env` file in the project root. You can copy `.env.example` (if it exists) or create one from scratch.
2.  Define necessary variables. Key variables include:

    *   `MODEL_DIR`: Path to the directory containing model files (e.g., `app/models/`).
    *   `CNN_SMALL_MODEL_NAME`, `CNN_LARGE_MODEL_NAME`, `VIT_SMALL_MODEL_NAME`, `VIT_LARGE_MODEL_NAME`: Specific filenames of your models.
    *   `HOST`: Host for the FastAPI service (e.g., `0.0.0.0`).
    *   `PORT`: Port for the FastAPI service (e.g., `8000`).
    *   `DEBUG`: FastAPI debug mode (`True` or `False`).

    Example `.env`:
    ```env
    MODEL_DIR="app/models/"
    CNN_SMALL_MODEL_NAME="best_model_CNN_Small_cnn_3s_dataset_102208.pth"
    CNN_LARGE_MODEL_NAME="best_model_CNN_Large_cnn_3s_dataset_114040.pth"
    VIT_SMALL_MODEL_NAME="best_model_ViT_Small_vit_3s_dataset_040441.pth"
    VIT_LARGE_MODEL_NAME="best_model_ViT_Large_vit_3s_dataset_044740.pth"
    HOST="0.0.0.0"
    PORT="8000"
    DEBUG="False"
    ```

### 4.2. Application Configuration (`app/config.py`)

The `app/config.py` file defines Pydantic models for settings. These settings are loaded from environment variables (and thus your `.env` file). This includes model paths, audio processing parameters (`TARGET_SAMPLE_RATE`, `CHUNK_DURATION_SECONDS`, etc.), and server settings.

## 5. Running the Application

You can run this application either as a web service or as a command-line tool.

### 5.1. As a Web Service (FastAPI)

The web application provides a user interface to upload an audio file and see the prediction results from all four models.

**Start the FastAPI server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
(Adjust host and port as needed, or rely on `.env` settings. `--reload` is for development.)

Access the application by navigating to `http://localhost:8000` (or your configured host/port) in your web browser.

-   **API Documentation:** Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs`.
-   **Prediction Endpoint:** `POST /predict_audio` (defined in `app/routers/prediction_pipeline.py`).
    *   Accepts `audio_file` (multipart/form-data) and an optional `model_name` query parameter.
    *   The web UI uses this endpoint to get predictions from all models.

### 5.2. As a Command-Line Interface (CLI)

The CLI allows for quick predictions directly from the terminal.

**Syntax:**
```bash
python cli.py <audio_file_path> [--model_name <model_key>]
```
-   `<audio_file_path>`: Path to the audio file.
-   `[--model_name <model_key>]`: Optional. Specify model (`cnn_small`, `cnn_large`, `vit_small`, `vit_large`). Defaults to `cnn_small`.

**Example:**
```bash
python cli.py "path/to/your/audio.wav" --model_name vit_large
```

## 6. Workflow Overview

The application processes audio as follows:
1.  **Upload:** User uploads an audio file via the web UI or provides a path via CLI.
2.  **Requests (Web UI):** The frontend sends separate prediction requests to the backend for each of the four models using the `/predict_audio` endpoint.
3.  **Processing (Backend - `app.routers.prediction_pipeline.predict_audio_pipeline`):**
    *   The audio is received and converted to mono.
    *   It's segmented into 3-second chunks (using `app.audio_processing.audio_segmentation`).
    *   Each chunk is transformed into a Mel spectrogram, normalized, and resized (using `app.audio_processing.spectrogram_processing`).
4.  **Inference (Backend):** The selected model (loaded in `app.main` and passed to the router) performs inference on each processed chunk.
5.  **Results:** Predictions (label and confidence) for each chunk are returned. The web UI aggregates and displays these results for each model. The CLI prints them.

For a more detailed workflow, see `workflow.md`.

## 7. Testing

The `app/tests/` directory is intended for unit and integration tests.
Currently, there are no specific test files like `test_audio_preprocessing.py` present in the latest file listing.
To run tests (if/when available, e.g., using `pytest`):
```bash
pytest app/tests/
```
You may need to install `pytest`: `pip install pytest`.
Test coverage should ideally include:
*   Audio processing utilities.
*   Model loading and inference logic.
*   API endpoint behavior.
*   CLI command execution.

## 8. Deployment

A `Dockerfile` is provided for containerizing the application.

### 8.1. Using Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t deepfake_audio_detector .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -d -p 8000:8000 --env-file .env deepfake_audio_detector
    ```
    *   Ensure your `.env` file is configured, especially if `MODEL_DIR` needs to be different or if models are mounted via volumes.
    *   To use models from your host machine instead of those copied into the image (if the Dockerfile doesn't copy them or you want to override):
        ```bash
        docker run -d -p 8000:8000 --env-file .env -v /path/to/your/models_on_host:/app/models deepfake_audio_detector
        ```
        (Ensure `MODEL_DIR` in your `.env` points to `/app/models` for this to work correctly inside the container).

### 8.2. Other Considerations

*   **Production Web Server:** For production, consider running Uvicorn with Gunicorn as a process manager: `gunicorn -k uvicorn.workers.UvicornWorker app.main:app -w 4 --bind 0.0.0.0:8000`.
*   **Environment Variables:** Always use environment variables for configuration in production.
*   **Model Storage:** For larger deployments, consider storing models in a dedicated artifact repository or cloud storage.

## 9. Contributing

Contributions are welcome!
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes. Adhere to coding standards and add tests if applicable.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

This guide should help you get started with the Deepfake Audio Detection project. If you encounter any issues, please refer to the source code and ensure your environment is set up correctly.
