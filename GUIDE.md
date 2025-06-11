# Deepfake Audio Detection - User Guide

## 1. Project Overview

This project provides a system for detecting deepfake audio. It can operate as a web service (API) or a command-line interface (CLI) tool. The system uses PyTorch models (CNN and Vision Transformer architectures) to classify audio segments as either "real" or "fake".

## 2. Project Structure

Here's an outline of the key directories and files within the project:

```
.
├── app/                        # Main application source code
│   ├── __init__.py
│   ├── audio_processing/       # Audio splitting and manipulation utilities
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── config.py               # Application configuration (Pydantic settings)
│   ├── main.py                 # FastAPI application entry point, model loading
│   ├── model_definitions.py    # PyTorch model class definitions (CNN_Audio, ViT_Audio)
│   ├── models/                 # Directory for storing .pth model files
│   │   ├── best_model_CNN_Large_...pth
│   │   └── ... (other .pth files)
│   ├── routers/                # FastAPI routers
│   │   ├── __init__.py
│   │   └── predict.py          # Prediction endpoint logic
│   ├── static/                 # Static files for web interface (CSS, JS)
│   ├── templates/              # HTML templates for web interface
│   └── tests/                  # Unit and integration tests
│       ├── __init__.py
│       └── test_audio_preprocessing.py
├── cli.py                      # Command-Line Interface script
├── Dockerfile                  # For building a Docker container
├── GUIDE.md                    # This guide
├── requirements.txt            # Pip dependencies
├── environment.yml             # Conda environment (extensive, see notes in Installation)
├── .env                        # Example environment file (create your own from this)
└── ... (other project files)
```

## 3. Installation

Follow these steps to set up the project environment:

### 3.1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### 3.2. Set Up a Python Virtual Environment (Recommended)

Using a virtual environment helps manage dependencies and avoid conflicts.

**Using `venv` (standard Python):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using Conda:**
```bash
conda create -n deepfake_audio_env python=3.9 # Or your preferred Python version
conda activate deepfake_audio_env
```

### 3.3. Install Dependencies

**Primary method (using pip and `requirements.txt`):**
This is the recommended method and aligns with the `Dockerfile`.
```bash
pip install -r requirements.txt
```

**Alternative (using Conda and `environment.yml`):**
The provided `environment.yml` is very extensive and may contain packages beyond the immediate needs of this application. It also appears to have encoding issues (UTF-16 with BOM and null characters) and might require cleanup before use.
If you choose to use it:
```bash
# Ensure environment.yml is cleaned up (UTF-8, no BOM, no null chars between letters)
# conda env create -f environment.yml
```
It's generally recommended to create a minimal Conda environment and install `requirements.txt` into it if you prefer Conda.

## 4. Configuration

Application behavior is configured through `app/config.py`, which loads settings from environment variables and a `.env` file.

### 4.1. `.env` File

1.  Create a `.env` file in the project root directory (you can copy/rename an example if one is provided, or create it from scratch).
2.  Define your environment-specific settings in this file. Key variables include:

    *   `MODEL_DIR`: Path to the directory containing your `.pth` model files (default: `app/models/`).
    *   `CNN_SMALL_MODEL_NAME`, `CNN_LARGE_MODEL_NAME`, `VIT_SMALL_MODEL_NAME`, `VIT_LARGE_MODEL_NAME`: Filenames of your specific model files within `MODEL_DIR`.
    *   `HOST`: Host address for the FastAPI service (default: `0.0.0.0`).
    *   `PORT`: Port for the FastAPI service (default: `8000`).
    *   `DEBUG`: Set to `True` for FastAPI debug mode and auto-reload (default: `False`).

    Example `.env` content:
    ```env
    MODEL_DIR="app/models/"
    CNN_SMALL_MODEL_NAME="best_model_CNN_Small_cnn_3s_dataset_102208.pth"
    # ... other model names ...
    HOST="0.0.0.0"
    PORT="8000"
    DEBUG="False"
    ```

### 4.2. Audio Processing Parameters

Parameters like `TARGET_SAMPLE_RATE`, `CHUNK_DURATION_SECONDS`, `N_MELS`, `SPECTROGRAM_WIDTH`, etc., are defined in `app/config.py` and are not typically changed via `.env` unless `app/config.py` is modified to load them from environment variables.

## 5. Running the Application

You can run this application either as a web service or as a command-line tool.

### 5.1. As a Web Service (FastAPI)

The web service provides an API endpoint for audio predictions.

**Command to start the server:**
```bash
uvicorn app.main:app --host <your_host_ip_or_0.0.0.0> --port <your_port> --reload
```
*   Replace `<your_host_ip_or_0.0.0.0>` with the IP address to bind to (e.g., `0.0.0.0` to be accessible from other machines, or `127.0.0.1` for local access only).
*   Replace `<your_port>` with the desired port (e.g., `8000`).
*   `--reload` enables auto-reloading when code changes (useful for development). This can be omitted for production.

The values for host and port will default to those in your `.env` file or `app/config.py` if not specified in the command.

**Accessing the API:**
*   **Interactive API Docs (Swagger UI):** Open your browser and go to `http://<your_host_ip>:<your_port>/docs`
*   **Prediction Endpoint:** `POST /predict_audio`
    *   **Request:** `multipart/form-data` with:
        *   `audio_file`: The audio file to analyze.
        *   `model_name` (query parameter, optional): The key of the model to use (e.g., `cnn_small`, `cnn_large`, `vit_small`, `vit_large`). Defaults to `cnn_small`.
    *   **Example using `curl`:**
        ```bash
        curl -X POST -F "audio_file=@/path/to/your/audio.wav" "http://127.0.0.1:8000/predict_audio?model_name=cnn_small"
        ```
    *   **Example using Python `requests`:**
        ```python
        import requests

        files = {'audio_file': open('/path/to/your/audio.wav', 'rb')}
        params = {'model_name': 'cnn_small'}
        response = requests.post("http://127.0.0.1:8000/predict_audio", files=files, params=params)

        if response.status_code == 200:
            print(response.json())
        else:
            print(f"Error: {response.status_code}", response.text)
        ```

### 5.2. As a Command-Line Interface (CLI)

The CLI tool allows you to make predictions directly from your terminal.

**Command Syntax:**
```bash
python cli.py <audio_file_path> [--model_name <model_key>]
```

*   **`audio_file_path` (required):** Path to the audio file you want to analyze.
*   **`--model_name <model_key>` (optional):** Specifies which model to use.
    *   Choices: `cnn_small`, `cnn_large`, `vit_small`, `vit_large`.
    *   Defaults to `cnn_small` (or the filename part of `settings.CNN_SMALL_MODEL_NAME`).
    *   Model names are case-insensitive.

**Example Usage:**
```bash
python cli.py "path/to/sample_audio.wav"
python cli.py "another_audio.flac" --model_name vit_large
```

The CLI will output predictions for each chunk of the audio file.

## 6. Testing

The project includes unit tests to verify certain functionalities.

**Running Tests:**
Currently, tests can be run by directly executing the test files. The main test file is for audio preprocessing:
```bash
python app/tests/test_audio_preprocessing.py
```

**Test Coverage:**
*   `app/tests/test_audio_preprocessing.py`: Tests the `process_audio_for_model` function, ensuring correct output shape, data type, and value range for the processed spectrograms.

More tests can be added to cover model loading, prediction logic, and CLI behavior.

## 7. Deployment (Conceptual)

Here are some general guidelines for deploying this application.

### 7.1. Using Docker (Recommended)

A `Dockerfile` is provided to build a container image for the application.

**Build the Docker Image:**
```bash
docker build -t deepfake_audio_detector .
```

**Run the Docker Container:**
```bash
docker run -d -p 8000:8000 --env-file .env -v /path/to/your/models_on_host:/app/models deepfake_audio_detector
```
*   `-d`: Run in detached mode.
*   `-p 8000:8000`: Map port 8000 of the host to port 8000 in the container (adjust if your app uses a different port).
*   `--env-file .env`:  Pass environment variables from your local `.env` file to the container. This is useful for managing configurations like model names without rebuilding the image. Ensure your `.env` file is present and correctly configured.
*   `-v /path/to/your/models_on_host:/app/models`: **Important for models.** This mounts your local model directory into the container at `/app/models`. This is generally preferred over copying models into the image if models are large or change frequently. Ensure `MODEL_DIR` in your container's environment (via `.env`) is set to `/app/models` or the path you use inside the container. The Dockerfile now copies the `app/models` directory from the build context, so this volume mount is an alternative if you want to use models external to the build context.

### 7.2. Other Considerations

*   **Production Web Server:** While Uvicorn is great for development, for production, consider running it behind a more robust server like Gunicorn managing Uvicorn workers, or using a managed cloud service.
*   **Environment Variables:** Ensure all sensitive or environment-specific configurations are managed through environment variables (leveraging the `.env` file for local development and actual environment variables in production).
*   **Model Storage:** For larger deployments, models might be stored in cloud storage (like AWS S3, Google Cloud Storage) and downloaded to the application instances as needed.
*   **Scalability:** Depending on the load, you might need to deploy multiple instances of the application behind a load balancer.

This guide should help you get started with the Deepfake Audio Detection project. If you encounter any issues, please refer to the source code and ensure your environment is set up correctly.
