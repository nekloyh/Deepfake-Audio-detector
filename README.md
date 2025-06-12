# Deepfake Audio Detection

This project is a Deepfake Audio Detection system that can identify whether an audio recording is real or synthetically generated. It utilizes advanced machine learning models (CNN and ViT architectures) to perform audio classification. The application can be run as a web service with a user-friendly interface or as a command-line tool.

## Features

-   **Deepfake Audio Detection:** Classifies audio as "real" or "fake".
-   **Multiple Models:** Employs four different models for robust detection:
    -   CNN Small
    -   CNN Large
    -   ViT Small
    -   ViT Large
-   **Web Interface:** Provides an easy-to-use web UI for uploading audio files and viewing detection results from all models.
-   **CLI Tool:** Offers a command-line interface for users who prefer terminal-based operations.
-   **Docker Support:** Includes a Dockerfile for easy containerization and deployment.

## Project Structure

```
.
├── app/                        # Main application source code
│   ├── audio_processing/       # Audio segmentation and spectrogram utilities
│   ├── models/                 # Pre-trained model files (.pth)
│   ├── routers/                # FastAPI prediction endpoints
│   ├── static/                 # CSS and JavaScript for the web interface
│   ├── templates/              # HTML templates
│   ├── config.py               # Application configuration
│   ├── main.py                 # FastAPI application entry point
│   └── model_definitions.py    # PyTorch model class definitions
├── cli.py                      # Command-Line Interface script
├── Dockerfile                  # For building a Docker container
├── GUIDE.md                    # Detailed user guide
├── workflow.md                 # Description of the application workflow
├── requirements.txt            # Python dependencies for pip
├── environment.yml             # Conda environment definition (alternative)
└── README.md                   # This file
```

## Getting Started

### Prerequisites

-   Python 3.8+
-   Git

### Installation

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
        conda create -n deepfake_audio_env python=3.9
        conda activate deepfake_audio_env
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root (you can copy and rename `.env.example` if provided, or create one from scratch).
    Define necessary variables, especially model paths if they are not in the default `app/models/` directory.
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

## Usage

### 1. Web Application

The web application provides a user interface to upload an audio file and see the prediction results from all four models.

**Start the FastAPI server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
(Adjust host and port as needed. `--reload` is for development.)

Access the application by navigating to `http://localhost:8000` (or your configured host/port) in your web browser.

-   **API Documentation:** Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs`.
-   **Prediction Endpoint:** `POST /predict_audio` (accepts `audio_file` and an optional `model_name` query parameter).

### 2. Command-Line Interface (CLI)

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

## Workflow Overview

The application processes audio as follows:
1.  **Upload:** User uploads an audio file via the web UI.
2.  **Requests:** The frontend sends separate prediction requests to the backend for each of the four models.
3.  **Processing (Backend):**
    *   The audio is received and converted to mono.
    *   It's segmented into 3-second chunks.
    *   Each chunk is transformed into a Mel spectrogram, normalized, and resized.
4.  **Inference (Backend):** The selected model performs inference on each processed chunk.
5.  **Results:** Predictions (label and confidence) for each chunk are returned. The web UI aggregates and displays these results for each model.

For a more detailed workflow, see `workflow.md`.

## Docker Deployment

A `Dockerfile` is provided for containerizing the application.

1.  **Build the Docker image:**
    ```bash
    docker build -t deepfake_audio_detector .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -d -p 8000:8000 --env-file .env deepfake_audio_detector
    ```
    *   Ensure your `.env` file is configured, especially if `MODEL_DIR` needs to be different or if models are mounted via volumes.
    *   To use models from your host machine instead of those copied into the image:
        ```bash
        docker run -d -p 8000:8000 --env-file .env -v /path/to/your/models_on_host:/app/models deepfake_audio_detector
        ```
        (Ensure `MODEL_DIR` in your `.env` points to `/app/models` for this to work correctly inside the container).

## Contributing

Contributions are welcome! Please refer to `GUIDE.md` for more detailed information on the project structure and how to set up for development.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if one exists, otherwise specify).
```
