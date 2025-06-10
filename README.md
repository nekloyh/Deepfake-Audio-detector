# Audio Deepfake Detection Web Application

This project provides a web application to detect deepfake audio using pre-trained models (CNN and ViT, with small and large variants). The models are converted to ONNX format for efficient inference.

## Prerequisites

*   Python 3.8+
*   Docker (Recommended for running the web application)
*   Git (for cloning the repository)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory> # Replace <repository_directory> with the cloned folder name
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Make sure you have activated your virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Model Conversion (Optional)

The repository already includes pre-converted ONNX models in the `onnx_models/` directory. These were generated from the PyTorch (`.pth`) models found in the `results/` directory.

If you need to re-convert the models (e.g., after modifying the original PyTorch models or the conversion script), you can run:

```bash
python convert_to_onnx.py
```
This script will populate the `onnx_models/` directory with the latest ONNX model files. Ensure that the paths in `convert_to_onnx.py` to the `.pth` files are correct.

## Running the Web Application

There are two ways to run the web application:

### 1. Using Docker (Recommended)

This is the easiest and most reliable way to run the application, as it handles all environment setup.

1.  **Build the Docker Image:**
    From the project root directory (where the `Dockerfile` is located):
    ```bash
    docker build -t audio-deepfake-detector .
    ```

2.  **Run the Docker Container:**
    ```bash
    docker run -p 8000:8000 audio-deepfake-detector
    ```

3.  **Access the Application:**
    Open your web browser and navigate to:
    [http://localhost:8000](http://localhost:8000)

### 2. Running Locally (For Development)

This method requires you to have all dependencies installed in your local Python environment.

1.  **Ensure Dependencies are Installed:**
    Follow step 3 in "Setup Instructions."

2.  **Ensure ONNX Models are Present:**
    Make sure the `onnx_models/` directory exists and contains the `.onnx` files. If not, run the model conversion script as described above.

3.  **Start the FastAPI Application:**
    From the project root directory:
    ```bash
    uvicorn webapp.main:app --reload --host 0.0.0.0 --port 8000
    ```
    Alternatively, you can navigate to the `webapp` directory and run `python main.py`.

4.  **Access the Application:**
    Open your web browser and navigate to:
    [http://localhost:8000](http://localhost:8000)

## Directory Structure (Overview)

-   `convert_to_onnx.py`: Script to convert PyTorch models to ONNX.
-   `demo_workflow/models/`: Python definitions of the `BasicCNN` and `BasicViT` models.
-   `Dockerfile`: Instructions to build the Docker image for the web app.
-   `onnx_models/`: Contains the ONNX model files (`.onnx`).
-   `requirements.txt`: Lists Python dependencies.
-   `results/`: Contains the original PyTorch trained models (`.pth`).
-   `webapp/`: Contains the FastAPI web application.
    -   `main.py`: Core application logic, API endpoints.
    -   `static/`: Static files (CSS, JavaScript).
    -   `templates/`: HTML templates.
-   `README.md`: This file.
