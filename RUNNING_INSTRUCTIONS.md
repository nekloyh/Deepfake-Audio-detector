# Running the Deepfake Audio Detector Application

This guide provides instructions on how to set up and run the Deepfake Audio Detector application.

## 1. Prerequisites

*   **Python:** Version 3.8 or higher is recommended. You can download Python from [python.org](https://www.python.org/).
*   **Git:** (Optional) For cloning the repository. You can download Git from [git-scm.com](https://git-scm.com/).
*   **PyTorch:** The application now uses PyTorch for model inference. Installation instructions can be found at [pytorch.org](https://pytorch.org/). Ensure you install a version compatible with your system (CPU or GPU with CUDA).

## 2. Setup

### a. Clone the Repository (if you haven't already)

If you have Git installed, clone the repository to your local machine:

```bash
git clone <repository_url>
cd <repository_directory>
```

If you downloaded the code as a ZIP file, extract it and navigate to the project's root directory.

### b. Create and Activate a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage project dependencies.

*   **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

*   **Activate the virtual environment:**
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

### c. Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
This will install FastAPI, Uvicorn, PyTorch, Librosa, and other necessary libraries.

## 3. Model Placement

The application expects the trained PyTorch models (`.pth` files) to be located in the `models/.pth/` directory relative to the project root.

*   Ensure your `best_model_CNN_Small_cnn_3s_dataset_102208.pth` and `best_model_CNN_Large_cnn_3s_dataset_114040.pth` files (or other models you intend to use) are placed in:
    ```
    <project_root>/models/.pth/
    ```

*   The application is currently configured to load `CNN_Small` and `CNN_Large` models as specified in `app/config.py`. If you use different model filenames or add more models, you may need to update `app/config.py` accordingly.

## 4. Configuration

The main configuration for the application is in `app/config.py`.

*   **Environment Variables (`.env` file):**
    The application can load settings from a `.env` file located in the project root. While no new specific environment variables were added for this transition, existing ones (like `DEBUG`, `HOST`, `PORT`) can be set here. Example `.env` file:
    ```env
    DEBUG=True
    HOST=0.0.0.0
    PORT=8000
    ```

*   **Model Configuration (`app/config.py`):**
    *   `MODEL_DIR`: This is set to `"models/.pth/"`. Ensure this matches where you've placed your models.
    *   `CNN_SMALL_MODEL_NAME`, `CNN_LARGE_MODEL_NAME`: These specify the filenames of the models to be loaded.
    *   Audio processing parameters (`TARGET_SAMPLE_RATE`, `N_MELS`, etc.) are also defined here. Usually, these should match the parameters used during model training.

## 5. Running the Application

Once the setup and configuration are complete, you can run the FastAPI application using Uvicorn:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

*   `--reload`: Enables auto-reloading when code changes (useful for development). Remove this for production.
*   `--host 0.0.0.0`: Makes the server accessible from other devices on your network.
*   `--port 8000`: Specifies the port to run on.

After running the command, you should see output indicating that Uvicorn is running, similar to:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using statreload
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Attempting to load PyTorch models from directory: <absolute_path_to_your_project>/models/.pth/
INFO:     Using device: cuda  (or cpu if CUDA is not available)
INFO:     Loading PyTorch model: cnn_small from <absolute_path_to_your_project>/models/.pth/best_model_CNN_Small_cnn_3s_dataset_102208.pth (abs: ...)
INFO:     Successfully loaded PyTorch model: cnn_small
INFO:     Loading PyTorch model: cnn_large from <absolute_path_to_your_project>/models/.pth/best_model_CNN_Large_cnn_3s_dataset_114040.pth (abs: ...)
INFO:     Successfully loaded PyTorch model: cnn_large
INFO:     All configured PyTorch models loaded successfully.
INFO:     Application startup complete.
```

You can then access the application by opening a web browser and navigating to:

[http://localhost:8000](http://localhost:8000) or [http://<your_machine_ip>:8000](http://<your_machine_ip>:8000)

## 6. Troubleshooting

*   **`ModuleNotFoundError`:** If you encounter errors like `ModuleNotFoundError: No module named 'torch'`, ensure you have activated your virtual environment and installed all dependencies from `requirements.txt`.
*   **Model Not Found:** If the application logs warnings like "Model file not found" or "No PyTorch models will be loaded," double-check:
    *   The model files (`.pth`) are correctly placed in the `models/.pth/` directory.
    *   The filenames in `app/config.py` (`CNN_SMALL_MODEL_NAME`, `CNN_LARGE_MODEL_NAME`) exactly match the names of your `.pth` files.
*   **CUDA Issues:** If you expect GPU acceleration but the log says "Using device: cpu":
    *   Verify your PyTorch installation includes CUDA support (`torch.cuda.is_available()` should be `True` in a Python interpreter).
    *   Ensure your GPU drivers are correctly installed and compatible with the CUDA version PyTorch was compiled with.
*   **Frontend Issues:** If the web interface looks unstyled, ensure your browser is not caching old versions of CSS/JS files (try a hard refresh: Ctrl+Shift+R or Cmd+Shift+R). Note: There was an issue updating `style.css` during development, so styles might not be fully updated.

---
Happy Detecting!
