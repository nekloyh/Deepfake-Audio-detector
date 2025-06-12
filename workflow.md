# Application Workflow: Deepfake Audio Detector

This document outlines the complete data flow and processing pipeline for the Deepfake Audio Detector application.

## 1. Data Flow Overview

The application processes an audio file uploaded by the user, runs predictions using four different pre-trained models, and displays the results for each model.

```
+-------------------+     +------------------------------------+     +------------------------------------+     +------------------------------------+      +----------------------+
|   User Interface  |---->|  FastAPI Backend                   |---->|  Audio Processing                  |---->|  Model Inference                   |----->|   User Interface     |
| (index.html, JS)  |     | (main.py, prediction_pipeline.py)  |     | (prediction_pipeline.py utils)     |     |(prediction_pipeline.py, models)    |      | (JS updates HTML)    |
+-------------------+     +------------------------------------+     +------------------------------------+     +------------------------------------+      +----------------------+
        |                                                                                                                                                              ^
        | 1. Upload Audio (Drag & Drop file)                                                                                                                           | 6. Display Results
        |--------------------------------------------------------------------------------------------------------------------------------------------------------------|
                                        | 2. HTTP POST Requests (one per model to /predict/)                                                                           |
                                        |------------------------------------------------------------------------------------------------------------------------------|
                                                                                | 3. File Reception & Initial Validation                                               |
                                                                                |--------------------------------------------------------------------------------------|
                                                                                                        | 4. Audio Segmentation & Spectrogram Generation (per chunk)   |
                                                                                                        |--------------------------------------------------------------|
                                                                                                                         | 5. Return JSON Predictions (for each model) |
                                                                                                                         |---------------------------------------------|
```

**Flow Steps:**

1.  **Audio Upload (Frontend):**
    *   The user visits the web application (`app/templates/index.html`).
    *   The user selects an audio file by dragging and dropping onto the designated area.
    *   Upon clicking "Detect Deepfake", the JavaScript (`app/static/script.js`) initiates the process.

2.  **HTTP POST Requests (Frontend to Backend):**
    *   The JavaScript sends separate asynchronous HTTP POST requests to the backend for each model defined in its `modelNames` array (e.g., "CNN_Small", "CNN_Large", "ViT_Small", "ViT_Large").
    *   Each request targets the `/predict/` endpoint (defined in `app/routers/prediction_pipeline.py`).
    *   The audio file (from `audioFile.files[0]`) and the `model_name` are included in the `FormData` of each request.

3.  **File Reception and Audio Processing (Backend):**
    *   The FastAPI backend (`app/main.py`, `app/routers/prediction_pipeline.py`) receives each request at the `/predict/` endpoint.
    *   The `predict_endpoint` function in `app/routers/prediction_pipeline.py` handles the uploaded `file` (as `UploadFile`) and the `model_name` (from `Form` data).
    *   The audio file is saved temporarily.
    *   The `predict_audio_file` function is called, which internally uses `segment_audio` (from `app.audio_processing.audio_segmentation`) to split the audio into chunks (e.g., 3-second duration as per `settings.CHUNK_DURATION_SECONDS`, with potential overlap defined by `settings.SEGMENT_OVERLAP_SECONDS`).
    *   Each chunk undergoes further processing within `predict_single_segment`, which calls:
        *   `create_mel_spectrogram` (from `app.audio_processing.spectrogram_processing`): Converts audio chunk to a Mel spectrogram using parameters from `settings` (e.g., `TARGET_SAMPLE_RATE`, `N_MELS`, `N_FFT`, `HOP_LENGTH`, `LOUDNESS_LUFS`).
        *   `preprocess_spectrogram_to_tensor` (from `app.audio_processing.spectrogram_processing`): Normalizes the spectrogram using `settings.PIXEL_MEAN` and `settings.PIXEL_STD`, resizes it to `settings.IMAGE_SIZE`, and converts it to a tensor. The tensor is reshaped to `(1, 1, IMAGE_SIZE, IMAGE_SIZE)` or similar for model input.

4.  **Model Inference (Backend):**
    *   The appropriate pre-loaded PyTorch model (selected based on `model_name`) is retrieved from `request.app.state.pytorch_models`.
    *   The processed spectrogram tensor for each chunk is moved to the device (CPU/GPU).
    *   The model performs inference on each chunk (`model(input_tensor)`).
    *   The output logits from all chunks are collected.
    *   `aggregate_predictions` function processes these logits:
        *   It may apply outlier removal.
        *   It aggregates chunk predictions using the method specified in `settings.AGGREGATION_METHOD` (e.g., "mean_probs", "majority_vote").
        *   It may apply class balancing based on `settings.BIAS_METHOD` and `settings.REAL_BIAS_FACTOR`.
        *   The final aggregated probabilities and predicted class index are determined.
    *   Prediction reliability is assessed.

5.  **Return JSON Predictions (Backend to Frontend):**
    *   A JSON response is prepared containing: filename, model used, predicted class index and name, aggregated probabilities, final confidence (adjusted by reliability), aggregation method, number of segments processed, reliability assessment, bias reduction details, and raw output (logits from chunks).
    *   The backend sends this JSON response back to the frontend for each of the initial requests.

6.  **Display Results (Frontend):**
    *   The JavaScript (`app/static/script.js`) receives the responses using `Promise.allSettled`.
    *   For each model's response:
        *   A new `div` with class `result-item-card` is dynamically created and appended to the `resultsContainer` in `app/templates/index.html`.
        *   The card is populated with the model name, filename, prediction (label and confidence), and raw output.
        *   The card's appearance (e.g., background color) might change based on the prediction (e.g., `status-real`, `status-fake`).
        *   If an error occurred for a specific model, the error message is displayed in its card.

## 2. Model Loading and Inference

*   **Model Loading (`app/main.py`):**
    *   Models are loaded during application startup via the `load_models` function (triggered by `@app.on_event("startup")`).
    *   It iterates through model filenames specified in `app/config.py:Settings` (e.g., `settings.CNN_SMALL_MODEL_NAME`).
    *   For each model:
        *   The corresponding architecture (e.g., `CNN_Audio`, `ViT_Audio` from `app/model_definitions.py`) is instantiated. Parameters for these architectures (like number of channels, patch size, embedding dimensions) are hardcoded within `app/main.py` during instantiation.
        *   Model weights are loaded from `.pth` files in `settings.MODEL_DIR` using `torch.load(..., weights_only=False)`.
        *   The model is set to evaluation mode (`model.eval()`) and moved to the appropriate device (CUDA/CPU).
        *   Loaded models are stored in `app.state.pytorch_models`, keyed by a simplified name (e.g., "cnn_small").

*   **Inference (`app/routers/prediction_pipeline.py`):**
    *   When a `/predict/` request arrives, the `model_name` from the form data is used to retrieve the model object from `request.app.state.pytorch_models`.
    *   Input tensors (processed audio chunks) are passed to the model: `output_logits = model(input_tensor)`.
    *   This is done within a `torch.no_grad()` context.
    *   The raw logits are collected and then aggregated as described in Step 4 of "Data Flow Overview".
    *   The final predicted class index is determined by `np.argmax` on the aggregated probabilities.
    *   Labels are mapped using `settings.LABELS` (e.g., `{0: "real", 1: "fake"}`).

## 3. Frontend Result Display

*   **HTML Structure (`app/templates/index.html`):**
    *   The main results area is `<div id="resultsContainer">`.
    *   It initially contains a placeholder message.
    *   There are no pre-defined static divs for each model's output; results are displayed dynamically.

*   **JavaScript Logic (`app/static/script.js`):**
    *   When a new file is submitted:
        *   The `resultsContainer` is cleared.
        *   The placeholder is hidden.
        *   A loader (`loaderContainer`) is displayed.
    *   After receiving responses from all model prediction requests (via `Promise.allSettled`):
        *   The loader is hidden.
        *   For each model's result:
            *   A `div` element with class `result-item-card` is created.
            *   This card is populated with:
                *   Model name.
                *   Filename.
                *   Prediction (label and confidence).
                *   Raw output (from the JSON response).
            *   The card is styled based on the prediction (e.g., `status-real` or `status-fake` class).
            *   The card is appended to `resultsContainer`.
        *   Error messages are also displayed within these dynamic cards if a request fails.

## 4. System Extensibility

*   **Adding New Audio Processing Steps (Backend):**
    *   Modify the `process_audio_for_model` function in `app/routers/predict.py`.
    *   New steps (e.g., noise reduction, different feature extraction) can be inserted into this function.
    *   Ensure any changes to the output shape or format are compatible with the existing models or update models accordingly.
    *   Configuration parameters for new steps can be added to `app/config.py:Settings`.

*   **Changing the Workflow / Adding Models:**
    *   **Adding a New Model:**
        1.  **Model File:** Place the new model's weights file (e.g., `.pth`) in the `settings.MODEL_DIR` (e.g.,`app/models/`).
        2.  **Configuration (`app/config.py`):** Add a new setting in `Settings` class for the model's filename, e.g., `NEW_MODEL_XYZ_NAME: str = "new_model_xyz.pth"`.
        3.  **Model Definition (`app/model_definitions.py`):** If the new model uses a new architecture not already present, define its PyTorch `nn.Module` class here.
        4.  **Model Loading (`app/main.py`):**
            *   In the `load_models` function, add an `if hasattr(settings, "NEW_MODEL_XYZ_NAME") ...` block.
            *   Inside, add the new model key (e.g., "new_model_xyz") and its path to `current_model_paths`.
            *   Add an `elif model_name == "new_model_xyz":` block to instantiate the model (from `app.model_definitions.py`) with its specific parameters (these might need to be hardcoded in `app/main.py` or derived from `settings`).
            *   Ensure it's loaded into the `pytorch_models` dictionary.
        5.  **Frontend (`app/static/script.js`):**
            *   Add the new model's key (e.g., "New_Model_Xyz" - matching the key used in `FormData` which `app/main.py` will use for `pytorch_models` dict) to the `modelNames` array. The script uses these names for display and for the `model_name` field in the form data.
        6.  **HTML (`app/templates/index.html`):** No changes are strictly necessary in the HTML as the results are dynamically generated by `app/static/script.js`.
        7.  **JavaScript (`app/static/script.js`):** No changes are typically needed here if the new model follows the same prediction result structure, as the script iterates through `modelNames` and creates result cards dynamically.
    *   **Modifying Model Interaction:**
        *   The backend `/predict/` endpoint supports individual model prediction via the `model_name` form field.
        *   The frontend `app/static/script.js` currently sends requests for all models listed in its `modelNames` array. To change this (e.g., allow user selection), the JavaScript logic for creating and sending `FormData` would need modification.
        *   For more complex routing or model chaining, new endpoints or modifications to `predict_audio` might be necessary.

*   **Updating Models:**
    *   Replace the model file in `app/models/`.
    *   Update the corresponding filename in `app/config.py` if it changes.
    *   If the model architecture changes, update `app/model_definitions.py` and potentially the instantiation parameters in `app/main.py`.
    *   Restart the application for changes to take effect.
