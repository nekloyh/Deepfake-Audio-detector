# Application Workflow: Deepfake Audio Detector

This document outlines the complete data flow and processing pipeline for the Deepfake Audio Detector application.

## 1. Data Flow Overview

The application processes an audio file uploaded by the user, runs predictions using four different pre-trained models, and displays the results for each model.

```
+-------------------+      +----------------------+      +---------------------+      +--------------------+      +----------------------+
|   User Interface  |----->|  FastAPI Backend     |----->|  Audio Processing   |----->|  Model Inference   |----->|   User Interface     |
| (index.html, JS)  |      | (main.py, predict.py)|      | (predict.py utils)  |      |(predict.py, models)|      | (JS updates HTML)    |
+-------------------+      +----------------------+      +---------------------+      +--------------------+      +----------------------+
        |                                                                                                                  ^
        | 1. Upload Audio                                                                                                  | 5. Display Results
        |------------------------------------------------------------------------------------------------------------------|
                                        | 2. HTTP POST Requests (one per model)                                            |
                                        |----------------------------------------------------------------------------------|
                                                                | 3. Process Audio & Predict (for each model)              |
                                                                |----------------------------------------------------------|
                                                                                        | 4. Return JSON Predictions (for each model) |
                                                                                        |---------------------------------------------|
```

**Flow Steps:**

1.  **Audio Upload (Frontend):**
    *   The user visits the web application (`index.html`).
    *   The user selects an audio file using the file input field.
    *   Upon clicking "Detect Deepfake", the JavaScript (`static/script.js`) initiates the process.

2.  **HTTP POST Requests (Frontend to Backend):**
    *   The JavaScript sends four separate asynchronous HTTP POST requests to the backend.
    *   Each request targets the `/predict_audio` endpoint.
    *   A `model_name` query parameter is appended to each request URL (e.g., `/predict_audio?model_name=cnn_small`, `/predict_audio?model_name=cnn_large`, etc.) to specify which model to use for that request.
    *   The audio file is included in the `FormData` of each request under the key `"audio_file"`.

3.  **File Reception and Audio Processing (Backend):**
    *   The FastAPI backend (`app/main.py`, `app/routers/predict.py`) receives each of the four requests.
    *   For each request, the `predict_audio` endpoint in `app/routers/predict.py` handles the uploaded `audio_file` and the `model_name`.
    *   The raw audio bytes are read.
    *   The audio is converted to a mono waveform.
    *   The waveform is split into 3-second chunks (or the duration specified in `app/config.py:settings.CHUNK_DURATION_SECONDS`). Padding is applied to the last chunk if necessary.
    *   Each chunk undergoes further processing via `process_audio_for_model` in `app/routers/predict.py`:
        *   **Resampling:** Audio is resampled to `settings.TARGET_SAMPLE_RATE` (e.g., 16000 Hz).
        *   **Padding/Trimming:** Ensures the chunk matches `settings.CHUNK_DURATION_SECONDS` at the target sample rate.
        *   **Mel Spectrogram:** Converted to a Mel spectrogram using `librosa.feature.melspectrogram` with parameters like `N_FFT`, `HOP_LENGTH`, `N_MELS` from `settings`.
        *   **Logarithmic Scaling:** Converted to decibels (dB) using `librosa.power_to_db`.
        *   **Normalization:** Normalized to a [0, 1] range based on `settings.MIN_DB_LEVEL`.
        *   **Dimension Adjustment:** The spectrogram is padded or truncated to ensure its dimensions match `(settings.N_MELS, settings.SPECTROGRAM_WIDTH)`.
        *   **Batch & Channel Dimension:** Reshaped to `(1, 1, N_MELS, SPECTROGRAM_WIDTH)` to match model input requirements.

4.  **Model Inference (Backend):**
    *   The appropriate pre-loaded PyTorch model (selected based on the `model_name` from the request) is retrieved from `request.app.state.pytorch_models`.
    *   The processed spectrogram tensor for each chunk is moved to the appropriate device (CPU/GPU).
    *   The model performs inference (`model(input_tensor)`).
    *   The output logits are processed to determine the predicted label ("real" or "fake") and confidence score.
    *   A JSON response containing the filename, chunk details (if applicable, though current implementation averages or takes the most relevant chunk implicitly per request), prediction, confidence, model used, and raw model output is prepared for each chunk. The current `/predict_audio` endpoint returns a list of predictions, one for each chunk processed from the input file.

5.  **Return JSON Predictions (Backend to Frontend):**
    *   The backend sends the JSON response back to the frontend for each of the four initial requests.

6.  **Display Results (Frontend):**
    *   The JavaScript (`static/script.js`) receives the four responses (or errors).
    *   `Promise.allSettled` is used to handle all responses.
    *   For each model's response:
        *   The corresponding HTML section (e.g., `<div id="cnn_small_results">`) is updated.
        *   The prediction (label and confidence) and other details are displayed within a `<pre>` tag for that model.
        *   If an error occurred for a specific model, the error message is displayed in its section.

## 2. Model Loading and Inference

*   **Model Loading (`app/main.py`):**
    *   Models are loaded during the application startup sequence, triggered by the `@app.on_event("startup")` decorator.
    *   The `load_models` asynchronous function iterates through model names defined in `app/config.py` (e.g., `settings.CNN_SMALL_MODEL_NAME`, `settings.VIT_LARGE_MODEL_NAME`).
    *   For each model:
        *   The model architecture (e.g., `CNN_Audio`, `ViT_Audio` from `app/model_definitions.py`) is instantiated with hardcoded parameters specific to each model type (e.g., number of channels, patch size, embedding dimensions). **Note:** These parameters must match the configuration used during model training.
        *   The pre-trained model weights are loaded from `.pth` files located in the `settings.MODEL_DIR` (e.g., `app/models/`). `torch.load()` with `weights_only=False` is used (this might be necessary if the checkpoint contains non-weight data, but `weights_only=True` is generally safer if applicable).
        *   The model is set to evaluation mode (`model.eval()`).
        *   The loaded model is stored in a dictionary `app.state.pytorch_models` keyed by its name (e.g., "cnn_small").
    *   The application uses the device (CUDA or CPU) detected by PyTorch.

*   **Inference (`app/routers/predict.py`):**
    *   When a `/predict_audio` request arrives, the specified `model_name` is used to retrieve the corresponding model object from `app.state.pytorch_models`.
    *   The input tensor (processed audio chunk) is passed to the model: `outputs = model(input_tensor)`.
    *   Predictions are made within a `torch.no_grad()` context to disable gradient calculations, saving memory and computation.
    *   The raw output (logits) is converted to a NumPy array.
    *   If the output has multiple classes, `np.argmax` finds the predicted class index. Confidence is the softmax probability of that class (or the logit value itself if softmax is not explicitly applied post-model).
    *   If the output is a single logit (binary classification), a threshold (e.g., 0.5 if sigmoid applied, or 0 if raw logits) determines the class.
    *   Labels are mapped using `settings.LABELS`.

## 3. Frontend Result Display

*   **HTML Structure (`app/templates/index.html`):**
    *   The main results area is `<div id="resultsContainer">`.
    *   Inside, there's an initial placeholder message.
    *   Four dedicated divs are present for each model's output:
        *   `<div id="cnn_small_results" class="model-result"><h4>CNN Small</h4><pre></pre></div>`
        *   `<div id="cnn_large_results" class="model-result"><h4>CNN Large</h4><pre></pre></div>`
        *   `<div id="vit_small_results" class="model-result"><h4>ViT Small</h4><pre></pre></div>`
        *   `<div id="vit_large_results" class="model-result"><h4>ViT Large</h4><pre></pre></div>`
    *   These `model-result` divs are initially hidden.

*   **JavaScript Logic (`app/static/script.js`):**
    *   When a new file is submitted:
        *   All previous results in the `<pre>` tags are cleared.
        *   All `model-result` divs are hidden.
        *   The main placeholder is hidden.
        *   A loader is displayed.
    *   After receiving responses from all four model prediction requests (via `Promise.allSettled`):
        *   The loader is hidden.
        *   For each model's result:
            *   The corresponding `<pre>` tag (e.g., `cnnSmallResultsPre`) is populated with formatted prediction details (filename, model used, prediction, confidence, raw output) or an error message.
            *   The corresponding `model-result` div (e.g., `cnnSmallResultsDiv`) is made visible (`style.display = "block"`).

## 4. System Extensibility

*   **Adding New Audio Processing Steps (Backend):**
    *   Modify the `process_audio_for_model` function in `app/routers/predict.py`.
    *   New steps (e.g., noise reduction, different feature extraction) can be inserted into this function.
    *   Ensure any changes to the output shape or format are compatible with the existing models or update models accordingly.
    *   Configuration parameters for new steps can be added to `app/config.py:Settings`.

*   **Changing the Workflow / Adding Models:**
    *   **Adding a New Model:**
        1.  **Model File:** Place the new model's weights file (e.g., `.pth`) in the `app/models/` directory.
        2.  **Configuration (`app/config.py`):** Add a new setting for the model's filename, e.g., `NEW_MODEL_NAME: str = "new_model.pth"`.
        3.  **Model Definition (`app/model_definitions.py`):** If the new model uses a new architecture, define its class here.
        4.  **Model Loading (`app/main.py`):**
            *   Add logic in the `load_models` function to instantiate and load the new model, similar to existing models. This includes defining its specific parameters if hardcoded.
            *   Add its name to the `current_model_paths` dictionary.
        5.  **Frontend (`app/static/script.js`):**
            *   Add the new model's name (e.g., "new_model") to the `modelNames` array in `script.js`.
        6.  **HTML (`app/templates/index.html`):**
            *   Add a new results div for the new model in `resultsContainer`, e.g., `<div id="new_model_results" class="model-result" style="display: none;"><h4>New Model</h4><pre></pre></div>`.
        7.  **JavaScript (`app/static/script.js`):**
            *   Update `modelResultElements` map to include the new model's result div and pre tag.
    *   **Modifying Model Interaction:**
        *   If instead of calling all models, a selection mechanism is reintroduced, the frontend JavaScript would need to be changed to send requests only for selected models.
        *   The backend `/predict_audio` endpoint already supports individual model prediction via the `model_name` parameter.
        *   For more complex routing or model chaining, new endpoints or modifications to `predict_audio` might be necessary.

*   **Updating Models:**
    *   Replace the model file in `app/models/`.
    *   Update the corresponding filename in `app/config.py` if it changes.
    *   If the model architecture changes, update `app/model_definitions.py` and potentially the instantiation parameters in `app/main.py`.
    *   Restart the application for changes to take effect.
