document.addEventListener("DOMContentLoaded", function () {
  const uploadForm = document.getElementById("uploadForm");
  const audioFile = document.getElementById("audioFile");
  const resultsContainer = document.getElementById("resultsContainer");
  const mainPlaceholder = resultsContainer
    ? resultsContainer.querySelector(".placeholder")
    : null;
  const submitButton = uploadForm
    ? uploadForm.querySelector(".submit-button")
    : null;
  const loaderContainer = document.getElementById("loaderContainer");
  const dropArea = document.getElementById("drop-area"); // Still use this as the clickable area
  const fileNameDisplay = document.getElementById("file-name-display");

  // Define your API base URL. IMPORTANT: Change this if your backend is on a different origin.
  const API_BASE_URL = ""; // e.g., "http://localhost:8000" if your backend is there

  const modelNames = ["CNN_Small", "CNN_Large", "ViT_Small", "ViT_Large"];
  const SUPPORTED_FILE_TYPES = [
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/mp3",
  ]; // Add more as needed

  // Helper function to show/hide the placeholder
  function togglePlaceholder(show, message = "", isError = false) {
    if (mainPlaceholder) {
      mainPlaceholder.textContent = message;
      mainPlaceholder.style.color = isError
        ? "var(--error-color, #dc3545)"
        : "var(--text-color, #333)";
      mainPlaceholder.style.display = show ? "block" : "none";
    }
  }

  // Function to update the file name display
  function updateFileNameDisplay(file) {
    if (file) {
      fileNameDisplay.textContent = `Selected file: ${file.name}`;
      fileNameDisplay.style.display = "block";
    } else {
      fileNameDisplay.textContent = "";
      fileNameDisplay.style.display = "none";
    }
  }

  // --- REMOVED DRAG AND DROP LISTENERS ---

  // Make the dropArea clickable to open the file dialog
  // This is the core part for "click to select"
  if (dropArea && audioFile) {
    dropArea.addEventListener("click", (event) => {
      // Prevent any default behavior of the div itself
      event.preventDefault();
      // Stop the event from bubbling up to parents
      event.stopPropagation();
      // Programmatically click the hidden file input
      audioFile.click();
    });
  }

  // Listen for file selection via the standard input
  if (audioFile) {
    audioFile.addEventListener("change", () => {
      if (audioFile.files.length > 0) {
        const selectedFile = audioFile.files[0];
        if (!SUPPORTED_FILE_TYPES.includes(selectedFile.type)) {
          togglePlaceholder(
            true,
            "Unsupported file type. Please upload an audio file (e.g., MP3, WAV).",
            true
          );
          updateFileNameDisplay(null);
          audioFile.value = ""; // Clear the file input to allow re-selection
          return;
        }
        updateFileNameDisplay(selectedFile);
        togglePlaceholder(false); // Hide placeholder if a valid file is selected
      } else {
        updateFileNameDisplay(null);
        togglePlaceholder(false); // Hide placeholder if no file is selected
      }
    });
  }

  // Form submission logic
  if (
    uploadForm &&
    submitButton &&
    resultsContainer &&
    loaderContainer &&
    dropArea && // dropArea still checked as it's the click target
    fileNameDisplay
  ) {
    uploadForm.addEventListener("submit", async function (event) {
      event.preventDefault();

      if (!audioFile.files || audioFile.files.length === 0) {
        resultsContainer.innerHTML = "";
        togglePlaceholder(true, "Please select an audio file.", true);
        updateFileNameDisplay(null);
        return;
      }

      const currentFile = audioFile.files[0];

      if (!SUPPORTED_FILE_TYPES.includes(currentFile.type)) {
        resultsContainer.innerHTML = "";
        togglePlaceholder(
          true,
          "Unsupported file type. Please upload an audio file (e.g., MP3, WAV).",
          true
        );
        updateFileNameDisplay(null);
        audioFile.value = ""; // Clear the file input to allow re-selection
        return;
      }

      // Clear previous results and hide placeholder for loading
      resultsContainer.innerHTML = "";
      togglePlaceholder(false); // Hide placeholder during processing

      loaderContainer.style.display = "block";
      submitButton.disabled = true;
      submitButton.textContent = "Processing...";

      const promises = modelNames.map((modelName) => {
        const formData = new FormData();
        formData.append("file", currentFile);
        formData.append("model_name", modelName);
        const predictUrl = `${API_BASE_URL}/predict/`;

        return fetch(predictUrl, {
          method: "POST",
          body: formData,
          headers: {
            Accept: "application/json",
          },
        })
          .then(async (response) => {
            let data;
            try {
              data = await response.json();
            } catch (jsonError) {
              let rawText = "";
              try {
                rawText = await response.text();
              } catch (textError) {
                /* ignore */
              }
              data = {
                detail: `Failed to parse JSON response. Server said: ${
                  rawText || "No response body."
                }`,
              };
            }
            return {
              modelName,
              data,
              responseOk: response.ok,
              status: response.status,
              statusText: response.statusText,
            };
          })
          .catch((error) => ({
            modelName,
            data: { detail: `Network Error: ${error.message}` },
            responseOk: false,
            error: true,
            status: 0,
            statusText: "Network Error",
          }));
      });

      try {
        const results = await Promise.allSettled(promises);

        resultsContainer.innerHTML = ""; // Clear again to ensure clean slate

        let allFailed = true; // Track if all models failed

        results.forEach((result) => {
          const card = document.createElement("div");
          card.classList.add("result-item-card");

          if (result.status === "fulfilled") {
            const { modelName, data, responseOk, status, statusText } =
              result.value;
            const displayModelName =
              data.model_used || data.model_name || modelName; // Prioritize server's model name

            let cardContent = `<h4>${displayModelName}</h4>`;

            if (responseOk && data && data.predicted_class_name !== undefined) {
              allFailed = false; // At least one model returned a valid prediction
              cardContent += `<p><strong>File:</strong> ${
                data.filename || currentFile.name
              }</p>`;
              cardContent += `<p><strong>Prediction:</strong> <span class="prediction-text">${
                data.predicted_class_name || "N/A"
              }</span></p>`;

              const confidence =
                typeof data.confidence === "number"
                  ? `${(data.confidence * 100).toFixed(2)}%` // Format as percentage
                  : "N/A";
              cardContent += `<p><strong>Confidence:</strong> ${confidence}</p>`;

              const rawOutput =
                data.raw_output !== undefined
                  ? JSON.stringify(data.raw_output, null, 2)
                  : "N/A";
              cardContent += `<p><strong>Raw Output:</strong></p><pre>${rawOutput}</pre>`;

              if (data.predicted_class_name.toLowerCase().includes("real")) {
                card.classList.add("status-real");
              } else if (
                data.predicted_class_name.toLowerCase().includes("fake")
              ) {
                card.classList.add("status-fake");
              } else {
                card.classList.add("status-neutral"); // For predictions that are neither real nor fake
              }
            } else {
              // Handle non-ok responses or missing/invalid prediction data
              card.classList.add("status-error"); // A new class for general errors
              let errorDetail = `Error with ${displayModelName} (${
                status || "Unknown"
              } ${statusText || "Error"})`;
              if (data && data.detail) {
                errorDetail += `<br>Details: ${
                  typeof data.detail === "string"
                    ? data.detail
                    : JSON.stringify(data.detail, null, 2)
                }`;
              } else if (data) {
                errorDetail += `<br>Details: Server returned data, but no predicted class name.`;
                errorDetail += `<br><pre>${JSON.stringify(
                  data,
                  null,
                  2
                )}</pre>`;
              } else {
                errorDetail += `<br>Details: No data returned from server.`;
              }
              cardContent += `<p class="error-message">${errorDetail}</p>`;
            }
            card.innerHTML = cardContent;
          } else {
            // Promise rejected (e.g., network error, `fetch` itself failed)
            card.classList.add("status-error");
            const reason = result.reason || {};
            const modelNameForError = reason.modelName || "Unknown Model";
            let errorMessage = `Failed to get result for ${modelNameForError}.`;
            if (reason.data && reason.data.detail) {
              errorMessage += `<br>Details: ${reason.data.detail}`;
            } else if (reason.message) {
              errorMessage += `<br>Details: ${reason.message}`;
            }
            card.innerHTML = `<h4>Error with ${modelNameForError}</h4><p class="error-message">${errorMessage}</p>`;
          }
          resultsContainer.appendChild(card);
        });

        if (allFailed) {
          togglePlaceholder(
            true,
            "No models returned successful predictions. Please check server logs or try again.",
            true
          );
        }
      } catch (error) {
        console.error("Error processing model predictions:", error);
        resultsContainer.innerHTML = "";
        togglePlaceholder(
          true,
          "An unexpected client-side error occurred. Please check the browser console for details.",
          true
        );
      } finally {
        loaderContainer.style.display = "none";
        submitButton.disabled = false;
        submitButton.textContent = "Detect Deepfake";
      }
    });
  } else {
    // Critical error: some essential elements are missing
    console.error(
      "Essential page elements not found for script.js. Script functionality will be impaired."
    );
    if (resultsContainer) {
      resultsContainer.innerHTML = "";
      togglePlaceholder(
        true,
        "Error: Critical page elements are missing. The application cannot function correctly. Please ensure your HTML has the required IDs.",
        true
      );
    } else if (document.body) {
      document.body.innerHTML =
        '<p style="color:red; text-align:center; font-size:1.2em; padding:20px;">Error: Critical page elements are missing. Application cannot load.</p>';
    }
  }
});
