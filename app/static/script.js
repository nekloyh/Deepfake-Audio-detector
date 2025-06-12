document.addEventListener("DOMContentLoaded", function () {
  const uploadForm = document.getElementById("uploadForm");
  const audioFile = document.getElementById("audioFile"); // Keep this for actual file handling
  const resultsContainer = document.getElementById("resultsContainer");
  const mainPlaceholder = resultsContainer
    ? resultsContainer.querySelector(".placeholder")
    : null;
  const submitButton = uploadForm
    ? uploadForm.querySelector(".submit-button")
    : null;
  const loaderContainer = document.getElementById("loaderContainer");
  const dropArea = document.getElementById("drop-area"); // New
  const fileNameDisplay = document.getElementById("file-name-display"); // New

  const modelNames = ["CNN_Small", "CNN_Large", "ViT_Small", "ViT_Large"];

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

  // Setup Drag and Drop
  if (dropArea && audioFile) {
    // Ensure audioFile input exists for assignment
    dropArea.addEventListener("dragover", (event) => {
      event.preventDefault(); // Necessary to allow drop
      dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragleave", () => {
      dropArea.classList.remove("dragover");
    });

    dropArea.addEventListener("drop", (event) => {
      event.preventDefault();
      dropArea.classList.remove("dragover");
      const files = event.dataTransfer.files;
      if (files.length > 0) {
        audioFile.files = files; // Assign dropped files to the input
        // Trigger change event. The audioFile's own change listener will handle UI updates.
        const changeEvent = new Event("change");
        audioFile.dispatchEvent(changeEvent);
      }
    });

    // Make the dropArea clickable to open the file dialog
    dropArea.addEventListener("click", () => {
      audioFile.click();
    });
  }

  // Listen for file selection via the standard input
  if (audioFile) {
    audioFile.addEventListener("change", () => {
      if (audioFile.files.length > 0) {
        updateFileNameDisplay(audioFile.files[0]);
      } else {
        updateFileNameDisplay(null);
      }
    });
  }

  // Form submission logic
  if (
    uploadForm &&
    submitButton &&
    resultsContainer &&
    mainPlaceholder && // Optional, but good to check
    loaderContainer &&
    dropArea &&
    fileNameDisplay // Ensure all key elements are present
  ) {
    uploadForm.addEventListener("submit", async function (event) {
      event.preventDefault();

      if (!audioFile.files || audioFile.files.length === 0) {
        resultsContainer.innerHTML = ""; // Clear previous results
        mainPlaceholder.textContent = "Please select an audio file.";
        mainPlaceholder.style.color = "var(--error-color, #dc3545)";
        mainPlaceholder.style.display = "block";
        resultsContainer.appendChild(mainPlaceholder);
        updateFileNameDisplay(null); // Clear file name display
        return;
      }

      // Clear previous results and hide placeholder for loading
      resultsContainer.innerHTML = "";
      if (mainPlaceholder) {
        mainPlaceholder.style.display = "none"; // Hide placeholder
      }

      if (loaderContainer) loaderContainer.style.display = "block";
      submitButton.disabled = true;
      submitButton.textContent = "Processing...";
      const currentFile = audioFile.files[0]; // Store for use in results cards

      const promises = modelNames.map((modelName) => {
        const formData = new FormData();
        formData.append("file", currentFile);
        formData.append("model_name", modelName);
        // Assuming API_BASE_URL is defined globally or adjust as needed
        const predictUrl = `/predict/`;

        return fetch(predictUrl, {
          method: "POST",
          body: formData,
          headers: {
            Accept: "application/json", // Optional: Explicitly ask for JSON
          },
        })
          .then((response) =>
            response
              .json()
              .then((data) => ({
                modelName,
                data,
                responseOk: response.ok,
                status: response.status,
                statusText: response.statusText,
              }))
              // More robust error handling for non-JSON or empty error responses
              .catch(async () => {
                let errorText = "Failed to parse JSON response.";
                try {
                  errorText = await response.text();
                  if (!errorText)
                    errorText = `Server returned status ${response.status} with no message.`;
                } catch (e) {
                  /* ignore text parsing error */
                }
                return {
                  modelName,
                  data: { detail: errorText },
                  responseOk: false,
                  status: response.status,
                  statusText: response.statusText,
                };
              })
          )
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

        resultsContainer.innerHTML = ""; // Clear again just in case, and to remove placeholder if it wasn't hidden

        if (results.length === 0 && mainPlaceholder) {
          mainPlaceholder.textContent = "No results to display.";
          mainPlaceholder.style.display = "block";
          resultsContainer.appendChild(mainPlaceholder);
        }

        results.forEach((result) => {
          const card = document.createElement("div");
          card.classList.add("result-item-card");

          if (result.status === "fulfilled") {
            const { modelName, data, responseOk, status, statusText } =
              result.value;
            // Use model_name from data if available (e.g. if server returns it), else fallback to mapped modelName
            const displayModelName =
              data.model_used || data.model_name || modelName;

            let cardContent = `<h4>${displayModelName}</h4>`;

            if (responseOk && data) {
              // Ensure data is present
              cardContent += `<p><strong>File:</strong> ${
                data.filename || currentFile.name
              }</p>`;
              cardContent += `<p><strong>Prediction:</strong> <span class="prediction-text">${
                data.predicted_class_name || "N/A"
              }</span></p>`;
              // Check if confidence is a valid number before toFixed
              const confidence =
                typeof data.confidence === "number"
                  ? data.confidence.toFixed(2) + "%"
                  : "N/A";
              cardContent += `<p><strong>Confidence:</strong> ${confidence}</p>`;

              const rawOutput =
                data.raw_output !== undefined
                  ? JSON.stringify(data.raw_output, null, 2)
                  : "N/A";
              cardContent += `<p><strong>Raw Output:</strong></p><pre>${rawOutput}</pre>`;

              if (data.predicted_class_name) {
                if (data.predicted_class_name.toLowerCase().includes("real")) {
                  card.classList.add("status-real");
                } else if (
                  data.predicted_class_name.toLowerCase().includes("fake")
                ) {
                  card.classList.add("status-fake");
                } else {
                  // Neutral or undefined status
                }
              } else {
                card.classList.add("status-fake"); // Treat as error/fake if no prediction
              }
            } else {
              // Handle non-ok responses or missing data
              card.classList.add("status-fake"); // Or a specific "status-error"
              let errorDetail = `Error processing with ${displayModelName}: ${
                status || "Unknown"
              } ${statusText || "Error"}`;
              if (data && data.detail) {
                errorDetail += `<br>Details: ${
                  typeof data.detail === "string"
                    ? data.detail
                    : JSON.stringify(data.detail, null, 2)
                }`;
              } else if (!data) {
                errorDetail += `<br>Details: No data returned from server.`;
              }
              cardContent += `<p>${errorDetail}</p>`;
            }
            card.innerHTML = cardContent;
          } else {
            // Promise rejected (network error, etc.)
            card.classList.add("status-fake"); // Or "status-error"
            const reason = result.reason || {};
            const modelNameForError = reason.modelName || "Unknown Model";
            let errorMessage = `Failed to get result for ${modelNameForError}.`;
            if (reason.data && reason.data.detail) {
              errorMessage += `<br>Details: ${reason.data.detail}`;
            } else if (reason.message) {
              errorMessage += `<br>Details: ${reason.message}`;
            }

            card.innerHTML = `<h4>Error with ${modelNameForError}</h4><p>${errorMessage}</p>`;
          }
          resultsContainer.appendChild(card);
        });

        if (resultsContainer.children.length === 0 && mainPlaceholder) {
          mainPlaceholder.textContent =
            "No results were generated, or an error occurred preventing display.";
          mainPlaceholder.style.display = "block";
          resultsContainer.appendChild(mainPlaceholder);
        }
      } catch (error) {
        // Catch errors from Promise.allSettled or other unexpected issues
        console.error("Error processing model predictions:", error);
        resultsContainer.innerHTML = ""; // Clear any partial results
        if (mainPlaceholder) {
          mainPlaceholder.textContent =
            "An unexpected error occurred. Please check the console.";
          mainPlaceholder.style.color = "var(--error-color, #dc3545)";
          mainPlaceholder.style.display = "block";
          resultsContainer.appendChild(mainPlaceholder);
        }
      } finally {
        if (loaderContainer) loaderContainer.style.display = "none";
        submitButton.disabled = false;
        submitButton.textContent = "Detect Deepfake";
      }
    });
  } else {
    // Critical error: some essential elements are missing
    console.error(
      "Essential page elements not found for script.js. Script functionality will be impaired."
    );
    if (resultsContainer && mainPlaceholder) {
      // Attempt to show an error on the page
      resultsContainer.innerHTML = ""; // Clear
      mainPlaceholder.textContent =
        "Error: Critical page elements are missing. The application cannot function correctly.";
      mainPlaceholder.style.color = "var(--error-color, #dc3545)";
      mainPlaceholder.style.display = "block";
      resultsContainer.appendChild(mainPlaceholder);
    } else if (document.body) {
      // Fallback if even resultsContainer is missing
      document.body.innerHTML =
        '<p style="color:red; text-align:center; font-size:1.2em; padding:20px;">Error: Critical page elements are missing. Application cannot load.</p>';
    }
  }
});
