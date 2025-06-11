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

  const modelNames = ["cnn_small", "cnn_large", "vit_small", "vit_large"];

  const modelResultElements = {};
  modelNames.forEach((modelName) => {
    const div = document.getElementById(`${modelName}_results`);
    const pre = div ? div.querySelector("pre") : null;
    if (div && pre) {
      modelResultElements[modelName] = { div, pre };
    } else {
      console.error(`Result elements for model ${modelName} not found.`);
    }
  });

  if (
    uploadForm &&
    submitButton &&
    resultsContainer &&
    mainPlaceholder &&
    Object.keys(modelResultElements).length === modelNames.length
  ) {
    uploadForm.addEventListener("submit", async function (event) {
      event.preventDefault();

      if (!audioFile.files || !audioFile.files.length) {
        if (mainPlaceholder) {
          mainPlaceholder.textContent = "Please select an audio file.";
          mainPlaceholder.style.color = "var(--error-color)";
          mainPlaceholder.style.display = "block";
        }
        // Hide all model-specific results
        modelNames.forEach((modelName) => {
          if (modelResultElements[modelName]) {
            modelResultElements[modelName].div.style.display = "none";
            modelResultElements[modelName].pre.textContent = "";
          }
        });
        return;
      }

      // Show loader and disable button
      if (mainPlaceholder) mainPlaceholder.style.display = "none"; // Hide main placeholder
      modelNames.forEach((modelName) => {
        // Clear and hide previous results
        if (modelResultElements[modelName]) {
          modelResultElements[modelName].div.style.display = "none";
          modelResultElements[modelName].pre.textContent = "";
        }
      });
      if (loaderContainer) loaderContainer.style.display = "block";
      submitButton.disabled = true;
      submitButton.textContent = "Processing...";

      const file = audioFile.files[0];
      const promises = modelNames.map((modelName) => {
        const formData = new FormData();
        formData.append("file", file); // Use the same file for all models
        formData.append("model_name", modelName);
        const predictUrl = `/predict/`;

        return fetch(predictUrl, {
          method: "POST",
          body: formData,
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
              .catch(async (e_json) => {
                // Handle cases where response.json() fails (e.g. non-JSON error)
                let errorText = await response
                  .text()
                  .catch(() => "Could not retrieve error message.");
                return {
                  modelName,
                  data: { detail: errorText },
                  responseOk: false,
                  status: response.status,
                  statusText: response.statusText,
                };
              })
          )
          .catch((error) => ({ modelName, error, responseOk: false })); // Network errors or other issues
      });

      try {
        const results = await Promise.allSettled(promises);

        if (mainPlaceholder) mainPlaceholder.style.display = "none"; // Ensure placeholder is hidden

        results.forEach((result) => {
          if (result.status === "fulfilled") {
            const { modelName, data, responseOk, status, statusText } =
              result.value;
            const elements = modelResultElements[modelName];
            if (!elements) return;

            elements.div.style.display = "block";
            let resultText = "";
            if (responseOk) {
              resultText = `File: ${data.filename || file.name}
`;
              resultText += `Model Used: ${data.model_used}
`;
              resultText += `Prediction: ${data.prediction}
`;
              resultText += `Confidence: ${(data.confidence * 100).toFixed(2)}%

`;
              resultText += `Raw Output:
${JSON.stringify(data.raw_model_output, null, 2)}`;
              elements.pre.style.color = "var(--text-color)";
            } else {
              let errorDetail = `Server Error: ${status} ${statusText}`;
              if (data && data.detail) {
                errorDetail += `
Details: ${
                  typeof data.detail === "string"
                    ? data.detail
                    : JSON.stringify(data.detail, null, 2)
                }`;
              } else if (result.value.error) {
                // Catch network error from the fetch.catch
                errorDetail = `Network Error: ${result.value.error.message}`;
              }
              resultText = errorDetail;
              elements.pre.style.color = "var(--error-color)";
            }
            elements.pre.textContent = resultText;
          } else {
            // promise was rejected
            // This case should ideally be less frequent due to .catch within the map
            console.error("Promise rejected:", result.reason);
            // Attempt to find which model it was if possible, though result.reason might not have modelName
            // For now, log and potentially display a generic error in a fallback area if we had one
            // Or find the modelName if the error object contains it (depends on how errors were structured)
            if (
              result.reason &&
              result.reason.modelName &&
              modelResultElements[result.reason.modelName]
            ) {
              const elements = modelResultElements[result.reason.modelName];
              elements.div.style.display = "block";
              elements.pre.textContent = `An unexpected error occurred for model ${result.reason.modelName}. Please check the console.`;
              elements.pre.style.color = "var(--error-color)";
            }
          }
        });
      } catch (error) {
        // This catch is for Promise.allSettled itself, which shouldn't happen.
        console.error("Error processing model predictions:", error);
        if (mainPlaceholder) {
          mainPlaceholder.textContent =
            "An unexpected error occurred while processing results. Please check the console.";
          mainPlaceholder.style.color = "var(--error-color)";
          mainPlaceholder.style.display = "block";
        }
      } finally {
        if (loaderContainer) loaderContainer.style.display = "none";
        submitButton.disabled = false;
        submitButton.textContent = "Detect Deepfake";
      }
    });
  } else {
    console.error(
      "Essential page elements (uploadForm, submitButton, resultsContainer, mainPlaceholder, or model result elements) not found. Script will not run correctly."
    );
    if (resultsContainer) {
      // If resultsContainer exists, but others might be missing
      const errorMsgDiv = document.createElement("div");
      errorMsgDiv.textContent =
        "Error: Page elements missing, app cannot function.";
      errorMsgDiv.style.color = "var(--error-color)";
      resultsContainer.appendChild(errorMsgDiv);
    }
  }
});
