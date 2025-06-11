document.addEventListener("DOMContentLoaded", function () {
  const uploadForm = document.getElementById("uploadForm");
  const audioFile = document.getElementById("audioFile");
  const modelSelector = document.getElementById("modelSelector");
  const resultsOutput = document.getElementById("resultsOutput");
  const submitButton = uploadForm.querySelector(".submit-button");
  const loaderContainer = document.getElementById("loaderContainer");

  if (uploadForm) {
      uploadForm.addEventListener("submit", async function (event) {
          event.preventDefault();

          if (!audioFile.files || !audioFile.files.length) {
              resultsOutput.textContent = "Please select an audio file.";
              resultsOutput.style.color = "var(--error-color)";
              return;
          }

          // Show loader and disable button
          resultsOutput.textContent = ""; // Clear previous results or placeholder
          resultsOutput.style.display = "none"; // Hide text output area
          loaderContainer.style.display = "block";
          submitButton.disabled = true;
          submitButton.textContent = "Processing...";

          const formData = new FormData();
          formData.append("file", audioFile.files[0]);
          // Get the selected model name from the dropdown
          const selectedModel = modelSelector.value;

          try {
              // Construct the URL with the model_name as a query parameter
              const predictUrl = `/predict_audio?model_name=${encodeURIComponent(selectedModel)}`;
              
              const response = await fetch(predictUrl, {
                  method: "POST",
                  body: formData,
              });

              resultsOutput.style.display = "block"; // Show text output area again
              let resultText = "";
              let isError = false;

              if (response.ok) {
                  const data = await response.json();
                  // Format the JSON for better readability
                  resultText = `File: ${data.filename}
`;
                  resultText += `Model Used: ${data.model_used}
`;
                  resultText += `Prediction: ${data.prediction}
`;
                  resultText += `Confidence: ${(data.confidence * 100).toFixed(2)}%

`;
                  resultText += `Raw Output:
${JSON.stringify(data.raw_model_output, null, 2)}`;
                  resultsOutput.style.color = "var(--text-color)";
              } else {
                  isError = true;
                  let errorDetail = `Server Error: ${response.status} ${response.statusText}`;
                  try {
                      const errorData = await response.json();
                      errorDetail += `
Details: ${JSON.stringify(errorData.detail || errorData, null, 2)}`;
                  } catch (e) {
                      // If error response is not JSON, try to read as plain text
                      try {
                          const plainErrorText = await response.text();
                          errorDetail += `
${plainErrorText}`;
                      } catch (e_text) {
                          errorDetail += "Could not retrieve detailed error message from server.";
                      }
                  }
                  resultText = errorDetail;
                  resultsOutput.style.color = "var(--error-color)";
              }
              resultsOutput.textContent = resultText;

          } catch (error) {
              console.error("Error submitting file:", error);
              resultsOutput.style.display = "block"; // Show text output area
              resultsOutput.textContent = "An unexpected error occurred while submitting the file. Please check the console for more details.";
              resultsOutput.style.color = "var(--error-color)";
          } finally {
              // Hide loader and re-enable button
              loaderContainer.style.display = "none";
              submitButton.disabled = false;
              submitButton.textContent = "Detect Deepfake";
          }
      });
  } else {
      console.error("Upload form not found.");
      if (resultsOutput) {
          resultsOutput.textContent = "Error: Upload form element not found in HTML.";
          resultsOutput.style.color = "var(--error-color)";
      }
  }
});
