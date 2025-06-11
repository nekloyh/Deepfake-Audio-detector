document.addEventListener("DOMContentLoaded", function () {
  const uploadForm = document.getElementById("uploadForm");
  const audioFile = document.getElementById("audioFile");
  const resultsOutput = document.getElementById("resultsOutput");

  if (uploadForm) {
    uploadForm.addEventListener("submit", async function (event) {
      event.preventDefault();

      resultsOutput.textContent = "Processing...";

      if (!audioFile.files || !audioFile.files.length) {
        resultsOutput.textContent = "Please select an audio file.";
        return;
      }

      const formData = new FormData();
      formData.append("file", audioFile.files[0]);

      try {
        const response = await fetch("/predict/", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          resultsOutput.textContent = JSON.stringify(data, null, 2);
        } else {
          let errorText = `Error: ${response.status} ${response.statusText}`;
          try {
            // Try to parse error response as JSON, it might contain more details
            const errorData = await response.json();
            errorText += `\n${JSON.stringify(errorData, null, 2)}`;
          } catch (e) {
            // If error response is not JSON, try to read as plain text
            try {
              const plainErrorText = await response.text();
              errorText += `\n${plainErrorText}`;
            } catch (e_text) {
              // Fallback if reading as text also fails
              errorText +=
                "\nCould not retrieve detailed error message from server.";
            }
          }
          resultsOutput.textContent = errorText;
        }
      } catch (error) {
        console.error("Error submitting file:", error);
        resultsOutput.textContent =
          "An error occurred while submitting the file. Check the console for details.";
      }
    });
  } else {
    console.error("Upload form not found.");
    if (resultsOutput)
      resultsOutput.textContent =
        "Error: Upload form element not found in HTML.";
  }
});
