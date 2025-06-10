document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const audioFile = document.getElementById('audioFile');
    const predictButton = document.getElementById('predictButton');
    const resultsDiv = document.getElementById('results');
    const loadingIndicator = document.getElementById('loadingIndicator');

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        resultsDiv.innerHTML = '<p>Submit an audio file to see predictions.</p>'; // Clear previous results

        if (!audioFile.files || audioFile.files.length === 0) {
            resultsDiv.innerHTML = '<p style="color: red;">Please select an audio file first.</p>';
            return;
        }

        const formData = new FormData();
        formData.append('file', audioFile.files[0]);

        predictButton.disabled = true;
        loadingIndicator.style.display = 'block';
        resultsDiv.innerHTML = ''; // Clear previous results or messages

        try {
            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                // Placeholder: Displaying raw JSON for now
                // Actual display logic will be more sophisticated later
                resultsDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            } else {
                const errorData = await response.json();
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${errorData.detail || response.statusText}</p>`;
            }
        } catch (error) {
            console.error('Error during prediction:', error);
            resultsDiv.innerHTML = `<p style="color: red;">An unexpected error occurred. Please check the console.</p>`;
        } finally {
            predictButton.disabled = false;
            loadingIndicator.style.display = 'none';
        }
    });
});
