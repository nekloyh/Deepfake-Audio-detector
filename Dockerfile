# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by libs like librosa/soundfile
# (e.g., libsndfile1). This can vary based on the base image and specific needs.
# For python:3.9-slim, libsndfile1 is often needed.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app ./app

# Copy ONNX models into the image.
# This assumes models are in app/models/ as per current structure.
# If MODEL_DIR is configurable and set to a path outside /app, adjust accordingly
# or use volumes. For bundling, this path is fine.
COPY ./app/models ./app/models

# Expose the port the app runs on
# This should match the port Uvicorn runs on, fetched from settings.PORT (default 8000)
EXPOSE 8000 

# Define the command to run the application
# Uvicorn will listen on the host and port defined in settings (via .env or defaults)
# The CMD here uses "0.0.0.0" and "8000" directly. If settings.HOST/PORT are different
# and not passed as ENV to container, this might mismatch. However, 0.0.0.0:8000 is standard for containers.
# The config.py defaults to 0.0.0.0 and 8000, so this is consistent.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
