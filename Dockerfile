# 1. Base Image
FROM python:3.10-slim

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 3. Working Directory
WORKDIR /app

# 4. Install System Dependencies
# libsndfile1 for SoundFile/Librosa, ffmpeg for torchaudio/librosa broader format support
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Copy requirements.txt and Install Python Dependencies
# This is done before copying app code to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
# Copy web application
COPY webapp/ ./webapp/
# Copy ONNX models (assuming they are present in the build context)
COPY onnx_models/ ./onnx_models/
# Copy model definitions (potentially needed by convert_to_onnx.py or if models have complex structures)
COPY demo_workflow/models/ ./demo_workflow/models/
# Copy the ONNX conversion script
COPY convert_to_onnx.py .

# 7. Expose Port
EXPOSE 8000

# 8. Default Command
# Runs the FastAPI application using Uvicorn
CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
