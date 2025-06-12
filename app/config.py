from typing import List, Literal
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Explicitly load .env file at the beginning.
# find_dotenv will search for .env in current directory or parent directories.
# Alternatively, you can specify path to .env file: load_dotenv(dotenv_path=".env")
# If .env is in the same directory as this config.py, it should be found.
# For dockerized environments, env vars are usually passed directly, not via .env file.
# load_dotenv() will not override existing environment variables.
load_dotenv()


class Settings(BaseSettings):
    APP_NAME: str = "Deepfake Audio Detector"
    # Set to True for enabling uvicorn reload and debugging features.

    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model settings
    MODEL_DIR: str = "app/models/"  # Default path, relative to project root
    CNN_SMALL_MODEL_NAME: str = "best_model_CNN_Small_cnn_3s_dataset_102208.pth"
    CNN_LARGE_MODEL_NAME: str = "best_model_CNN_Large_cnn_3s_dataset_114040.pth"

    # Keep these commented out or assign valid ONNX filenames if you are using them.
    # Otherwise, the model loading logic might try to find non-existent files.
    VIT_SMALL_MODEL_NAME: str = "best_model_ViT_Small_vit_3s_dataset_040441.pth"
    VIT_LARGE_MODEL_NAME: str = "best_model_ViT_Large_vit_3s_dataset_044740.pth"

    # Audio processing settings (keeping these as fixed defaults for now, can be env vars if needed)
    TARGET_SAMPLE_RATE: int = 16000
    CHUNK_DURATION_SECONDS: float = 3.0
    N_MELS: int = 128
    SPECTROGRAM_WIDTH: int = 224
    N_FFT: int = 2048  # FFT window size for STFT.
    HOP_LENGTH: int = 512  # Hop length for STFT (number of samples between frames).
    MIN_DB_LEVEL: float = -80.0  # Minimum decibel level; used as the floor for dB spectrograms before normalization.
    LOUDNESS_LUFS: float = -23.0  # Target loudness (LUFS)
    # Spectrogram values below this are clipped to this level.
    TOP_DB: float = 80.0  # Used with librosa.power_to_db (ref=np.max, top_db=TOP_DB). Values quieter than
    # (max_signal_power - TOP_DB) are floored. Effectively defines the dynamic range.

    # New parameters
    SEGMENT_OVERLAP_SECONDS: float = 1.5
    IMAGE_SIZE: int = 224
    PIXEL_MEAN: float =  -0.0137
    PIXEL_STD: float = 0.7317

    # Labels
    LABELS: dict = {0: "real", 1: "fake"}
    REAL_LABEL_INDEX: int = 0
    FAKE_LABEL_INDEX: int = 1
    REAL_BIAS_FACTOR: float = 1.1 # Hệ số nhân cho xác suất REAL (>1.0 = thiên vị REAL, <1.0 = thiên vị FAKE)
    BIAS_METHOD: str = "real_bias" # "equal_weight", "real_bias", "inverse_freq", "custom"
    # Aggregate methods
    # "mean_logits", "mean_probs", "majority_vote", "weighted_majority_vote", "logit_argmax_vote"
    AGGREGATION_METHOD: str = "mean_probs"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Pydantic-settings will automatically attempt to load variables from .env
        # and then from actual environment variables, with environment variables taking precedence.
        # It also handles type casting.
        # Use `extra = 'ignore'` if you don't want errors for undefined env vars in .env
        # extra = 'ignore'


settings = Settings()

# Example of how to use in other modules:
# from app.config import settings
# print(settings.APP_NAME)
# print(settings.HOST)

# print(settings.TARGET_SAMPLE_RATE)
