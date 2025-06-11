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
    # Pydantic requires a pure boolean value here, without inline comments.
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model settings
    MODEL_DIR: str = "app/models_onnx"  # Default path, relative to project root
    CNN_SMALL_MODEL_NAME: str = "CNN_Small.onnx"
    CNN_LARGE_MODEL_NAME: str = "CNN_Large.onnx"

    # Keep these commented out or assign valid ONNX filenames if you are using them.
    # Otherwise, the model loading logic might try to find non-existent files.
    # VIT_SMALL_MODEL_NAME: str = "ViT_Small.onnx"
    # VIT_LARGE_MODEL_NAME: str = "ViT_Large.onnx"

    # Audio processing settings (keeping these as fixed defaults for now, can be env vars if needed)
    TARGET_SAMPLE_RATE: int = 16000
    CHUNK_DURATION_SECONDS: float = 3.0
    N_MELS: int = 224
    SPECTROGRAM_WIDTH: int = 224
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    MIN_DB_LEVEL: float = -80.0
    TOP_DB: float = 80.0

    # Labels
    LABELS: dict = {0: "real", 1: "fake"}
    REAL_LABEL_INDEX: int = 0
    FAKE_LABEL_INDEX: int = 1

    class Config:
        env_file = (
            ".env"  # Specifies the .env file to load (pydantic-settings specific)
        )
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
