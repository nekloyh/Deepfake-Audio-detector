# CNN Trainer Notebook Explanation

This document provides a detailed explanation of the `cnn-trainer.ipynb` Jupyter notebook, which implements a ResNet-like Convolutional Neural Network (CNN) for audio deepfake detection. The notebook processes spectrogram inputs derived from audio data and trains a CNN model to classify audio as "Real" or "Fake." Below, we describe the key functions, the overall workflow, and the expected results for each component.

## Overview
The notebook is designed to:
- Load and process spectrogram data from a preprocessed dataset.
- Define a ResNet-like CNN architecture with three configurations (Small, Medium, Large) targeting different parameter counts (~5M, 10-12M, 25-30M).
- Train the model using a specified dataset, log metrics to Weights & Biases (WandB), and evaluate performance on validation and test sets.
- Save the best model based on validation F1-score and generate a confusion matrix for visualization.

The notebook is structured into several cells, each handling specific tasks such as imports, configurations, model definition, dataset handling, training, evaluation, and execution.

## Key Functions and Classes

### 1. Imports and Setup
**Purpose**: Initialize the environment by importing necessary libraries and setting random seeds for reproducibility.
- **Libraries**: Includes PyTorch for model building and training, NumPy and Pandas for data handling, Scikit-learn for metrics, TQDM for progress bars, Matplotlib and Seaborn for visualization, and WandB for experiment tracking.
- **Random Seed**: Sets `torch.manual_seed(42)` and `np.random.seed(42)` to ensure consistent results.
- **Expected Output**: No direct output, but ensures a reproducible environment.

### 2. WandB Login
**Function**: 
```python
try:
    with open("wandb_api_key.txt", "r") as file:
        wandb_api_key = file.read().strip()
    wandb.login(key=wandb_api_key)
    print("WandB login successful using wandb_api_key.")
except Exception as e:
    print(f"Failed to login to WandB: {e}. Falling back to manual login.")
    wandb.login()
```
- **Purpose**: Authenticates with WandB for experiment tracking. Attempts to read an API key from a file (`wandb_api_key.txt`) and falls back to manual login if the file is unavailable.
- **Expected Output**: Prints a success message (`WandB login successful using wandb_api_key.`) or a fallback message if the file-based login fails. Outputs WandB logs indicating login status, e.g., `wandb: Currently logged in as: nekloyh (nekloyh-none)`.

### 3. DataConfig Class
**Class**: 
```python
class DataConfig:
    SEED = 42
    SR = 16000
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 0.0
    FMAX = 8000.0
    NUM_TIME_MASKS = 2
    NUM_FREQ_MASKS = 2
    TIME_MASK_MAX_WIDTH = 60
    FREQ_MASK_MAX_WIDTH = 25
    MASK_REPLACEMENT_VALUE = -80.0
    NORM_EPSILON = 1e-6
    LOUDNESS_LUFS = -23.0
    USE_GLOBAL_NORMALIZATION = False
    USE_RANDOM_CROPPING = True
    CACHE_DIR = "F:\\Deepfake-Audio-Detector\\processed_dataset"
```
- **Purpose**: Defines parameters for audio data processing, including sample rate, spectrogram settings (e.g., FFT window, Mel bands), augmentation settings (SpecAugment), and dataset paths.
- **Expected Output**: No direct output, as it serves as a configuration container. These parameters are used by the dataset and preprocessing pipeline.

### 4. ModelConfig Class
**Class**:
```python
class ModelConfig:
    def __init__(self, name, audio_length_seconds, overlap_ratio, apply_augmentation=False, apply_waveform_augmentation=False):
        self.name = name
        self.audio_length_seconds = audio_length_seconds
        self.overlap_ratio = overlap_ratio
        self.apply_augmentation = apply_augmentation
        self.apply_waveform_augmentation = apply_waveform_augmentation
        frames = (audio_length_seconds * DataConfig.SR) / DataConfig.HOP_LENGTH
        self.max_frames_spec = int(np.ceil(frames))
```
- **Purpose**: Configures dataset-specific parameters, such as audio segment length and overlap ratio for splitting audio into chunks. Calculates `max_frames_spec` for spectrogram dimensions.
- **Expected Output**: No direct output. Used to define dataset configurations in `ALL_MODEL_CONFIGS`, e.g., `cnn_balanced_dataset` (8s segments, 0.5 overlap) and `cnn_performance_dataset` (4s segments, 0.75 overlap).

### 5. CNNConfig Class
**Class**:
```python
class CNNConfig:
    def __init__(self, name, image_size, channels=1, num_classes=2, blocks=[3, 4, 6, 3], filters=[64, 128, 256, 512], bottleneck=False):
        self.name = name
        self.image_height, self.image_width = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.blocks = blocks
        self.filters = filters
        self.bottleneck = bottleneck
        assert len(blocks) == len(filters), ...
        assert all(f > 0 for f in filters), ...
        assert all(b >= 0 for b in blocks), ...
```
- **Purpose**: Defines CNN model parameters, including input image size, number of classes (2 for Real/Fake), block counts per stage, filter sizes, and whether to use bottleneck blocks (for Medium and Large models).
- **Expected Output**: No direct output. Used to configure the CNN architecture (e.g., `CNN_Small` with BasicBlock, `CNN_Medium` and `CNN_Large` with BottleneckBlock).

### 6. CNN Architecture (AudioCNN, BasicBlock, BottleneckBlock)
**Classes**:
- **BasicBlock**: Implements a standard ResNet block with two 3x3 convolutions, batch normalization, and ReLU. Used for `CNN_Small`.
- **BottleneckBlock**: Implements a ResNet bottleneck block with 1x1, 3x3, and 1x1 convolutions for parameter efficiency. Used for `CNN_Medium` and `CNN_Large`.
- **AudioCNN**:
```python
class AudioCNN(nn.Module):
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        block = BottleneckBlock if config.bottleneck else BasicBlock
        self.in_channels = 64
        self.conv1 = nn.Conv2d(config.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = []
        for num_blocks, out_channels in zip(config.blocks, config.filters):
            layers.append(self._make_layer(block, out_channels, num_blocks))
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(config.filters[-1] * block.expansion, config.num_classes)
```
- **Purpose**: Defines a ResNet-like CNN with an initial 7x7 convolution, max pooling, four stages of residual blocks, adaptive average pooling, and a final fully connected layer for classification.
- **Expected Output**: A model ready for training, with ~5M parameters for `CNN_Small`, ~10-12M for `CNN_Medium`, and ~25-30M for `CNN_Large`.

### 7. AudioDataset Class
**Class**:
```python
class AudioDataset(Dataset):
    def __init__(self, cache_dir, set_type, n_mels, max_frames_spec):
        ...
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        ...
```
- **Purpose**: Loads preprocessed spectrogram data from `.npy` files and corresponding labels from a `metadata.csv` file. Ensures spectrograms are 3D (1, N_MELS, max_frames_spec) and converts them to PyTorch tensors.
- **Expected Output**: Returns tuples of `(spectrogram, label)` for each data sample. If a file fails to load, returns `None`, which is filtered by `custom_collate_fn`.

### 8. custom_collate_fn
**Function**:
```python
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
```
- **Purpose**: Filters out `None` values from the dataset (due to loading errors) and collates valid samples into batches for the DataLoader.
- **Expected Output**: A batch of spectrograms and labels as PyTorch tensors, or `None` if the batch is empty.

### 9. train_model
**Function**:
```python
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, run_name, dataset_name):
    ...
```
- **Purpose**: Trains the CNN model for a specified number of epochs. Performs forward and backward passes, updates model weights, evaluates on the validation set, and logs metrics (loss, accuracy, F1-score, ROC AUC) to WandB. Saves the best model based on validation F1-score.
- **Expected Output**:
  - Console output per epoch, e.g., `Epoch 1 Train Loss: 0.XXXX, Val Loss: 0.XXXX, Val Acc: 0.XXXX, Val F1: 0.XXXX, Val ROC AUC: 0.XXXX`.
  - Saves the best model to `best_cnn_model_{run_name}.pth` when a new best validation F1-score is achieved.
  - WandB logs for metrics and model checkpoints.

### 10. evaluate_model
**Function**:
```python
def evaluate_model(model, data_loader, criterion, device, return_cm=False):
    ...
```
- **Purpose**: Evaluates the model on a dataset (validation or test), computing loss, predictions, labels, probabilities, and optionally a confusion matrix.
- **Expected Output**: Returns average loss, predictions, true labels, and probabilities as NumPy arrays. If `return_cm=True`, also returns the confusion matrix.

### 11. plot_confusion_matrix
**Function**:
```python
def plot_confusion_matrix(cm, labels=["Real", "Fake"], run_name="", save_dir="."):
    ...
```
- **Purpose**: Creates and saves a confusion matrix visualization using Seaborn’s heatmap.
- **Expected Output**: A PNG file (`confusion_matrix_{run_name}.png`) in the `results` directory and a WandB log of the image.

### 12. TrainingConfig Class
**Class**:
```python
class TrainingConfig:
    def __init__(self, model_size, dataset_name, epochs, learning_rate, batch_size, num_workers):
        ...
```
- **Purpose**: Configures training parameters (model size, dataset, epochs, learning rate, batch size, num_workers) with validation checks.
- **Expected Output**: No direct output, used to pass parameters to `run_training`.

### 13. run_training
**Function**:
```python
def run_training(training_config):
    ...
```
- **Purpose**: Orchestrates the training process by setting up the model, datasets, DataLoaders, optimizer, and loss function. Calls `train_model` and evaluates the final model on the test set.
- **Expected Output**:
  - Console output summarizing configuration, dataset sizes, and device.
  - Training progress and metrics via `train_model`.
  - Test set evaluation results, e.g., `Test Loss: 0.XXXX, Test Accuracy: 0.XXXX, Test F1-score: 0.XXXX, Test ROC AUC: 0.XXXX`.
  - Confusion matrix visualization and WandB logs.

## Workflow
1. **Setup**:
   - Import libraries and set random seeds.
   - Authenticate with WandB for experiment tracking.

2. **Configuration**:
   - Define `DataConfig` for audio processing parameters.
   - Define `ModelConfig` for dataset-specific settings (e.g., segment length).
   - Define `CNNConfig` for model architecture (Small, Medium, Large).
   - Define `TrainingConfig` with training parameters (e.g., batch size, epochs).

3. **Data Loading**:
   - Use `AudioDataset` to load preprocessed spectrograms and labels from the cache directory.
   - Create DataLoaders with `custom_collate_fn` for training, validation, and test sets.

4. **Model Initialization**:
   - Initialize `AudioCNN` with the appropriate `CNNConfig` based on the model size.
   - Set up the Adam optimizer and CrossEntropyLoss.

5. **Training**:
   - Run `train_model` to train the model, log metrics to WandB, and save the best model based on validation F1-score.

6. **Evaluation**:
   - Evaluate the trained model on the test set using `evaluate_model`.
   - Generate and save a confusion matrix using `plot_confusion_matrix`.
   - Log test metrics and confusion matrix to WandB.

7. **Execution**:
   - The final cell sets up `training_params`, initializes `TrainingConfig`, and calls `run_training`.

## Expected Results
- **Dataset Sizes** (from the notebook output):
  - Train: 61,022 samples
  - Validation: 13,205 samples
  - Test: 13,650 samples
- **Training Metrics**: Per-epoch logs of train/validation loss, accuracy, F1-score, and ROC AUC, e.g., `Epoch 1 Train Loss: 0.XXXX, Val Loss: 0.XXXX, ...`.
- **Test Metrics**: Final test set metrics, e.g., `Test Loss: 0.XXXX, Test Accuracy: 0.XXXX, Test F1-score: 0.XXXX, Test ROC AUC: 0.XXXX`.
- **Artifacts**:
  - Best model saved as `best_cnn_model_{run_name}.pth`.
  - Confusion matrix saved as `results/confusion_matrix_{run_name}.png`.
- **WandB Logs**: Metrics, model checkpoints, and confusion matrix visualizations logged to the `audio-deepfake-detection` project.

## Notes
- The notebook assumes a preprocessed dataset exists at `F:\Deepfake-Audio-Detector\processed_dataset\cnn_balanced_dataset` with `metadata.csv` and `.npy` files.
- The `CNN_Small` configuration is used by default (batch size 32, ~5M parameters). For `CNN_Medium` or `CNN_Large`, adjust `model_size` and reduce `batch_size` (e.g., 16 or 8) to fit VRAM constraints.
- The notebook is designed to run on a CUDA-enabled GPU, falling back to CPU if unavailable.

This Markdown file provides a comprehensive guide to understanding the `cnn-trainer.ipynb` notebook, its functions, and the expected outcomes, facilitating both usage and debugging.