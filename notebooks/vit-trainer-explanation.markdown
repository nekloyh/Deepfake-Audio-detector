# Vision Transformer (ViT) Trainer Notebook Explanation

This document provides a detailed explanation of the `vit_trainer.ipynb` Jupyter notebook, which implements a Vision Transformer (ViT) model for audio deepfake detection. The notebook processes spectrogram inputs derived from audio data and trains a ViT model to classify audio as "Real" or "Fake." Below, we describe the key functions, the overall workflow, expected results for each component, and comparisons to the CNN model from `cnn-trainer.ipynb`.

## Overview
The `vit_trainer.ipynb` notebook is designed to:
- Load and process spectrogram data from a preprocessed dataset.
- Define a Vision Transformer architecture with three configurations (Small, Medium, Large) targeting different parameter counts (~2-3M for Small, ~4-6M for Medium, ~10-12M for Large).
- Train the model using a specified dataset, log metrics to Weights & Biases (WandB), and evaluate performance on validation and test sets.
- Save the best model based on validation F1-score and generate a confusion matrix for visualization.

The notebook shares structural similarities with `cnn-trainer.ipynb` but replaces the CNN architecture with a ViT, which processes spectrograms as sequences of patches using transformer layers. Key differences include the model architecture, dataset configurations, and parameter tuning.

## Key Functions and Classes

### 1. Imports and Setup
**Purpose**: Initialize the environment by importing necessary libraries and setting random seeds for reproducibility.
- **Libraries**: Similar to the CNN notebook, includes PyTorch, NumPy, Pandas, Scikit-learn, TQDM, Matplotlib, Seaborn, and WandB. Additionally imports `einops` and `einops.layers.torch` for tensor rearrangement operations critical to ViT’s patch-based processing.
- **Random Seed**: Sets `torch.manual_seed(42)` and `np.random.seed(42)` for consistency, identical to the CNN notebook.
- **Comparison to CNN**: The addition of `einops` is unique to ViT, enabling efficient patch embedding and attention operations.
- **Expected Output**: No direct output, ensures a reproducible environment.

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
- **Purpose**: Authenticates with WandB for experiment tracking, identical to the CNN notebook.
- **Expected Output**: Prints a success message (`WandB login successful using wandb_api_key.`) or a fallback message. Outputs WandB logs, e.g., `wandb: Currently logged in as: nekloyh (nekloyh-none)`.
- **Comparison to CNN**: No differences; both notebooks use the same WandB login logic.

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
- **Purpose**: Defines audio processing parameters, including sample rate, spectrogram settings, SpecAugment augmentation, and dataset paths, identical to the CNN notebook.
- **Expected Output**: No direct output, serves as a configuration container.
- **Comparison to CNN**: Identical configuration, ensuring consistency in data preprocessing between CNN and ViT models.

### 4. ModelConfig Class
**Class**:
```python
class ModelConfig:
    def __init__(self, name, audio_length_seconds, overlap_ratio, apply_augmentation=False, apply_waveform_augmentation=False, patch_width=16):
        self.name = name
        self.audio_length_seconds = audio_length_seconds
        self.overlap_ratio = overlap_ratio
        self.apply_augmentation = apply_augmentation
        self.apply_waveform_augmentation = apply_waveform_augmentation
        frames = (audio_length_seconds * DataConfig.SR) / DataConfig.HOP_LENGTH
        self.max_frames_spec = int(np.ceil(frames / patch_width) * patch_width)
```
- **Purpose**: Configures dataset-specific parameters, including audio segment length, overlap ratio, and augmentation settings. Adjusts `max_frames_spec` to be divisible by `patch_width` (default 16) to match ViT’s patch-based processing.
- **Expected Output**: No direct output, defines dataset configurations in `ALL_MODEL_CONFIGS`.
- **Comparison to CNN**: Adds `patch_width` parameter and ensures `max_frames_spec` divisibility by `patch_width`, unlike the CNN’s simpler frame calculation. ViT datasets are:
  - `vit_balanced_dataset`: 8s segments, 0.5 overlap.
  - `vit_performance_dataset`: 10s segments, 0.0 overlap (no sliding window), differing from CNN’s 4s/0.75 overlap for performance dataset.

### 5. ViTConfig Class
**Class**:
```python
class ViTConfig:
    def __init__(self, name, image_size, patch_size, dim, depth, heads, mlp_dim, dropout=0.1, emb_dropout=0.1, channels=1, num_classes=2):
        self.name = name
        self.image_height, self.image_width = image_size
        self.patch_height, self.patch_width = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.channels = channels
        self.num_classes = num_classes
        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, ...
        self.num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        self.patch_dim = self.channels * self.patch_height * self.patch_width
```
- **Purpose**: Defines ViT model parameters, including input image size, patch size, embedding dimension, transformer depth, attention heads, MLP dimension, and dropout rates.
- **Expected Output**: No direct output, configures ViT architecture (Small, Medium, Large).
- **Comparison to CNN**: Replaces `CNNConfig`’s block and filter settings with transformer-specific parameters (e.g., `dim`, `depth`, `heads`). ViT uses fixed 16x16 patches, unlike CNN’s convolutional feature extraction.

### 6. ViT Architecture (AudioViT, PreNorm, FeedForward, Attention, Transformer)
**Classes**:
- **PreNorm**: Applies layer normalization before a function (attention or feed-forward).
- **FeedForward**: Implements a two-layer MLP with GELU activation and dropout.
- **Attention**: Computes multi-head self-attention using query, key, and value projections, with `einops` for tensor reshaping.
- **Transformer**: Stacks multiple layers of attention and feed-forward modules with residual connections.
- **AudioViT**:
```python
class AudioViT(nn.Module):
    def __init__(self, *, config: ViTConfig):
        super().__init__()
        self.config = config
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(config.patch_dim),
            nn.Linear(config.patch_dim, config.dim),
            nn.LayerNorm(config.dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, config.num_patches + 1, config.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
        self.dropout = nn.Dropout(config.emb_dropout)
        self.transformer = Transformer(config.dim, config.depth, config.heads, config.mlp_dim, config.dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.num_classes))
```
- **Purpose**: Defines a ViT that splits spectrograms into 16x16 patches, embeds them, adds a CLS token and positional embeddings, processes them through transformer layers, and outputs class logits via the CLS token.
- **Expected Output**: A model ready for training, with parameter counts of ~2-3M (Small), ~4-6M (Medium), and ~10-12M (Large).
- **Comparison to CNN**: Unlike CNN’s convolutional layers, ViT uses patch-based transformer processing, requiring fewer parameters but potentially higher memory due to attention mechanisms. CNN uses residual blocks, while ViT relies on attention and feed-forward layers.

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
- **Purpose**: Loads preprocessed spectrograms and labels from `.npy` files and `metadata.csv`, ensuring 3D shape (1, N_MELS, max_frames_spec) and converting to PyTorch tensors.
- **Expected Output**: Returns `(spectrogram, label)` tuples, with `None` for failed loads filtered by `custom_collate_fn`.
- **Comparison to CNN**: Identical to CNN’s `AudioDataset`, but spectrogram dimensions are adjusted in `ModelConfig` to be patch-divisible for ViT.

### 8. custom_collate_fn
**Function**:
```python
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
```
- **Purpose**: Filters `None` values from dataset errors and collates valid samples into batches.
- **Expected Output**: Batches of spectrograms and labels as tensors, or `None` for empty batches.
- **Comparison to CNN**: Identical to CNN’s implementation.

### 9. train_model
**Function**:
```python
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, run_name, dataset_name):
    ...
```
- **Purpose**: Trains the ViT model, performs forward/backward passes, evaluates on validation set, logs metrics (loss, accuracy, F1-score, ROC AUC) to WandB, and saves the best model based on validation F1-score.
- **Expected Output**:
  - Per-epoch console output, e.g., `Epoch 1 Train Loss: 0.XXXX, Val Loss: 0.XXXX, Val Acc: 0.XXXX, Val F1: 0.XXXX, Val ROC AUC: 0.XXXX`.
  - Saves best model to `best_vit_model_{run_name}.pth`.
  - WandB logs for metrics and checkpoints.
- **Comparison to CNN**: Nearly identical, except saves models with `vit` prefix and uses ViT-specific configurations.

### 10. evaluate_model
**Function**:
```python
def evaluate_model(model, data_loader, criterion, device, return_cm=False):
    ...
```
- **Purpose**: Evaluates the model, computing loss, predictions, labels, probabilities, and optionally a confusion matrix.
- **Expected Output**: Returns average loss, predictions, labels, probabilities, and confusion matrix (if `return_cm=True`) as NumPy arrays.
- **Comparison to CNN**: Identical to CNN’s implementation.

### 11. plot_confusion_matrix
**Function**:
```python
def plot_confusion_matrix(cm, labels=["Real", "Fake"], run_name="", save_dir="."):
    ...
```
- **Purpose**: Visualizes and saves a confusion matrix using Seaborn’s heatmap.
- **Expected Output**: Saves `confusion_matrix_{run_name}.png` in `results` directory and logs to WandB.
- **Comparison to CNN**: Identical to CNN’s implementation.

### 12. TrainingConfig Class
**Class**:
```python
class TrainingConfig:
    def __init__(self, model_size, dataset_name, epochs, learning_rate, batch_size, num_workers):
        ...
```
- **Purpose**: Configures training parameters with validation checks, supporting `ViT_Small`, `ViT_Medium`, `ViT_Large`.
- **Expected Output**: No direct output, passes parameters to `run_training`.
- **Comparison to CNN**: Similar, but validates ViT-specific model sizes instead of CNN sizes.

### 13. run_training
**Function**:
```python
def run_training(training_config):
    ...
```
- **Purpose**: Orchestrates training by setting up the ViT model, datasets, DataLoaders, optimizer, and loss function. Calls `train_model` and evaluates on the test set.
- **Expected Output**:
  - Console output for configuration, dataset sizes, and device.
  - Training progress via `train_model`.
  - Test set metrics, e.g., `Test Loss: 0.XXXX, Test Accuracy: 0.XXXX, Test F1-score: 0.XXXX, Test ROC AUC: 0.XXXX`.
  - Confusion matrix visualization and WandB logs.
- **Comparison to CNN**: Similar structure, but configures ViT models and uses ViT-specific dataset configurations. ViT models have lower parameter counts but may require more memory due to attention.

## Workflow
1. **Setup**:
   - Import libraries, set seeds, and authenticate with WandB.
2. **Configuration**:
   - Define `DataConfig`, `ModelConfig`, `ViTConfig`, and `TrainingConfig`.
3. **Data Loading**:
   - Load spectrograms and labels using `AudioDataset` and create DataLoaders with `custom_collate_fn`.
4. **Model Initialization**:
   - Initialize `AudioViT` with `ViTConfig`, Adam optimizer, and CrossEntropyLoss.
5. **Training**:
   - Train via `train_model`, log metrics to WandB, and save the best model.
6. **Evaluation**:
   - Evaluate on test set using `evaluate_model`, generate confusion matrix with `plot_confusion_matrix`, and log results to WandB.
7. **Execution**:
   - Set `training_params`, initialize `TrainingConfig`, and call `run_training`.

## Expected Results
- **Dataset Sizes**: Depend on `vit_balanced_dataset` preprocessing, but expect similar scale to CNN (~61k train, ~13k val, ~13k test for 8s segments).
- **Training Metrics**: Per-epoch logs, e.g., `Epoch 1 Train Loss: 0.XXXX, Val Loss: 0.XXXX, ...`.
- **Test Metrics**: Final test results, e.g., `Test Loss: 0.XXXX, Test Accuracy: 0.XXXX, Test F1-score: 0.XXXX, Test ROC AUC: 0.XXXX`.
- **Artifacts**:
  - Best model saved as `best_vit_model_{run_name}.pth`.
  - Confusion matrix saved as `results/confusion_matrix_{run_name}.png`.
- **WandB Logs**: Metrics, model checkpoints, and confusion matrix visualizations in the `audio-deepfake-detection` project.
- **Comparison to CNN**: ViT may achieve comparable or better performance with fewer parameters but requires more memory and potentially longer training due to attention computations.

## Notes
- Assumes a preprocessed dataset at `F:\Deepfake-Audio-Detector\processed_dataset\vit_balanced_dataset`.
- `ViT_Small` is default (batch size 32, ~2-3M parameters). For `ViT_Medium` or `ViT_Large`, reduce batch size (e.g., 16 or 8) to fit VRAM.
- Runs on CUDA if available, otherwise CPU.
- ViT’s patch-based approach may be more sensitive to spectrogram dimensions and preprocessing compared to CNN’s convolutional flexibility.

This Markdown file provides a comprehensive guide to the `vit_trainer.ipynb` notebook, its functions, expected outcomes, and comparisons to the CNN model, aiding in usage and debugging.