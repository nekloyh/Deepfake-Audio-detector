import torch
import torch.nn as nn
from dataclasses import dataclass, field


# Define Config class (copied from models/convert_cnn_small_to_onnx.py)
@dataclass
class Config:
    # Data processing parameters
    SEED: int = 42
    SR: int = 16000
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    N_MELS: int = 128
    FMIN: float = 0.0
    FMAX: float = 8000.0
    NUM_TIME_MASKS: int = 2
    NUM_FREQ_MASKS: int = 2
    TIME_MASK_MAX_WIDTH: int = 30
    FREQ_MASK_MAX_WIDTH: int = 15
    MASK_REPLACEMENT_VALUE: float = -80.0
    NORM_EPSILON: float = 1e-6
    LOUDNESS_LUFS: float = -23.0
    USE_GLOBAL_NORMALIZATION: bool = True
    USE_RANDOM_CROPPING: bool = True
    CACHE_DIR_BASE: str = "/kaggle/input/cnn-3s-dataset"
    DATASET_SUBDIR: str = "cnn_3s_dataset"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    metadata_file: str = "kaggle_metadata.csv"

    # Model architecture
    img_size: int = 224
    num_classes: int = 2
    in_channels: int = 1
    dropout: float = 0.1

    # CNN specific architecture parameters
    cnn_conv_channels: list[int] = field(default_factory=list)
    cnn_pool_after_conv: list[bool] = field(default_factory=list)
    linear_output_units_1st_fc: int = 512

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 20
    weight_decay: float = 1e-4
    num_workers: int = 4

    # Data augmentation
    apply_augmentation: bool = True
    augmentation_prob: float = 0.5
    audio_length_seconds: float = 3.0
    overlap_ratio: float = 0.5

    model_size: str = ""
    dataset_name: str = ""

    def validate(self):
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert len(self.cnn_conv_channels) == len(self.cnn_pool_after_conv), (
            "cnn_conv_channels and cnn_pool_after_conv must have the same length"
        )


# Define CNN_Audio class (copied from models/convert_cnn_small_to_onnx.py)
class CNN_Audio(torch.nn.Module):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: int,
        linear_output_units_1st_fc: int,
        cnn_conv_channels: list[int],
        cnn_pool_after_conv: list[bool],
        dropout: float = 0.1,
    ):
        super(CNN_Audio, self).__init__()
        self.in_channels = in_channels
        self.cnn_conv_channels = cnn_conv_channels
        self.cnn_pool_after_conv = cnn_pool_after_conv
        self.img_size = img_size
        self.dropout = dropout
        self.num_classes = num_classes

        layers = []
        in_dim = self.in_channels

        for i, out_dim in enumerate(self.cnn_conv_channels):
            layers.append(
                torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
            )
            layers.append(torch.nn.BatchNorm2d(out_dim))
            layers.append(torch.nn.ReLU(inplace=True))

            if self.cnn_pool_after_conv[i]:
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

            layers.append(torch.nn.Dropout2d(self.dropout))
            in_dim = out_dim

        self.conv_layers = torch.nn.Sequential(*layers)

        # Calculate output size of conv layers dynamically
        with torch.no_grad():
            dummy_input_for_init = torch.randn(
                1, self.in_channels, self.img_size, self.img_size
            )
            dummy_output_conv = self.conv_layers(dummy_input_for_init)
            # self.flattened_size = dummy_output_conv.view(1, -1).size(1) # Not used directly

        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))

        with torch.no_grad():
            dummy_pooled = self.adaptive_pool(dummy_output_conv)
            self.pooled_size = dummy_pooled.view(1, -1).size(1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.pooled_size, linear_output_units_1st_fc),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                linear_output_units_1st_fc, linear_output_units_1st_fc // 2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(linear_output_units_1st_fc // 2, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
