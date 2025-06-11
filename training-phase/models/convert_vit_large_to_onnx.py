import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from dataclasses import dataclass
import os


# Define the Config class from the notebook
@dataclass
class Config:
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
    CACHE_DIR_BASE: str = "/kaggle/input/vit-3s-dataset"
    DATASET_SUBDIR: str = "vit_3s_dataset"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    metadata_file: str = "kaggle_metadata.csv"
    img_size: int = 224
    patch_size: int = 16
    num_classes: int = 2
    in_channels: int = 1
    dim: int = 384
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 768
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 20
    weight_decay: float = 1e-4
    num_workers: int = 4
    apply_augmentation: bool = True
    augmentation_prob: float = 0.5
    audio_length_seconds: float = 3.0
    overlap_ratio: float = 0.5
    model_size: str = ""
    dataset_name: str = ""

    def validate(self):
        assert self.img_size % self.patch_size == 0, (
            "img_size must be divisible by patch_size"
        )
        assert self.dim % self.heads == 0, "dim must be divisible by heads"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"

    def get_full_cache_dir(self):
        return os.path.join(self.CACHE_DIR_BASE, self.DATASET_SUBDIR)


# Define the ViT_Audio model from the notebook
class ViT_Audio(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_classes,
        in_channels,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert img_size % patch_size == 0, (
            "Image dimensions must be divisible by the patch size."
        )
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels, patch_dim, kernel_size=patch_size, stride=patch_size
            ),
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        self.ln = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        cls_token_final = x[:, 0]
        x = self.ln(cls_token_final)
        return self.mlp_head(x)


def convert_to_onnx(model, config, checkpoint_path, output_path):
    # Set model to evaluation mode
    model.eval()

    # Create a dummy input tensor with the correct shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, config.in_channels, config.img_size, config.img_size)

    # Move model and dummy input to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = dummy_input.to(device)

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model successfully exported to {output_path}")


def main():
    # Define ViT_Large configuration
    config = Config(
        img_size=224,
        patch_size=16,
        num_classes=2,
        in_channels=1,
        dim=384,
        depth=6,
        heads=8,
        mlp_dim=768,
        dropout=0.1,
        model_size="ViT_Large",
        dataset_name="vit_3s_dataset",
    )

    # Initialize the model
    model = ViT_Audio(
        img_size=config.img_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
    )

    # Load the trained weights
    checkpoint_path = "best_model_ViT_Large.pth"  # Update this path if necessary
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Output path for the ONNX model
    output_path = "vit_large.onnx"

    # Convert to ONNX
    convert_to_onnx(model, config, checkpoint_path, output_path)


if __name__ == "__main__":
    main()
