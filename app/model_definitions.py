import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from dataclasses import field  # dataclass is no longer used


# Define CNN_Audio class (copied from models/convert_cnn_small_to_onnx.py)
class CNN_Audio(nn.Module):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: int,
        linear_output_units_1st_fc: int,
        cnn_conv_channels: list[int],
        cnn_pool_after_conv: list[bool],
        dropout: float = 0.3,
    ):
        super(CNN_Audio, self).__init__()
        self.in_channels = in_channels
        self.cnn_conv_channels = cnn_conv_channels
        self.cnn_pool_after_conv = cnn_pool_after_conv
        self.img_size = img_size
        self.dropout = dropout
        self.num_classes = num_classes

        # Build convolutional layers with proper architecture
        layers = []
        in_dim = self.in_channels

        for i, out_dim in enumerate(self.cnn_conv_channels):
            # Convolutional block
            layers.append(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.ReLU(inplace=True))

            # Optional pooling
            if self.cnn_pool_after_conv[i]:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            # Dropout for regularization
            layers.append(nn.Dropout2d(self.dropout))
            in_dim = out_dim

        self.conv_layers = nn.Sequential(*layers)

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, self.in_channels, self.img_size, self.img_size)
            dummy_output = self.conv_layers(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

        # Adaptive average pooling to reduce feature map size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate size after adaptive pooling
        with torch.no_grad():
            dummy_pooled = self.adaptive_pool(dummy_output)
            self.pooled_size = dummy_pooled.view(1, -1).size(1)

        # Classifier with proper architecture
        self.classifier = nn.Sequential(
            nn.Linear(self.pooled_size, linear_output_units_1st_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(linear_output_units_1st_fc, linear_output_units_1st_fc // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(linear_output_units_1st_fc // 2, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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
    ):  # THÊM dropout VÀO ĐÂY
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
            dropout=dropout,  # TRUYỀN dropout VÀO ĐÂY
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
        x = x + self.pos_embed  # Positional embedding
        x = self.transformer(
            x
        )  # Chú ý rằng PyTorch's TransformerEncoderLayer/Encoder tự xử lý dropout nội bộ

        cls_token_final = x[:, 0]
        x = self.ln(cls_token_final)
        return self.mlp_head(x)
