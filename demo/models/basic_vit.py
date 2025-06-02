# scripts/models/basic_vit.py

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification


class BasicViT(nn.Module):
    def __init__(self, num_classes=2, n_mels=128, target_spec_width=256):
        super(BasicViT, self).__init__()

        # Cấu hình ViT rất cơ bản (Tiny/Small-like)
        # image_size: kích thước của spectrogram (height, width)
        # patch_size: kích thước của các "patch" mà ViT chia ảnh ra. Phải chia hết image_size.
        # hidden_size: kích thước của embedding vector cho mỗi patch.
        # num_hidden_layers: số lượng lớp Transformer Encoder.
        # num_attention_heads: số lượng attention heads trong mỗi lớp.
        # num_channels: số kênh của input image (1 cho spectrogram).

        # Ví dụ cấu hình một ViT-Tiny/Small từ scratch
        # Đảm bảo patch_size chia hết n_mels và target_spec_width
        # Nếu n_mels=128, target_spec_width=256:
        # Patch_size (16,16) -> (128/16, 256/16) = (8, 16) patches
        self.vit_config = ViTConfig(
            image_size=(n_mels, target_spec_width),
            patch_size=(16, 16),  # Hoặc (8, 8), (16, 8) tùy thuộc vào input
            num_channels=1,  # Spectrograms là grayscale
            num_labels=num_classes,  # Số lớp đầu ra
            hidden_size=192,  # Kích thước embedding (ViT-Tiny: 192)
            num_hidden_layers=6,  # Số lớp Transformer (ViT-Tiny: 12, giảm để đơn giản)
            num_attention_heads=3,  # Số head attention (ViT-Tiny: 3)
            intermediate_size=768,  # Kích thước MLP (ViT-Tiny: 768)
            qkv_bias=True,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            # Các thông số khác có thể giữ mặc định
        )

        # Khởi tạo ViTForImageClassification từ cấu hình này
        # Nó sẽ tự động tạo lớp nhúng patch và lớp phân loại cuối cùng
        self.model = ViTForImageClassification(self.vit_config)

        # Vì chúng ta đã đặt num_labels trong config, lớp classifier cuối cùng đã được cấu hình đúng.
        # Nếu cần điều chỉnh, có thể truy cập qua self.model.classifier
        # self.model.classifier = nn.Linear(self.vit_config.hidden_size, num_classes) # Nếu cần thay đổi

    def forward(self, x):
        # x shape: (Batch, 1, N_MELS, TARGET_SPEC_WIDTH)

        # ViTForImageClassification của Hugging Face mong đợi input là (batch_size, num_channels, height, width)
        # và sẽ tự động xử lý việc tạo patch embeddings.

        # Output của ViTForImageClassification là một đối tượng (ImageClassifierOutput)
        # chứa các logits.
        outputs = self.model(x)
        logits = outputs.logits  # Lấy phần logits (đầu ra chưa sigmoid/softmax)
        return logits


# Ví dụ sử dụng và kiểm tra
if __name__ == "__main__":
    print("\n--- Testing Basic ViT Model ---")
    from data_preprocessing import N_MELS, TARGET_SPEC_WIDTH

    model = BasicViT(num_classes=2, n_mels=N_MELS, target_spec_width=TARGET_SPEC_WIDTH)
    print(f"Basic ViT model architecture:\n{model}")

    # Tạo một batch input giả lập
    dummy_input = torch.randn(
        4, 1, N_MELS, TARGET_SPEC_WIDTH
    )  # Batch size 4, 1 kênh, 128 mel bands, 256 frames

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Expected: (4, 2)
    assert output.shape == (4, 2), "Output shape mismatch for BasicViT"
    print("Basic ViT test passed!")
