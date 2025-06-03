# scripts/models/basic_cnn.py

import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(
        self, num_classes=2, input_channels=1, n_mels=128, target_spec_width=256
    ):
        super(BasicCNN, self).__init__()

        # Các lớp Convolutional Layers
        # Input: (Batch, 1, 128, 256)
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=(3, 3), padding=1
        )  # Output: (Batch, 32, 128, 256)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )  # Output: (Batch, 32, 64, 128)

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=(3, 3), padding=1
        )  # Output: (Batch, 64, 64, 128)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )  # Output: (Batch, 64, 32, 64)

        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=(3, 3), padding=1
        )  # Output: (Batch, 128, 32, 64)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )  # Output: (Batch, 128, 16, 32)

        # Tính toán kích thước đầu ra sau các lớp Conv và Pool
        # Công thức: (InputSize - KernelSize + 2*Padding) / Stride + 1
        # Hoặc đơn giản hơn, chạy thử một forward pass với dummy input để kiểm tra

        # Kích thước đầu ra sau pool3: (128, 16, 32)
        # Flattened size: 128 * 16 * 32 = 65536

        # Để làm cho nó linh hoạt hơn với N_MELS và TARGET_SPEC_WIDTH khác nhau,
        # chúng ta có thể tính toán kích thước đầu ra động.
        dummy_input = torch.randn(1, input_channels, n_mels, target_spec_width)
        with torch.no_grad():
            x = self.pool1(self.relu1(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
            self.flattened_features = x.view(x.size(0), -1).shape[
                1
            ]  # Lấy kích thước phẳng

        # Lớp Fully Connected (Classifier)
        self.fc = nn.Linear(self.flattened_features, num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, N_MELS, TARGET_SPEC_WIDTH)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # Flatten tensor cho lớp FC
        x = x.view(
            x.size(0), -1
        )  # Flatten từ (Batch, Channels, H, W) thành (Batch, Channels*H*W)

        x = self.fc(x)
        return x


# Ví dụ sử dụng và kiểm tra
if __name__ == "__main__":
    print("\n--- Testing Basic CNN Model ---")
    # Đảm bảo N_MELS và TARGET_SPEC_WIDTH khớp với data_preprocessing.py
    from data_preprocessing import N_MELS, TARGET_SPEC_WIDTH

    model = BasicCNN(
        num_classes=2,
        input_channels=1,
        n_mels=N_MELS,
        target_spec_width=TARGET_SPEC_WIDTH,
    )
    print(f"Basic CNN model architecture:\n{model}")

    # Tạo một batch input giả lập
    dummy_input = torch.randn(
        4, 1, N_MELS, TARGET_SPEC_WIDTH
    )  # Batch size 4, 1 kênh, 128 mel bands, 256 frames

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Expected: (4, 2)
    assert output.shape == (4, 2), "Output shape mismatch for BasicCNN"
    print("Basic CNN test passed!")
