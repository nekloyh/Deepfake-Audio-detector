import torch
import torch.onnx
import os
import numpy as np
from dataclasses import dataclass, field


# Define Config class (same as in the original notebook)
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


# Define CNN_Audio class (same as in the original notebook)
class CNN_Audio(torch.nn.Module):
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

        with torch.no_grad():
            dummy_input_for_init = torch.randn(1, self.in_channels, self.img_size, self.img_size)
            dummy_output_conv = self.conv_layers(dummy_input_for_init)
            print(f"DEBUG: Shape of conv_layers output before pooling: {dummy_output_conv.shape}") # Make sure it's 14x14
            self.flattened_size = dummy_output_conv.view(1, -1).size(1)
        self.adaptive_pool = torch.nn.AvgPool2d(kernel_size=(4, 4), stride=(3, 3))

        with torch.no_grad():
            dummy_pooled = self.adaptive_pool(dummy_output_conv)
            print(f"DEBUG: Shape after AvgPool2d: {dummy_pooled.shape}") # Should be 1, 512, 4, 4
            self.pooled_size = dummy_pooled.view(1, -1).size(1)
            print(f"DEBUG: Calculated pooled_size: {self.pooled_size}") # Should be 8192

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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def convert_and_verify_cnn_large_to_onnx():
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize configuration for CNN_Large
        base_config = Config()
        base_params = {
            f.name: getattr(base_config, f.name)
            for f in base_config.__dataclass_fields__.values()
            if f.init
            and f.name
            not in [
                "model_size",
                "dataset_name",
                "cnn_conv_channels",
                "cnn_pool_after_conv",
                "linear_output_units_1st_fc",
            ]
        }

        # --- CẤU HÌNH CHO CNN_LARGE ---
        config = Config(
            **base_params,
            model_size="CNN_Large",
            dataset_name="cnn_3s_dataset",
            cnn_conv_channels=[64, 128, 256, 512, 512],  # Kênh và lớp cho CNN_Large
            cnn_pool_after_conv=[True, True, True, True, False],  # Cấu hình pooling
            linear_output_units_1st_fc=192,  # Đơn vị cho lớp FC đầu tiên
        )
        print(f"Config for model: {config.model_size}")

        # Initialize model
        model = CNN_Audio(
            img_size=config.img_size,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            linear_output_units_1st_fc=config.linear_output_units_1st_fc,
            cnn_conv_channels=config.cnn_conv_channels,
            cnn_pool_after_conv=config.cnn_pool_after_conv,
            dropout=config.dropout,
        )
        model = model.to(device)
        model.eval()  # Rất quan trọng: đặt mô hình ở chế độ đánh giá

        # Load the trained model weights
        # --- CẬP NHẬT ĐƯỜNG DẪN ĐẾN TỆP .PTH CỦA CNN_LARGE CỦA BẠN ---
        checkpoint_path = "F:\\Deepfake-Audio-Detector\\models\\.pth\\best_model_CNN_Large_cnn_3s_dataset_114040.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}\n"
                "Please update 'checkpoint_path' to your CNN_Large model's .pth file."
            )

        torch.serialization.add_safe_globals([np._core.multiarray.scalar]) #type: ignore

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded successfully")

        # Create dummy input for ONNX export
        dummy_input = torch.randn(
            1, config.in_channels, config.img_size, config.img_size
        ).to(device)

        # Define output path
        output_onnx_path = "F:\\Deepfake-Audio-Detector\\models\\.onnx\\CNN_Large.onnx"  # Tên tệp ONNX đầu ra

        # Export to ONNX
        print("\nExporting model to ONNX format...")
        torch.onnx.export(
            model,
            dummy_input,  # type: ignore
            output_onnx_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            training=torch.onnx.TrainingMode.EVAL,  # Add this line
        )
        print(
            f"Successfully converted {config.model_size} to ONNX format at {output_onnx_path}"
        )

        # --- BƯỚC XÁC MINH (RẤT QUAN TRỌNG) ---
        print("\nVerifying ONNX model...")
        import onnxruntime

        # Chạy suy luận với mô hình PyTorch
        with torch.no_grad():
            pytorch_output = model(dummy_input).cpu().numpy()

        # Chạy suy luận với mô hình ONNX
        try:
            sess_options = onnxruntime.SessionOptions()
            sess = onnxruntime.InferenceSession(
                output_onnx_path,
                sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            print(f"ONNX Runtime providers: {sess.get_providers()}")
        except Exception as e:
            print(
                f"Could not initialize ONNX Runtime with CUDA, falling back to CPU. Error: {e}"
            )
            sess = onnxruntime.InferenceSession(
                output_onnx_path, providers=["CPUExecutionProvider"]
            )

        onnx_input_name = sess.get_inputs()[0].name
        onnx_output_name = sess.get_outputs()[0].name

        onnx_input = {onnx_input_name: dummy_input.cpu().numpy()}
        onnx_output = sess.run([onnx_output_name], onnx_input)[0]

        # So sánh kết quả
        np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-4)
        print("Verification successful: PyTorch and ONNX outputs match!")
        print(f"Sample PyTorch output (first 5 values):\n{pytorch_output[0, :5]}")
        print(f"Sample ONNX output (first 5 values):\n{onnx_output[0, :5]}")

        if np.argmax(pytorch_output) == np.argmax(onnx_output):
            print("ONNX and PyTorch predictions match ✅")
        else:
            print("Predicted classes differ ❌")

        # Optional: still print difference
        diff = np.abs(pytorch_output - onnx_output)
        print(f"Max difference: {diff.max()}, Mean difference: {diff.mean()}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(
            f"An unexpected error occurred during conversion or verification: {str(e)}"
        )
        raise


if __name__ == "__main__":
    convert_and_verify_cnn_large_to_onnx()
