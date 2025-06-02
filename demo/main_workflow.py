# scripts/main_workflow.py

import torch
import torch.nn as nn
import logging
import os

# Import các thành phần từ các file đã tạo
from data_preprocessing import get_dataloader, N_MELS, TARGET_SPEC_WIDTH, DATASET_ROOT
from models.basic_cnn import BasicCNN
from models.basic_vit import BasicViT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_basic_workflow(model_type="cnn", batch_size=4, num_workers=0):
    """
    Minh họa workflow từ tải dữ liệu, tiền xử lý đến đưa vào model.
    """
    logging.info(f"--- Bắt đầu Workflow cho Model: {model_type.upper()} ---")

    # 1. Tải Dữ liệu
    # Chúng ta sẽ chỉ lấy một số mẫu của tập train để minh họa
    # Bao gồm real (0) và fake level 1 (tts_basic)
    train_loader = get_dataloader(
        set_type="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        include_fake_levels=[0, 1],
    )

    if train_loader is None:
        logging.error(
            "Không thể tải dữ liệu train. Vui lòng kiểm tra các file metadata và đường dẫn."
        )
        return

    logging.info(f"Đã tải thành công {len(train_loader)} batch từ Train Loader.")

    # 2. Khởi tạo Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Sử dụng thiết bị: {device}")

    model = None
    if model_type == "cnn":
        model = BasicCNN(
            num_classes=2,
            input_channels=1,
            n_mels=N_MELS,
            target_spec_width=TARGET_SPEC_WIDTH,
        )
    elif model_type == "vit":
        model = BasicViT(
            num_classes=2, n_mels=N_MELS, target_spec_width=TARGET_SPEC_WIDTH
        )
    else:
        logging.error(
            f"Model type '{model_type}' không được hỗ trợ. Vui lòng chọn 'cnn' hoặc 'vit'."
        )
        return

    model.to(device)
    logging.info(f"Đã khởi tạo Model {model_type.upper()} và chuyển sang {device}.")
    logging.info(f"Cấu trúc Model {model_type.upper()}:\n{model}")

    # 3. Minh họa một vòng lặp huấn luyện đơn giản (chỉ để kiểm tra luồng)
    # Trong một dự án thực tế, bạn sẽ có vòng lặp huấn luyện đầy đủ với optimizer, loss, backprop, v.v.
    logging.info("\n--- Minh họa quá trình forward pass với một batch ---")

    # Lấy một batch từ DataLoader
    # Sử dụng iter() và next() để lấy batch đầu tiên
    try:
        specs, labels, fake_levels, paths = next(iter(train_loader))
    except StopIteration:
        logging.error(
            "Train DataLoader rỗng sau khi lọc. Không có dữ liệu để minh họa."
        )
        return
    except Exception as e:
        logging.error(f"Lỗi khi lấy batch từ DataLoader: {e}")
        return

    if specs is None:
        logging.error(
            "Batch nhận được từ DataLoader là None. Có thể do lỗi trong quá trình tiền xử lý."
        )
        return

    logging.info(
        f"Kích thước Spectrograms trong batch: {specs.shape}"
    )  # Expected: (batch_size, 1, N_MELS, TARGET_SPEC_WIDTH)
    logging.info(f"Labels trong batch: {labels}")
    logging.info(f"Fake Levels trong batch: {fake_levels}")

    # Chuyển dữ liệu lên thiết bị
    specs = specs.to(device)
    labels = labels.to(device)

    # Thực hiện forward pass
    model.eval()  # Chuyển model sang chế độ đánh giá
    with torch.no_grad():  # Không cần tính gradient trong minh họa này
        outputs = model(specs)

    logging.info(
        f"Kích thước Output của Model: {outputs.shape}"
    )  # Expected: (batch_size, 2)
    logging.info(f"Outputs (Logits) từ Model (chỉ 2 ví dụ đầu):\n{outputs[:2]}")

    # Để chuyển đổi logits thành xác suất và dự đoán
    if (
        model_type == "cnn"
    ):  # CNN thường dùng sigmoid cho binary class nếu num_classes=1, hoặc softmax nếu num_classes=2 và cross-entropy
        # Với num_classes=2 và BCEWithLogitsLoss, chúng ta thường chỉ cần một đầu ra (binary classification)
        # Nhưng ở đây, BasicCNN và BasicViT đều có 2 đầu ra.
        # Nếu output là 2 logits (ví dụ: cho class 0 và class 1), thì dùng softmax
        probabilities = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
    else:  # ViT của Hugging Face thường cho logits sẵn
        probabilities = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    logging.info(f"Probabilities từ Model (chỉ 2 ví dụ đầu):\n{probabilities[:2]}")
    logging.info(
        f"Predicted Classes từ Model (chỉ 2 ví dụ đầu):\n{predicted_classes[:2]}"
    )

    logging.info("--- Workflow minh họa hoàn tất ---")


if __name__ == "__main__":
    # Chạy workflow với CNN
    print("\n\n###########################################")
    run_basic_workflow(
        model_type="cnn", batch_size=8
    )  # Thử batch size lớn hơn nếu GPU cho phép
    print("###########################################\n\n")

    # Chạy workflow với ViT
    print("\n\n###########################################")
    run_basic_workflow(model_type="vit", batch_size=8)
    print("###########################################\n\n")
