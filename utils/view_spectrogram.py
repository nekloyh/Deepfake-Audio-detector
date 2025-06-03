import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import argparse


class DataConfig:
    SEED = 42  # Random seed for reproducibility

    SR = 16000  # Sample rate (Hz)
    N_FFT = 2048  # FFT window size
    HOP_LENGTH = 512  # Hop length for spectrogram
    N_MELS = 128  # Number of Mel bands
    FMIN = 0.0  # Minimum frequency (Hz)
    FMAX = 8000.0  # Maximum frequency (Hz)

    MASK_REPLACEMENT_VALUE = -80.0  # Value for masked regions in spectrogram
    NORM_EPSILON = 1e-6  # Small value to prevent division by zero
    LOUDNESS_LUFS = -23.0  # Target loudness (LUFS)


def view_spectrogram(
    npy_file_path: str,
    display_type: str = "mel",
    save_path: str = "F:\\Deepfake-Audio-Detector\\tests\\view_spectrogram",
    vmin_display: float = DataConfig.MASK_REPLACEMENT_VALUE,
):
    """
    Tải và hiển thị Mel-spectrogram hoặc spectrogram từ một file .npy.

    Args:
        npy_file_path (str): Đường dẫn đến file .npy chứa spectrogram.
        display_type (str): Loại spectrogram để hiển thị ('mel' hoặc 'linear').
                            Mặc định là 'mel' vì các file .npy được tạo từ Mel-spectrogram.
        save_path (str, optional): Đường dẫn để lưu hình ảnh spectrogram. Nếu None, sẽ hiển thị.
    """
    if not os.path.exists(npy_file_path):
        print(f"Lỗi: File .npy không tồn tại tại đường dẫn '{npy_file_path}'")
        return

    try:
        spectrogram_db = np.load(npy_file_path)
    except Exception as e:
        print(f"Lỗi khi tải file .npy '{npy_file_path}': {e}")
        return

    print(
        f"Tải thành công spectrogram từ '{npy_file_path}'. Hình dạng: {spectrogram_db.shape}"
    )

    if spectrogram_db.ndim not in [2, 3]:
        print("Lỗi: Hình dạng spectrogram không hợp lệ. Mong đợi 2D (n_mels, n_frames) hoặc 3D (1, n_mels, n_frames).")
        print(f"Hình dạng hiện tại: {spectrogram_db.shape}")
        return

    # Nếu là 3D (ví dụ: (1, n_mels, n_frames) do unsqueeze(0) trong quá trình lưu), thì loại bỏ chiều đầu tiên
    if spectrogram_db.ndim == 3 and spectrogram_db.shape[0] == 1:
        spectrogram_db = spectrogram_db.squeeze(0)

    plt.figure(figsize=(12, 6))

    if display_type == "mel":
        # Hàm specshow của librosa yêu cầu giá trị tần số Mel là trục Y
        # Với Mel-spectrogram đã ở dạng dB, chúng ta chỉ cần hiển thị
        librosa.display.specshow(
            spectrogram_db,
            sr=DataConfig.SR,
            x_axis="time",
            y_axis="mel",
            fmin=DataConfig.FMIN,
            fmax=DataConfig.FMAX,
            hop_length=DataConfig.HOP_LENGTH,
            cmap="viridis",
            vmax=0.0,  # Giá trị dB max, thường là 0 dB (ref=np.max)
            vmin=vmin_display,  # Giá trị dB min cho màu sắc
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mel-Spectrogram from {os.path.basename(npy_file_path)}")
        plt.ylabel("Mel Frequency")
    elif display_type == "linear":
        # Để hiển thị linear spectrogram từ Mel-spectrogram, bạn cần chuyển đổi ngược lại
        # Tuy nhiên, chuyển đổi ngược từ Mel-spectrogram đã ở dB là không chính xác hoàn toàn.
        # Thường thì bạn sẽ hiển thị spectrogram (linear scale) hoặc Mel-spectrogram (mel scale).
        # Nếu muốn hiển thị spectrogram tuyến tính, bạn cần tải file âm thanh gốc và tính toán lại.
        # Ở đây, chúng ta vẫn hiển thị nó nhưng trên trục tần số tuyến tính
        print("Cảnh báo: Hiển thị Mel-spectrogram trên trục tần số tuyến tính.")
        librosa.display.specshow(
            spectrogram_db,
            sr=DataConfig.SR,
            x_axis="time",
            y_axis="linear",
            hop_length=DataConfig.HOP_LENGTH,
            cmap="viridis",
            vmax=0.0,
            vmin=vmin_display,
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectrogram (Linear Freq) from {os.path.basename(npy_file_path)}")
        plt.ylabel("Frequency (Hz)")
    else:
        print(
            f"Loại hiển thị '{display_type}' không được hỗ trợ. Chỉ hỗ trợ 'mel' hoặc 'linear'."
        )
        return

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Spectrogram đã được lưu tại '{save_path}'")
        except Exception as e:
            print(f"Lỗi khi lưu hình ảnh: {e}")
        plt.close()  # Đóng plot sau khi lưu để tránh hiển thị nếu không cần
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hiển thị Mel-spectrogram hoặc spectrogram từ file .npy."
    )
    parser.add_argument(
        "npy_file", type=str, help="Đường dẫn đến file .npy chứa spectrogram."
    )
    parser.add_argument(
        "--type",
        type=str,
        default="mel",
        choices=["mel", "linear"],
        help="Loại hiển thị spectrogram: 'mel' (mặc định) hoặc 'linear'.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Đường dẫn để lưu hình ảnh spectrogram (ví dụ: output.png). Nếu không chỉ định, sẽ hiển thị cửa sổ.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=DataConfig.MASK_REPLACEMENT_VALUE,  # Giá trị mặc định mới
        help="Giá trị dB tối thiểu để hiển thị spectrogram. Giá trị âm. Mặc định là -80.0 dB. Có thể điều chỉnh về -50.0 để hiển thị rõ hơn các chi tiết.",
    )
    # hiện ảnh lên màn hình
    # python view_spectrogram.py <đường_dẫn_đến_file.npy>
    
    # lưu ảnh
    # python view_spectrogram.py <đường_dẫn_đến_file.npy> --save spectrogram_output.png
    
    args = parser.parse_args()

    view_spectrogram(args.npy_file, args.type, args.save, args.vmin)
