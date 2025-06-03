# Giải thích chi tiết các hàm trong `data_preprocessing.py` và tác động đến folder gốc

File `data_preprocessing.py` này thực hiện tiền xử lý dữ liệu âm thanh để tạo các tập dữ liệu được cache cho các mô hình học sâu. Dưới đây là giải thích chi tiết về từng class và hàm, cùng với ý nghĩa của đầu ra hiện tại.

## 1. Cấu hình (`DataConfig` và `DataLoaderConfig`)

### 1.1 `DataConfig`
Class này định nghĩa các tham số cấu hình tĩnh cho việc xử lý dữ liệu âm thanh.

* **SR**: Tốc độ lấy mẫu (sample rate) của âm thanh, mặc định 16000 Hz.
* **N_FFT**: Kích thước cửa sổ FFT (Fast Fourier Transform), mặc định 2048.
* **HOP_LENGTH**: Độ dài bước nhảy giữa các cửa sổ FFT, mặc định 512.
* **N_MELS**: Số lượng dải Mel (Mel bands) trong spectrogram, mặc định 128.
* **FMIN**, **FMAX**: Tần số tối thiểu và tối đa cho Mel-spectrogram, mặc định 0.0 Hz và 8000.0 Hz.
* **NUM_TIME_MASKS**, **NUM_FREQ_MASKS**: Số lượng mặt nạ thời gian và tần số cho SpecAugment.
* **TIME_MASK_MAX_WIDTH**, **FREQ_MASK_MAX_WIDTH**: Chiều rộng tối đa của mặt nạ thời gian và tần số.
* **MASK_REPLACEMENT_VALUE**: Giá trị thay thế cho các vùng bị che mặt nạ trong spectrogram, mặc định -80.0 dB.
* **NORM_EPSILON**: Một giá trị nhỏ để tránh chia cho 0 khi chuẩn hóa.
* **LOUDNESS_LUFS**: Độ lớn mục tiêu theo đơn vị LUFS (Loudness Units Full Scale) cho chuẩn hóa độ lớn.
* **USE_GLOBAL_NORMALIZATION**: Cờ để sử dụng chuẩn hóa toàn cục (trung bình/độ lệch chuẩn của toàn bộ dataset).
* **USE_RANDOM_CROPPING**: Cờ để áp dụng cắt ngẫu nhiên cho spectrogram.
* **DATASET_ROOT**: Thư mục gốc chứa dữ liệu dataset thô.
* **CACHE_DIR**: Thư mục để lưu trữ dữ liệu đã được xử lý và cache.

### 1.2 `DataLoaderConfig`
Class này cấu hình các tham số cho việc tạo DataLoader, bao gồm kích thước batch và các tùy chọn augmentation.

* **audio_length_seconds**: Độ dài mong muốn của đoạn âm thanh tính bằng giây.
* **batch_size**: Kích thước batch cho DataLoader.
* **num_workers**: Số lượng tiến trình con để tải dữ liệu.
* **apply_augmentation_to_train**: Có áp dụng augmentation cho tập huấn luyện hay không.
* **apply_waveform_augmentation**: Có áp dụng augmentation ở mức waveform hay không.
* **limit_files**: Giới hạn số lượng file được xử lý (để debug hoặc thử nghiệm).
* **overlap_ratio**: Tỷ lệ chồng lấn giữa các đoạn âm thanh khi phân đoạn.
* **max_frame_spec**: Số khung tối đa trong spectrogram, được tính toán dựa trên `audio_length_seconds`, `SR` và `HOP_LENGTH`.

## 2. Hàm hỗ trợ

### 2.1 `_load_and_segment_audio`
Tải một file âm thanh và phân đoạn nó thành các phần có độ dài cố định, đồng thời chuẩn hóa độ lớn.

* **file_path**: Đường dẫn đến file âm thanh.
* **sr**: Tốc độ lấy mẫu.
* **segment_length**: Độ dài mong muốn của mỗi đoạn âm thanh tính bằng giây.
* **overlap_ratio**: Tỷ lệ chồng lấn giữa các đoạn.

**Đầu ra**: Một danh sách các mảng NumPy, mỗi mảng là một đoạn âm thanh. Trả về danh sách rỗng nếu có lỗi hoặc âm thanh quá nhỏ.

### 2.2 `_audio_to_mel_spectrogram`
Chuyển đổi dạng sóng âm thanh thành Mel-spectrogram với kích thước trục thời gian cố định.

* **y**: Mảng NumPy của dạng sóng âm thanh.
* **sr, n_fft, hop_length, n_mels, fmin, fmax**: Các tham số cho việc tính toán Mel-spectrogram.
* **max_frames_spec**: Số khung thời gian tối đa mong muốn của spectrogram.
* **random_crop**: Có cắt ngẫu nhiên spectrogram nếu nó dài hơn `max_frames_spec`.

**Đầu ra**: Một mảng NumPy đại diện cho Mel-spectrogram đã được chuẩn hóa kích thước. Các vùng thiếu sẽ được đệm bằng `DataConfig.MASK_REPLACEMENT_VALUE`.

### 2.3 `_compute_global_stats`
Tính toán trung bình và độ lệch chuẩn toàn cục của các spectrogram từ một danh sách các file âm thanh.

* **filepaths**: Danh sách các đường dẫn đến file âm thanh.
* **segment_length**: Độ dài đoạn âm thanh để tính toán spectrogram.
* **max_frames_spec**: Số khung tối đa cho spectrogram.

**Đầu ra**: Một tuple chứa trung bình và độ lệch chuẩn toàn cục.

## 3. Các lớp Augmentation

### 3.1 `SpecAugment`
Class này thực hiện SpecAugment, một kỹ thuật tăng cường dữ liệu bằng cách che các vùng tần số và thời gian trên spectrogram.

* **freq_mask**: Đối tượng `torchaudio.transforms.FrequencyMasking`.
* **time_mask**: Đối tượng `torchaudio.transforms.TimeMasking`.
* **num_freq_masks**, **num_time_masks**: Số lượng mặt nạ được áp dụng.

**Hàm `forward`**: Áp dụng che tần số và thời gian cho một spectrogram đầu vào (Tensor PyTorch).
**Đầu ra**: Spectrogram đã được augmentation.

### 3.2 `WaveformAugment`
Class này áp dụng các kỹ thuật tăng cường dữ liệu ở mức waveform (dạng sóng âm thanh).

* **sr**: Tốc độ lấy mẫu.
* **pitch_shift**: Đối tượng `torchaudio.transforms.PitchShift`.

**Hàm `apply`**: Áp dụng ngẫu nhiên các augmentation như thêm nhiễu, dịch chuyển cao độ (pitch shift), và giãn/nén thời gian (time stretch) lên dạng sóng âm thanh.
**Đầu ra**: Dạng sóng âm thanh đã được augmentation (mảng NumPy).

## 4. Cấu hình mô hình (`ModelConfig`)

### 4.1 `ModelConfig`
Class này định nghĩa cấu hình cụ thể cho từng mô hình, bao gồm độ dài âm thanh, tỷ lệ chồng lấn, và các tùy chọn augmentation.

* **name**: Tên của cấu hình mô hình.
* **audio_length_seconds**: Độ dài đoạn âm thanh cho mô hình này.
* **overlap_ratio**: Tỷ lệ chồng lấn khi phân đoạn âm thanh.
* **apply_augmentation**: Có áp dụng SpecAugment cho mô hình này hay không.
* **apply_waveform_augmentation**: Có áp dụng Waveform Augmentation cho mô hình này hay không.
* **max_frames_spec**: Số khung tối đa của spectrogram cho mô hình này.

## 5. Lớp tạo Dataset (`DatasetCreator`)

### 5.1 `DatasetCreator`
Class chính quản lý việc tạo ra các tập dữ liệu được cache cho các cấu hình mô hình khác nhau.

* **model_configs**: Danh sách các đối tượng `ModelConfig`.
* **label_mapping**: Ánh xạ nhãn `{"real": 0, "fake": 1}`.
* **spec_augmenter**: Đối tượng `SpecAugment`.
* **waveform_augmenter**: Đối tượng `WaveformAugment`.

### 5.2 `load_metadata`
Tải metadata (thông tin file, nhãn) từ file CSV cho một loại tập dữ liệu (`train`, `val`, `test`).

* **set_type**: Loại tập dữ liệu (`"train"`, `"val"`, `"test"`).

**Đầu ra**: Một DataFrame của pandas chứa metadata.

### 5.3 `validate_and_get_full_path`
Xác thực đường dẫn file âm thanh và trả về đường dẫn đầy đủ, kiểm tra sự tồn tại và tính hợp lệ của file.

* **set_type**: Loại tập dữ liệu (dùng để xây dựng đường dẫn đầy đủ).
* **audio_path_relative**: Đường dẫn tương đối của file âm thanh trong CSV.

**Đầu ra**: Đường dẫn đầy đủ đến file âm thanh nếu hợp lệ, nếu không trả về `None`.

### 5.4 `create_cached_datasets`
Hàm chính để tạo các tập dữ liệu được cache cho tất cả các cấu hình mô hình được cung cấp.

* Tạo thư mục cache cho từng cấu hình mô hình.
* Đối với mỗi loại tập (`train`, `val`, `test`):
    * Tải metadata.
    * Tính toán trung bình và độ lệch chuẩn toàn cục nếu `USE_GLOBAL_NORMALIZATION` được bật.
    * Lặp qua từng hàng metadata:
        * Xác thực và lấy đường dẫn đầy đủ của file âm thanh.
        * Tải và phân đoạn âm thanh.
        * Áp dụng waveform augmentation (nếu cấu hình cho phép và là tập `train`).
        * Chuyển đổi sang Mel-spectrogram và chuẩn hóa kích thước.
        * Chuẩn hóa spectrogram (toàn cục hoặc theo mẫu).
        * Áp dụng SpecAugment (nếu cấu hình cho phép và là tập `train`).
        * Lưu spectrogram đã xử lý vào file `.npy` trong thư mục cache.
        * Ghi lại thông tin của mẫu đã xử lý vào một DataFrame mới.
    * Lưu DataFrame metadata đã xử lý vào file `metadata.csv` trong thư mục cache của tập dữ liệu đó.

## 6. Khối `if __name__ == "__main__":`

Phần này của mã là điểm khởi chạy khi file được thực thi trực tiếp.

* Xóa thư mục `CACHE_DIR` nếu nó tồn tại để đảm bảo làm mới hoàn toàn.
* Tạo lại thư mục `CACHE_DIR`.
* Định nghĩa hai đối tượng `ModelConfig`: `cnn_config_balanced` và `vit_config_balanced` với các tham số tương tự (8.0 giây, overlap 0.5, cả hai loại augmentation).
* Khởi tạo `DatasetCreator` với các cấu hình này và gọi `create_cached_datasets()` để tạo dữ liệu.
* Định nghĩa thêm hai đối tượng `ModelConfig`: `cnn_config_performance` và `vit_config_performance` với các tham số khác nhau (4.0 giây, overlap 0.75 cho CNN; 10.0 giây, overlap 0.0 cho ViT). Điều này cho thấy ý định tạo các bộ dữ liệu được tối ưu hóa cho các mục tiêu hiệu suất khác nhau (ví dụ: tốc độ so với độ chính xác).
* Khởi tạo lại `DatasetCreator` với các cấu hình hiệu suất và gọi `create_cached_datasets()` một lần nữa.

## Ý nghĩa đầu ra hiện tại

Khi file `data_preprocessing.py` được chạy, đầu ra chính sẽ là:

1.  **Thư mục `processed_dataset`**: Trong thư mục `F:\Deepfake-Audio-Detector`, một thư mục `processed_dataset` sẽ được tạo ra.
2.  **Thư mục con cho mỗi cấu hình mô hình**: Bên trong `processed_dataset`, các thư mục con sẽ được tạo tương ứng với tên của `ModelConfig` đã định nghĩa. Ví dụ:
    * `processed_dataset/cnn_balanced_dataset`
    * `processed_dataset/vit_balanced_dataset`
    * `processed_dataset/cnn_performance_dataset`
    * `processed_dataset/vit_performance_dataset`
3.  **Thư mục con `train`, `val`, `test`**: Mỗi thư mục mô hình sẽ chứa các thư mục con `train`, `val`, và `test`.
4.  **File `.npy` chứa spectrogram**: Bên trong các thư mục `train/real`, `train/fake`, `val/real`, `val/fake`, `test/real`, `test/fake`, bạn sẽ tìm thấy các file `.npy`. Mỗi file `.npy` chứa một Mel-spectrogram đã được tiền xử lý và chuẩn hóa (và có thể đã được augmentation) của một đoạn âm thanh. Tên file `.npy` là một hash MD5 được tạo từ đường dẫn file gốc và chỉ số đoạn, đảm bảo tính duy nhất.
5.  **File `metadata.csv`**: Trong mỗi thư mục `train`, `val`, `test` (ví dụ: `processed_dataset/cnn_balanced_dataset/train/metadata.csv`), sẽ có một file `metadata.csv`. File này chứa các cột sau:
    * **npy_path**: Đường dẫn tương đối đến file `.npy` đã lưu.
    * **original_path**: Đường dẫn tương đối của file âm thanh gốc trong dataset thô.
    * **label**: Nhãn số của mẫu (0 cho "real", 1 cho "fake").
    * **fake_level**: Mức độ giả mạo (nếu có trong metadata gốc), hoặc 0 nếu không xác định.
    * **segment_index**: Chỉ số của đoạn âm thanh trong trường hợp file gốc được chia thành nhiều đoạn.

**Tóm lại**, quá trình này tạo ra các bản sao tiền xử lý của dataset gốc, được tối ưu hóa cho từng cấu hình mô hình cụ thể. Thay vì phải xử lý âm thanh và tính toán spectrogram mỗi lần huấn luyện, mô hình có thể tải trực tiếp các file `.npy` đã được chuẩn hóa và sẵn sàng sử dụng, giúp tăng tốc đáng kể quá trình tải dữ liệu và huấn luyện. Mỗi `metadata.csv` đóng vai trò là một chỉ mục cho các file `.npy` đã được cache, giúp dễ dàng truy cập dữ liệu và nhãn tương ứng.
