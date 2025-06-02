import os
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import random

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Cấu hình Data Preprocessing ---
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SPEC_WIDTH = 256
DATASET_ROOT = "dataset"


class AudioDeepfakeDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        dataset_root,
        set_type,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        target_spec_width=TARGET_SPEC_WIDTH,
        augment=None,
        label_mapping={"real": 0, "fake": 1},
    ):
        self.metadata_df = pd.read_csv(metadata_path)
        self.dataset_root = dataset_root
        self.set_type = set_type
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_spec_width = target_spec_width
        self.augment = augment
        self.label_mapping = label_mapping

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        self.amplitude_to_db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

        logging.info(
            f"Dataset initialized with {len(self.metadata_df)} samples from {metadata_path}"
        )
        logging.info(
            f"Spectrogram config: N_MELS={self.n_mels}, TARGET_SPEC_WIDTH={self.target_spec_width}"
        )

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        audio_path_relative = row["path"]
        label_str = row["label"]
        fake_level = row.get("fake_level", 0)

        # Chuyển fake_level thành số nguyên
        try:
            fake_level = int(fake_level)
        except (ValueError, TypeError):
            logging.warning(
                f"Invalid fake_level '{fake_level}' for sample {audio_path_relative}. Using default 0."
            )
            fake_level = 0

        # Kiểm tra xem audio_path_relative đã chứa set_type chưa
        set_type_prefix = f"{self.set_type}{os.sep}"
        if audio_path_relative.startswith(set_type_prefix):
            full_audio_path = os.path.join(self.dataset_root, audio_path_relative)
        else:
            full_audio_path = os.path.join(
                self.dataset_root, self.set_type, audio_path_relative
            )

        if not os.path.exists(full_audio_path):
            logging.warning(
                f"File audio không tìm thấy: {full_audio_path}. Bỏ qua sample."
            )
            return None

        try:
            waveform, sr = torchaudio.load(full_audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            mel_spec = self.mel_spectrogram_transform(waveform)
            if mel_spec.ndim == 3 and mel_spec.shape[0] == 1:
                mel_spec = mel_spec.squeeze(0)
            mel_spec = self.amplitude_to_db_transform(mel_spec)

            current_width = mel_spec.shape[1]
            if current_width < self.target_spec_width:
                pad_amount = self.target_spec_width - current_width
                mel_spec = torch.nn.functional.pad(
                    mel_spec,
                    (0, pad_amount),
                    "constant",
                    -80.0,
                )
            elif current_width > self.target_spec_width:
                if self.augment:
                    start_idx = random.randint(
                        0, current_width - self.target_spec_width
                    )
                    mel_spec = mel_spec[
                        :, start_idx : start_idx + self.target_spec_width
                    ]
                else:
                    start_idx = (current_width - self.target_spec_width) // 2
                    mel_spec = mel_spec[
                        :, start_idx : start_idx + self.target_spec_width
                    ]

            mel_spec = mel_spec.unsqueeze(0)
            if self.augment:
                mel_spec = self.augment(mel_spec)

            label = torch.tensor(self.label_mapping[label_str], dtype=torch.long)
            return mel_spec, label, fake_level, row["path"]

        except Exception as e:
            logging.error(f"Lỗi khi tải hoặc xử lý audio '{full_audio_path}': {e}")
            return None


class SpecAugment(torch.nn.Module):
    def __init__(
        self,
        freq_mask_param=30,
        time_mask_param=100,
        num_freq_masks=1,
        num_time_masks=1,
    ):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spec):
        for _ in range(self.num_freq_masks):
            spec = self.freq_mask(spec)
        for _ in range(self.num_time_masks):
            spec = self.time_mask(spec)
        return spec


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        logging.warning("Batch rỗng sau khi lọc sample lỗi.")
        return None, None, None, None

    specs, labels, fake_levels, paths = zip(*batch)
    specs = torch.stack(specs)
    labels = torch.stack(labels)
    return specs, labels, torch.tensor(list(fake_levels), dtype=torch.long), list(paths)


def get_dataloader(
    set_type, batch_size, shuffle, num_workers=4, include_fake_levels=None
):
    metadata_path = os.path.join(
        DATASET_ROOT, set_type, f"combined_metadata_{set_type}.csv"
    )

    if not os.path.exists(metadata_path):
        logging.error(
            f"File metadata không tìm thấy: {metadata_path}. Vui lòng chạy combine_metadata.py trước."
        )
        return None

    dataset = AudioDeepfakeDataset(
        metadata_path=metadata_path,
        dataset_root=DATASET_ROOT,
        set_type=set_type,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        target_spec_width=TARGET_SPEC_WIDTH,
        augment=SpecAugment() if set_type == "train" else None,
    )

    if include_fake_levels is not None:
        if not isinstance(include_fake_levels, list):
            include_fake_levels = [include_fake_levels]
        dataset.metadata_df["fake_level"] = dataset.metadata_df["fake_level"].astype(
            str
        )
        include_fake_levels_str = [str(level) for level in include_fake_levels]
        initial_len = len(dataset.metadata_df)
        dataset.metadata_df = dataset.metadata_df[
            dataset.metadata_df["fake_level"].isin(include_fake_levels_str)
        ].reset_index(drop=True)
        logging.info(
            f"Đã lọc dataset '{set_type}' để chỉ bao gồm fake_level: {include_fake_levels_str}. "
            f"Từ {initial_len} mẫu giảm xuống còn {len(dataset.metadata_df)} mẫu."
        )
        if len(dataset.metadata_df) == 0:
            logging.warning(
                f"Không có mẫu nào trong dataset '{set_type}' sau khi lọc theo fake_level: {include_fake_levels_str}."
            )
            return None

    initial_len = len(dataset.metadata_df)
    dataset.metadata_df = dataset.metadata_df.dropna(subset=["path"]).reset_index(
        drop=True
    )
    if len(dataset.metadata_df) < initial_len:
        logging.warning(
            f"Đã loại bỏ {initial_len - len(dataset.metadata_df)} mẫu có đường dẫn audio bị thiếu/NaN trong tập '{set_type}'."
        )

    if len(dataset.metadata_df) == 0:
        logging.warning(f"Dataset '{set_type}' rỗng sau khi lọc bỏ các mẫu lỗi.")
        return None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    print("\n--- Testing Data Loaders ---")
    train_loader = get_dataloader(
        "train", batch_size=4, shuffle=True, include_fake_levels=[0, 1]
    )
    if train_loader:
        print(f"Số lượng batch trong train_loader: {len(train_loader)}")
        for i, (specs, labels, fake_levels, paths) in enumerate(train_loader):
            if specs is None:
                print("Batch rỗng hoặc lỗi, bỏ qua.")
                continue
            print(f"Train Batch {i + 1}:")
            print(f"  Spectrograms shape: {specs.shape}")
            print(f"  Labels: {labels}")
            print(f"  Fake Levels: {fake_levels}")
            print(f"  Paths (first): {paths[0]}")
            break
    else:
        print("Không thể tạo train_loader.")

    print("\n--- Testing Val Loader ---")
    val_loader = get_dataloader("val", batch_size=4, shuffle=False)
    if val_loader:
        print(f"Số lượng batch trong val_loader: {len(val_loader)}")
        for i, (specs, labels, fake_levels, paths) in enumerate(val_loader):
            if specs is None:
                print("Batch rỗng hoặc lỗi, bỏ qua.")
                continue
            print(f"Val Batch {i + 1}:")
            print(f"  Spectrograms shape: {specs.shape}")
            print(f"  Labels: {labels}")
            print(f"  Fake Levels: {fake_levels}")
            print(f"  Paths (first): {paths[0]}")
            break
    else:
        print("Không thể tạo val_loader.")
