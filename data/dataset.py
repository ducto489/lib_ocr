import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from loguru import logger
from utils import CTCLabelConverter, AttnLabelConverter
from data.vocab import Vocab


class OCRDataset(Dataset):
    def __init__(self, data_path, batch_max_length, frac, pred_name="attn", transform=None):
        self.data_path = data_path
        self.transform = transform
        self.batch_max_length = batch_max_length

        images, labels = process_tgt(data_path, batch_max_length, frac=frac)
        self.data = list(zip(images, labels))  # list(zip(df['image_name'], df['label']))
        logger.debug("Get Vocab")
        path = os.path.join(self.data_path, "tgt.csv")
        vocab = Vocab().get_vocab()
        logger.debug(f"{pred_name=}")
        if pred_name == "ctc":
            self.converter = CTCLabelConverter(vocab, device="cpu")
        else:
            self.converter = AttnLabelConverter(vocab, batch_max_length=self.batch_max_length, device="cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label = self.data[idx]
        image_path = os.path.join(self.data_path, "images", image_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        encoded_label, length = self.converter.encode([label])

        return image, torch.squeeze(encoded_label), length


def process_tgt(data_path, batch_max_length, frac):
    # Check for tgt.csv file
    tgt_path = os.path.join(data_path, "tgt.csv")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Label file not found at {tgt_path}")

    # Try different encodings for Vietnamese text support
    encodings = ["utf-8", "utf-8-sig", "utf-16"]
    for encoding in encodings:
        try:
            df = pd.read_csv(tgt_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            if encoding == encodings[-1]:
                raise UnicodeDecodeError(f"Failed to read CSV with encodings: {encodings}")
            continue

    # Validate required columns
    required_cols = ["image_name", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Convert values to strings and filter by length
    df["image_name"] = df["image_name"].astype(str)
    df["label"] = df["label"].astype(str)

    # Filter out samples exceeding batch_max_length
    total_samples = len(df)
    df = df[df["label"].str.len() <= batch_max_length].sample(frac=frac, random_state=42)
    filtered_samples = total_samples - len(df)
    if filtered_samples > 0:
        print(
            f"Filtered out {filtered_samples} samples ({filtered_samples / total_samples * 100:.2f}%) exceeding max length {batch_max_length}"
        )

    return list(df["image_name"]), list(df["label"])
