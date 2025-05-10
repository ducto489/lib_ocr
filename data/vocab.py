import os
import json
import pandas as pd


class Vocab:
    def __init__(self):
        """
        Initialize vocabulary from labels file
        Args:
            label_path: Path to labels file (CSV or JSON)
        """

    def get_vocab_json(self, label_path):
        """Get vocabulary from JSON file"""
        with open(label_path, "r") as f:
            data = json.load(f)
        data = data["labels"]
        vocab = set()
        for label in data.values():
            vocab.update(list(label))
        return list(vocab)

    def get_vocab_csv(self, label_path):
        """Get vocabulary from CSV file with Unicode support"""
        # Try different encodings for Vietnamese text
        for encoding in ["utf-8", "utf-8-sig", "utf-16"]:
            try:
                df = pd.read_csv(label_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not read CSV file with any of the supported encodings")

        if "label" not in df.columns:
            raise ValueError(f"CSV file must contain 'label' column. Found columns: {df.columns.tolist()}")

        vocab = set()
        # Convert labels to string to handle any numeric values
        for label in df["label"].astype(str):
            vocab.update(list(label))

        # Add Vietnamese characters
        vi_char_path = os.path.join(os.path.dirname(__file__), "vi_char.txt")
        if os.path.exists(vi_char_path):
            try:
                with open(vi_char_path, "r", encoding="utf-8") as f:
                    vi_chars = set(f.read().splitlines())
                vocab.update(vi_chars)
            except Exception as e:
                raise ValueError(f"Error reading Vietnamese characters file: {str(e)}")

        return sorted(list(vocab))  # Sort for consistent ordering

    def get_vocab(self):
        char_path = os.path.join(os.path.dirname(__file__), "vocab.txt")
        with open(char_path, "r", encoding="utf-8") as f:
            vocab = set(f.read().splitlines())
        return sorted(list(vocab))
