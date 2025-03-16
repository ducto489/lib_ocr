import os
import json
import csv
import pandas as pd


class Vocab:
    def __init__(self, label_path):
        """
        Initialize vocabulary from labels file
        Args:
            label_path: Path to labels file (CSV or JSON)
        """
        self.label_path = label_path
        if self.label_path.endswith('.csv'):
            self.vocab = self.get_vocab_csv()
        else:
            self.vocab = self.get_vocab()

    def get_vocab(self):
        """Get vocabulary from JSON file"""
        with open(self.label_path, 'r') as f:
            data = json.load(f)
        data = data['labels']
        vocab = set()
        for label in data.values():
            vocab.update(list(label))
        return list(vocab)

    def get_vocab_csv(self):
        """Get vocabulary from CSV file with Unicode support"""
        # Try different encodings for Vietnamese text
        for encoding in ['utf-8', 'utf-8-sig', 'utf-16']:
            try:
                df = pd.read_csv(self.label_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not read CSV file with any of the supported encodings")

        if 'label' not in df.columns:
            raise ValueError(f"CSV file must contain 'label' column. Found columns: {df.columns.tolist()}")

        vocab = set()
        # Convert labels to string to handle any numeric values
        for label in df['label'].astype(str):
            vocab.update(list(label))

        # Add Vietnamese characters
        vi_char_path = os.path.join(os.path.dirname(__file__), 'vi_char.txt')
        if os.path.exists(vi_char_path):
            try:
                with open(vi_char_path, 'r', encoding='utf-8') as f:
                    vi_chars = set(f.read().splitlines())
                vocab.update(vi_chars)
            except Exception as e:
                raise ValueError(f"Error reading Vietnamese characters file: {str(e)}")

        return sorted(list(vocab))  # Sort for consistent ordering


if __name__ == '__main__':
    # Use the correct folder name
    vocab = Vocab('/hdd1t/mduc/data/train/tgt.csv')
    vocab_list = vocab.get_vocab_csv()
    print(f"Vocabulary size: {len(vocab_list)}")
    print("\nVocabulary list:")
    for i, char in enumerate(vocab_list):
        print(f"{i+1:3d}. '{char}'")