import os
import json

class Vocab:
    def __init__(self, label_path):
        """
        Initialize vocabulary from labels file
        Args:
            label_path: Path to labels.json file
        """
        self.label_path = label_path
        self.vocab = self.get_vocab()

    def get_vocab(self):
        with open(self.label_path, 'r') as f:
            data = json.load(f)
        data = data['labels']
        vocab = set()
        for label in data.values():
            vocab.update(list(label))
        return list(vocab)

if __name__ == '__main__':
    # Use the correct folder name
    vocab = Vocab('../training_images/labels.json')
    print(vocab.get_vocab())
    