import torch
from torchmetrics import Metric
import time
import logging
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from loguru import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CTCLabelConverter:
    def __init__(self, vocab, device='cuda'):
        # Add blank token at position 0
        self.SPACE = ' '
        self.BLANK = '[blank]'
        self.UNKNOWN = '[UNK]'
        
        self.vocab = [self.BLANK] + list(vocab) + [self.UNKNOWN]
        self.dict = {char: idx for idx, char in enumerate(self.vocab)}
        self.device = device

    def encode(self, texts):
        """
        Convert text strings to encoded tensor and lengths
        """
        lengths = [len(text) for text in texts]
        
        # Convert characters to indices, using UNKNOWN token for unknown chars
        encoded = []
        for text in texts:
            text_encoded = []
            for char in text:
                if char in self.dict:
                    text_encoded.append(self.dict[char])
                else:
                    text_encoded.append(self.dict[self.UNKNOWN])
            encoded.append(text_encoded)
        
        # Create padded tensor
        max_length = max(lengths)
        batch_size = len(texts)
        batch_tensor = torch.zeros(batch_size, max_length).long()
        
        for i, text_encoded in enumerate(encoded):
            batch_tensor[i, :len(text_encoded)] = torch.tensor(text_encoded)
            
        batch_tensor = batch_tensor.to(self.device)
        lengths = torch.tensor(lengths).to(self.device)
        
        return batch_tensor, lengths

    def decode(self, text_indices, length):
        """
        Convert encoded indices back to text strings
        """
        texts = []
        for indices, l in zip(text_indices, length):
            text = ''.join([self.vocab[idx] for idx in indices[:l]])
            texts.append(text)
        return texts
    
    def decode_v1(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (
                    not (i > 0 and t[i - 1] == t[i])
                ):  # removing repeated characters and blank.
                    char_list.append(self.vocab[t[i]])
            text = "".join(char_list)

            texts.append(text)
        return texts
    
class CTCLabelConverter_clovaai(object):
    """Convert between text-label and text-index"""

    def __init__(self, character, device):
        self.device = device
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = [
            "[CTCblank]"
        ] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][: len(text)] = torch.LongTensor(text)
        return (batch_text.to(self.device), torch.IntTensor(length).to(self.device))

    def decode(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (
                    not (i > 0 and t[i - 1] == t[i])
                ):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = "".join(char_list)

            texts.append(text)
        return texts


class AttnLabelConverter:
    def __init__(self, character):
        list_token = ['[GO]', '[EOS]']
        self.character = list_token + list(character)
        self.dict = {char: idx for idx, char in enumerate(self.character)}

    def encode(self, text, batch_max_length=50):
        """convert text-label into text-index."""
        length = [len(s) + 1 for s in text]
        batch_max_length += 1
        batch_text = torch.zeros(len(text), batch_max_length + 1).long()
        for i, t in enumerate(text):
            text = list(t)
            text.append('[EOS]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

def test_ctc_label_converter(text = ["hel` lo", "world"]):
    vocab = "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"
    converter = CTCLabelConverter(vocab, device="cpu")
    text_index, length = converter.encode(text)
    print("Original:", text)
    print("Encoded:", text_index)
    print("Decoded:", converter.decode_v1(text_index, length))

def test_ctc_label_converter_clovaai():
    import string
    vocab = string.printable[:-6]
    converter = CTCLabelConverter_clovaai(vocab, device="cpu")
    text = ["hello ", "world"]
    text_index, length = converter.encode(text)
    print("Original:", text)
    print("Encoded:", text_index)
    print("Decoded:", converter.decode(text_index, length))

class SentenceErrorRate(Metric):
    """Custom metric to compute Sentence Error Rate (SER).
    SER is the percentage of sentences that contain any errors."""
    
    def __init__(self):
        super().__init__()
        self.add_state("incorrect", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: list, target: list):
        """Update state with predictions and targets.
        
        Args:
            preds: List of predicted sentences
            target: List of target sentences
        """
        assert len(preds) == len(target), "Number of predictions and targets must match"
        
        # Count sentences with any errors
        incorrect = sum(1 for p, t in zip(preds, target) if p != t)
        self.incorrect += torch.tensor(incorrect)
        self.total += torch.tensor(len(preds))
        
    def compute(self):
        """Compute the sentence error rate."""
        return self.incorrect.float() / self.total

if __name__ == "__main__":
    test_ctc_label_converter(text = ["hello ", "world"])