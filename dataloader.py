
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data import OCRDataset


class OCRDataModule(LightningDataModule):
    def __init__(self, train_data_path, val_data_path, batch_size=32, num_workers=4):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        self.train_data = OCRDataset(self.train_data_path)
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        self.val_data = OCRDataset(self.val_data_path)
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)