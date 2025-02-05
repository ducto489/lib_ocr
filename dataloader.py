from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data import OCRDataset
from data.collate import OCRCollator
from loguru import logger


class OCRDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: str = "./training_images/",
        val_data_path: str = "./validation_images/",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        logger.debug(f"{train_data_path=}")
        logger.debug(f"{val_data_path=}")
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collator = OCRCollator()

    def train_dataloader(self):
        self.train_data = OCRDataset(self.train_data_path)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        self.val_data = OCRDataset(self.val_data_path)
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
