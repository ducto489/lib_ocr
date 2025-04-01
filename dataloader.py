from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data import OCRDataset
from data.collate import OCRCollator
from data.augmentations import data_transforms
from data.dataset import process_tgt
from loguru import logger

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class OCRDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: str = "./training_images/",
        val_data_path: str = "./validation_images/",
        batch_size: int = 32,
        num_workers: int = 4,
        batch_max_length: int = 50,
    ):
        logger.debug(f"{train_data_path=}")
        logger.debug(f"{val_data_path=}")
        logger.debug(f"{batch_size=}")
        logger.debug(f"{num_workers=}")
        logger.debug(f"{batch_max_length=}")

        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_max_length = batch_max_length
        self.collator = OCRCollator()

    def train_dataloader(self):
        self.train_data = OCRDataset(
            self.train_data_path,
            transform=data_transforms["train"],
            batch_max_length=self.batch_max_length
        )
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        self.val_data = OCRDataset(
            self.val_data_path,
            transform=data_transforms["val"],
            batch_max_length=self.batch_max_length
        )
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            persistent_workers=True,
        )

class DALI_OCRDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: str = "./training_images/",
        val_data_path: str = "./validation_images/",
        batch_size: int = 32,
        num_workers: int = 4,
        batch_max_length: int = 50,
    ):
        logger.debug(f"{train_data_path=}")
        logger.debug(f"{val_data_path=}")
        logger.debug(f"{batch_size=}")
        logger.debug(f"{num_workers=}")
        logger.debug(f"{batch_max_length=}")

        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_max_length = batch_max_length
        self.train_images_names, self.train_labels = process_tgt(self.train_data_path, batch_max_length=self.batch_max_length)
        self.val_images_names, self.val_labels = process_tgt(self.val_data_path, batch_max_length=self.batch_max_length)
        self.steps_per_epoch = len(self.train_images_names) // self.batch_size

    @pipeline_def(num_threads=4, batch_size=32, device_id=0)
    def get_dali_train_pipeline(self):
        images, indices = fn.readers.file(file_root=self.train_data_path, files=self.train_images_names, labels=list(range(len(self.train_images_names))), random_shuffle=True, name="Reader")
        images = fn.decoders.image(images, device="mixed")
        images = fn.resize(images, resize_y=100) 
        images = fn.normalize(images, scale=64, shift=128, dtype=types.UINT8)
        return images, indices

    @pipeline_def(num_threads=4, batch_size=32, device_id=0)
    def get_dali_val_pipeline(self):
        images, indices = fn.readers.file(file_root=self.val_data_path, files=self.val_images_names, labels=list(range(len(self.val_images_names))), random_shuffle=False, name="Reader")
        images = fn.decoders.image(images, device="mixed")
        images = fn.resize(images, resize_y=100) 
        images = fn.normalize(images, scale=64, shift=128, dtype=types.UINT8)
        return images, indices

    def train_dataloader(self):
        return DALIGenericIterator(
            [self.get_dali_train_pipeline()],
            ["images", "indices"],
        )

    def val_dataloader(self):
        return DALIGenericIterator(
            [self.get_dali_val_pipeline()],
            ["images", "indices"],
        )
