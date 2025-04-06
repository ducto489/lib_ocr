from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

from data import OCRDataset
from data.collate import OCRCollator
from data.augmentations import data_transforms
from data.dataset import process_tgt
from loguru import logger
import os
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

class LightningWrapper(DALIClassificationIterator):
            def __init__(self, dataset_size, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.dataset_size = dataset_size

            def __len__(self):
                return self.dataset_size

            def __next__(self):
                batch = super().__next__()[0]
                print(batch)
                x, target = batch["data"], batch["label"]
                target = target.squeeze(-1).long()
                x = x.detach().clone()
                target = target.detach().clone()
                return x, target
            
            def __getitem__(self, idx):
                return self.__next__()
            
            def __code__(self):
                return super().__code()

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
        self.train_data_path = os.path.join(self.train_data_path, "images")
        self.val_data_path = os.path.join(self.val_data_path, "images")

    def train_dataloader(self):
        logger.debug("Building train DALI pipelines...")
        train_pipeline = self.get_dali_train_pipeline()
        train_pipeline.build()
        logger.debug("Train DALI pipelines built.")
            
        self.train_dataloader = LightningWrapper(
            pipelines=train_pipeline, 
            reader_name="Reader", 
            dataset_size=self.steps_per_epoch,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP
        )
        return self.train_dataloader
    
    def val_dataloader(self):
        logger.debug("Building val DALI pipelines...")
        val_pipeline = self.get_dali_val_pipeline()
        val_pipeline.build()
        logger.debug("Val DALI pipelines built.")
        self.val_dataloader = LightningWrapper(
            pipelines=val_pipeline, 
            reader_name="Reader", 
            dataset_size=len(self.val_images_names) // self.batch_size,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP
        )
        return self.val_dataloader

    @pipeline_def(num_threads=4, batch_size=32, device_id=0)
    def get_dali_train_pipeline(self):
        images, indices = fn.readers.file(file_root=self.train_data_path, files=self.train_images_names, labels=list(range(len(self.train_images_names))), random_shuffle=True, name="Reader")
        images = fn.decoders.image(images, device="mixed")
        images = fn.resize(images, resize_y=100) 
        images = fn.normalize(images, dtype=types.UINT8)
        images = fn.pad(images, fill_value=0)
        indices = indices.gpu()
        return images, indices

    @pipeline_def(num_threads=4, batch_size=32, device_id=0)
    def get_dali_val_pipeline(self):
        images, indices = fn.readers.file(file_root=self.val_data_path, files=self.val_images_names, labels=list(range(len(self.val_images_names))), random_shuffle=False, name="Reader")
        images = fn.decoders.image(images, device="mixed")
        images = fn.resize(images, resize_y=100) 
        images = fn.normalize(images, dtype=types.UINT8)
        images = fn.pad(images, fill_value=0)
        indices = indices.gpu()
        return images, indices



if __name__ == '__main__':
    data = DALI_OCRDataModule
    data.__init__
    x = data.val_dataloader()
    print(x)
    for meta_data in x:
            # print(meta_data[0]["data"].shape)
            logger.debug(f"{meta_data=}")
            break
    
