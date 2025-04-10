from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

from data import OCRDataset
from data.collate import OCRCollator
from data.augmentations import data_transforms
from data.dataset import process_tgt
from utils import AttnLabelConverter, CTCLabelConverter
from data.vocab import Vocab
from loguru import logger
import os
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from PIL import Image
import numpy as np
from torchvision.io import decode_image
import torch
import random

class OCRDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: str = "./training_images/",
        val_data_path: str = "./validation_images/",
        batch_size: int = 32,
        num_workers: int = 4,
        batch_max_length: int = 50,
        pred_name: str = "attn"
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

class LightningWrapper(DALIGenericIterator):
    def __init__(self, pipelines, dataset_size, *args, **kwargs):
        super().__init__(pipelines = pipelines, *args, **kwargs)
        self.pipelines = pipelines
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __next__(self):
        batch = super().__next__()[0]
        # logger.debug(f"{len(batch)=}")
        x, target, length = batch["data"], batch["label"], batch["length"]
        # logger.debug(f"{target.size()=}")
        # From DALI to Torchvision format
        x = x.permute(0, 3, 1, 2)
        # target = target.squeeze(-1).int()
        x = x.detach().clone()
        target = target.detach().clone()
        # logger.debug(f"{target.size()=}")
        return x, target, length
    
    # def __getitem__(self, idx):
    #     return self.__next__()
    
    def __code__(self):
        return super().__code()
            
class ExternalInputCallable(object):
    def __init__(self, steps_per_epoch, data_path, converter, images_names, labels, transform=None, batch_max_length=50, batch_size=32):
        self.data_path = data_path
        self.transform = transform
        self.batch_max_length = batch_max_length
        self.steps_per_epoch = steps_per_epoch
        self.converter = converter
        self.batch_size = batch_size

        self.images_names = images_names
        self.labels = labels

        self.data = list(zip(images_names, labels))
        random.shuffle(self.data)

    def __call__(self, sample_info):
        # idx = sample_info.idx_in_epoch
        idx = sample_info.idx_in_epoch
        if idx >= len(self.data):
            logger.debug(f"Trigger skip with {idx=} and {len(self.data)=}")
            # Indicate end of the epoch
            raise StopIteration()
        
        max_retries = 5
        retries = 0
        success = False
        
        while not success and retries < max_retries:
            try:
                image_name, label = self.data[idx % len(self.data)]
                image_path = os.path.join(self.data_path, image_name)
                
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                    
                try:
                    image = np.fromfile(image_path, dtype=np.uint8)
                    encoded_label, length = self.converter.encode([label])
                    # logger.debug(f"{encoded_label.size()=}")
                    success = True
                    
                except (OSError, IOError) as e:
                    print(f"Warning: Skipping corrupted image {image_path}: {str(e)}")
                    idx = (idx + 1) % len(self.data)
                    retries += 1
                    
            except (IOError, FileNotFoundError) as e:
                print(f"Error loading image at index {idx}: {str(e)}")
                idx = (idx + 1) % len(self.data)
                retries += 1
        
        if not success:
            raise RuntimeError(f"Failed to load a valid image after {max_retries} attempts starting from index {idx}")
        
        return image, torch.squeeze(encoded_label), length

class DALI_OCRDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: str = "./training_images/",
        val_data_path: str = "./validation_images/",
        batch_size: int = 32,
        num_workers: int = 4,
        batch_max_length: int = 50,
        pred_name: str = "attn",
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

        logger.debug("Get Vocab")
        vocab = Vocab("/hdd1t/mduc/data/train/tgt.csv").get_vocab_csv()
        logger.debug(f"{pred_name=}")
        if pred_name=="ctc":
            self.converter = CTCLabelConverter(vocab, device="cpu")
        else:
            self.converter = AttnLabelConverter(vocab, device="cpu")
        logger.debug("Processing tgt.csv file")
        self.train_images_names, self.train_labels = process_tgt(self.train_data_path, batch_max_length=self.batch_max_length)
        self.val_images_names, self.val_labels = process_tgt(self.val_data_path, batch_max_length=self.batch_max_length)
        logger.debug("Done!")
        # self.train_labels = converter.encode(self.train_labels, batch_max_length=self.batch_max_length)
        # self.val_labels = converter.encode(self.val_labels, batch_max_length=self.batch_max_length)
        # logger.debug("Done!")
        logger.debug(f"{len(self.train_images_names)=}")
        logger.debug(f"{self.batch_size=}")
        self.steps_per_epoch = len(self.train_images_names) // self.batch_size
        logger.debug(f"{self.steps_per_epoch=}")
        self.train_data_path = os.path.join(self.train_data_path, "images")
        self.val_data_path = os.path.join(self.val_data_path, "images")

    def train_dataloader(self):
        logger.debug("Building train DALI pipelines...")
        train_pipeline = self.get_dali_train_pipeline(batch_size=self.batch_size)
        train_pipeline.build()
        logger.debug("Train DALI pipelines built.")
        # self.train_dataloader = DALIClassificationIterator(
        #     pipelines=train_pipeline,
        #     auto_reset=True,
        # )

        self.train_dataloader = LightningWrapper(
            pipelines=train_pipeline, 
            output_map=["data", "label", "length"],
            dataset_size=self.steps_per_epoch,
            auto_reset=False,
            last_batch_policy=LastBatchPolicy.FILL,
            # dynamic_shape=True
        )
        # self.train_dataloader.pipelines.run()
        return self.train_dataloader
    
    def val_dataloader(self):
        logger.debug("Building val DALI pipelines...")
        val_pipeline = self.get_dali_val_pipeline(batch_size=self.batch_size)
        val_pipeline.build()
        logger.debug("Val DALI pipelines built.")
        # self.val_dataloader = DALIClassificationIterator(
        #     pipelines=val_pipeline,
        #     auto_reset=True,
        # )
        self.val_dataloader = LightningWrapper(
            pipelines=val_pipeline, 
            output_map=["data", "label", "length"],
            dataset_size=len(self.val_images_names) // self.batch_size,
            auto_reset=False,
            last_batch_policy=LastBatchPolicy.FILL,
            # dynamic_shape=True
        )
        return self.val_dataloader

    @pipeline_def(num_threads=4, batch_size=32, device_id=0, py_start_method="spawn")#, exec_dynamic=True)
    def get_dali_train_pipeline(self):
        # images, _ = fn.readers.file(file_root=self.val_data_path, files=self.val_data_path, random_shuffle=False, name="Reader")
        images, indices, length = fn.external_source(
            source=ExternalInputCallable(
                steps_per_epoch = self.steps_per_epoch,
                data_path = self.train_data_path,
                converter = self.converter,
                transform=data_transforms["train"],
                batch_max_length=self.batch_max_length,
                images_names = self.train_images_names, 
                labels = self.train_labels,
                batch_size=self.batch_size                
            ),
            num_outputs=3,
            batch=False,
            parallel=True,
            dtype=[types.UINT8, types.INT64, types.INT64],
        )
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_y=100, dtype=types.FLOAT) 
        images = fn.normalize(images, dtype=types.FLOAT)
        # images = images.gpu()
        indices = indices.gpu()
        # images = fn.cast(images, dtype=types.FLOAT)
        images = fn.pad(images, fill_value=0)
        indices = fn.pad(indices, fill_value=0)
        length = fn.pad(length, fill_value=0)
        return images, indices, length

    @pipeline_def(num_threads=4, batch_size=32, device_id=0, py_start_method="spawn")#, exec_dynamic=True)
    def get_dali_val_pipeline(self):
        # images, _ = fn.readers.file(file_root=self.val_data_path, files=self.val_data_path, random_shuffle=False, name="Reader")
        images, indices, length = fn.external_source(
            source=ExternalInputCallable(
                steps_per_epoch = len(self.val_images_names) // self.batch_size,
                data_path = self.val_data_path,
                converter = self.converter,
                transform=data_transforms["val"],
                batch_max_length=self.batch_max_length,
                images_names = self.val_images_names, 
                labels = self.val_labels,
                batch_size=self.batch_size
            ),
            num_outputs=3,
            batch=False,
            parallel=True,
            dtype=[types.UINT8, types.INT64, types.INT64],
        )
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_y=100, dtype=types.FLOAT) 
        images = fn.normalize(images, dtype=types.FLOAT)
        # images = images.gpu()
        indices = indices.gpu()
        # images = fn.cast(images, dtype=types.FLOAT)
        images = fn.pad(images, fill_value=0)
        indices = fn.pad(indices, fill_value=0)
        length = fn.pad(length, fill_value=0)
        return images, indices, length

if __name__ == '__main__':
    data = DALI_OCRDataModule
    data.__init__
    x = data.val_dataloader()
    print(x)
    for meta_data in x:
            # print(meta_data[0]["data"].shape)
            logger.debug(f"{meta_data=}")
            break

