from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import LastBatchPolicy

from data import OCRDataset, OCRCollator, data_transforms, process_tgt, Vocab, ExternalInputCallable, LightningWrapper
from utils import AttnLabelConverter, CTCLabelConverter
from loguru import logger
import os
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn


class OCRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_max_length,
        dali: bool = False,
        train_data_path: str = "./training_images/",
        val_data_path: str = "./validation_images/",
        batch_size: int = 32,
        num_workers: int = 4,
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
        self.dali = dali
        self.pred_name = pred_name
        self.collator = OCRCollator()

        # Save hyperparameters for logging
        self.save_hyperparameters()

    def train_dataloader(self):
        self.train_data = OCRDataset(
            self.train_data_path,
            transform=data_transforms["train"],
            batch_max_length=self.batch_max_length,
            pred_name=self.pred_name
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
            batch_max_length=self.batch_max_length,
            pred_name=self.pred_name
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
        batch_max_length,
        dali : bool = True,
        train_data_path: str = "./training_images/",
        val_data_path: str = "./validation_images/",
        batch_size: int = 32,
        num_workers: int = 4,
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
        self.dali = dali
        self.pred_name = pred_name

        # Save hyperparameters for logging
        self.save_hyperparameters()

        logger.debug("Get Vocab")
        path = os.path.join(self.train_data_path, "tgt.csv")
        vocab = Vocab(path).get_vocab_csv()
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
        train_pipeline = self.get_dali_train_pipeline(batch_size=self.batch_size, num_threads=self.num_workers)
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
        val_pipeline = self.get_dali_val_pipeline(batch_size=self.batch_size, num_threads=self.num_workers)
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

    @pipeline_def(num_threads=8, batch_size=32, device_id=0, py_start_method="spawn", exec_dynamic=True)
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
            prefetch_queue_depth=2,
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

    @pipeline_def(num_threads=8, batch_size=32, device_id=0, py_start_method="spawn", exec_dynamic=True)
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
            prefetch_queue_depth=2,
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

    @pipeline_def(num_threads=8, batch_size=32, device_id=0, py_start_method="spawn", exec_dynamic=True)
    def get_dali_train_pipeline_aug(self):
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
            prefetch_queue_depth=2,
        )
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_y=100, dtype=types.FLOAT)

        # Apply brightness and contrast adjustment
        images = fn.brightness_contrast(images,
                                       brightness=0.2,  # Brightness adjustment factor
                                       contrast=0.2)    # Contrast adjustment factor

        # Apply HSV color space adjustment
        images = fn.hsv(images,
                       hue=0.1,        # Hue adjustment
                       saturation=0.2, # Saturation adjustment
                       value=0.1)      # Value adjustment

        # TODO: Use a better affine matrix
        # Apply affine transformation
        images = fn.warp_affine(images,
                                matrix=[1.0, 0.1, 0.0, 0.1, 1.0, 0.0],  # 2x3 affine transformation matrix
                                fill_value=0,
                                interp_type=types.INTERP_NN)

        images = fn.normalize(images, dtype=types.FLOAT)
        # images = images.gpu()
        indices = indices.gpu()
        # images = fn.cast(images, dtype=types.FLOAT)
        images = fn.pad(images, fill_value=0)
        indices = fn.pad(indices, fill_value=0)
        length = fn.pad(length, fill_value=0)
        return images, indices, length