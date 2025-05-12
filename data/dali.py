from loguru import logger
import os
import random

from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np
import torch


class LightningWrapper(DALIGenericIterator):
    def __init__(self, pipelines, dataset_size, *args, **kwargs):
        super().__init__(pipelines=pipelines, *args, **kwargs)
        self.pipelines = pipelines
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __next__(self):
        batch = super().__next__()[0]

        batch["data"] = batch["data"].permute(0, 3, 1, 2)
        batch["data"] = batch["data"].detach().clone()
        batch["label"] = batch["label"].detach().clone()
        return batch

    def __code__(self):
        return super().__code()


class PredictLightningWrapper(DALIGenericIterator):
    def __init__(self, pipelines, *args, **kwargs):
        super().__init__(pipelines=pipelines, *args, **kwargs)
        self.pipelines = pipelines

    def __next__(self):
        batch = super().__next__()[0]

        batch["data"] = batch["data"].permute(0, 3, 1, 2)
        batch["data"] = batch["data"].detach().clone()
        return batch

    def __code__(self):
        return super().__code()


class ExternalInputCallable(object):
    def __init__(self, steps_per_epoch, data_path, converter, images_names, labels, batch_size=32):
        self.data_path = data_path
        self.steps_per_epoch = steps_per_epoch
        self.converter = converter
        self.batch_size = batch_size

        self.images_names = images_names
        self.labels = labels

        self.data = list(zip(images_names, labels))
        random.shuffle(self.data)

    def __call__(self, sample_info):
        idx = sample_info.idx_in_epoch
        if idx >= len(self.data):
            logger.debug(f"Trigger skip with {idx=} and {len(self.data)=}")
            # Indicate end of the epoch
            raise StopIteration()
        image_name, label = self.data[idx % len(self.data)]
        image_path = os.path.join(self.data_path, image_name)

        with open(image_path, "rb") as f:
            file_bytes = f.read()

        image = np.frombuffer(file_bytes, dtype=np.uint8)
        encoded_label, length = self.converter.encode([label])
        return image, torch.squeeze(encoded_label), length
