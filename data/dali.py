from loguru import logger
import os
import io
import random

from nvidia.dali.plugin.pytorch import DALIGenericIterator
from PIL import Image
import numpy as np
import torch

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
    
    def __code__(self):
        return super().__code()
            
class ExternalInputCallable(object):
    def __init__(self, steps_per_epoch, data_path, converter, images_names, labels, batch_max_length, transform=None, batch_size=32):
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
                    with open(image_path, 'rb') as f:
                        file_bytes = f.read()
                    bytes_io = io.BytesIO(file_bytes)

                    # image verify() don't work so use load()
                    with Image.open(bytes_io) as img:
                        img.load()

                    image = np.frombuffer(file_bytes, dtype=np.uint8)
                    encoded_label, length = self.converter.encode([label], batch_max_length=self.batch_max_length)
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