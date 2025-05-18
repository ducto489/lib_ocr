from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from pytorch_lightning import LightningDataModule
import numpy as np


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


class Inference(LightningDataModule):
    def __init__(self, image_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_path = image_path

        self.MEAN = np.asarray([0.485, 0.456, 0.406])[None, None, :]
        self.STD = np.asarray([0.229, 0.224, 0.225])[None, None, :]
        self.SCALE = 1 / 255.0

    def predict_dataloader(self):
        predict_pipeline = self.get_dali_predict_pipeline()
        predict_pipeline.build()
        self.predict_dataloader = PredictLightningWrapper(
            pipelines=predict_pipeline,
            output_map=["data"],
        )
        return self.predict_dataloader

    @pipeline_def(
        num_threads=8,
        batch_size=1,
        device_id=0,
        py_start_method="spawn",
    )
    def get_dali_predict_pipeline(self):
        image, _ = fn.file_reader(file_root=self.image_path, shard_id=0, num_shards=1)
        image = fn.decoders.image(image, device="mixed", output_type=types.RGB)
        image = fn.resize(image, device="gpu", resize_y=100, dtype=types.FLOAT)
        image = fn.normalize(
            image, device="gpu", dtype=types.FLOAT, mean=self.MEAN / self.SCALE, stddev=self.STD, scale=self.SCALE
        )
        image = fn.pad(image, fill_value=0)
        return image
