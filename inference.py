from data import PredictLightningWrapper
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from pytorch_lightning import LightningDataModule
import numpy as np
import torch

from main import OCRModel


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


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OCR Inference Script")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image or directory of images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    inference = Inference(image_path=args.image_path)
    predict_dataloader = inference.predict_dataloader()
    model = OCRModel.load_from_checkpoint(
        args.checkpoint, strict=True, batch_max_length=200, dali=True, map_location="cuda", pred_name="attn"
    )
    model.eval()

    for batch in predict_dataloader:
        preds = model.predict_step(batch, 0)
        print(preds)
