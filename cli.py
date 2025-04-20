from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import LearningRateMonitor
from main import OCRModel
from dataloader import OCRDataModule, DALI_OCRDataModule
import os


class OCRTrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--save_dir", type=str, default="checkpoints",
                          help="Directory to save checkpoints")
        parser.link_arguments('model.batch_max_length', 'data.batch_max_length')
        parser.link_arguments('model.pred_name', 'data.pred_name')
        parser.link_arguments('data.batch_size', 'model.batch_size')
        parser.link_arguments('data.train_data_path', 'model.train_data_path')

    def before_fit(self):
        save_dir = self.config.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        # Add LearningRateMonitor callback
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.trainer.callbacks.append(lr_monitor)

def cli_main():
    OCRTrainingCLI(
        OCRModel,
        DALI_OCRDataModule,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=42
    )

if __name__ == "__main__":
    cli_main()
