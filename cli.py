from pytorch_lightning.cli import LightningCLI
from main import OCRModel
from utils import TimeBasedValidationCallback
from dataloader import OCRDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os


class OCRTrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add arguments for time-based validation
        parser.add_argument("--validation_interval", type=float, default=4.0, 
                          help="Run validation every N hours during training")
        parser.add_argument("--save_dir", type=str, default="checkpoints",
                          help="Directory to save checkpoints")
    
    def before_fit(self):
        # Create checkpoint directory if it doesn't exist
        save_dir = self.config.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        
        # Add time-based validation callback
        validation_interval = self.config.get("validation_interval", 4.0)
        time_validation_callback = TimeBasedValidationCallback(validation_interval=validation_interval)
        self.trainer.callbacks.append(time_validation_callback)


def cli_main():
    OCRTrainingCLI(
        OCRModel, 
        OCRDataModule,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=42
    )


if __name__ == "__main__":
    cli_main()
