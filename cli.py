from pytorch_lightning.cli import LightningCLI
from main import OCRModel
from test import DALI_OCRDataModule
# from dataloader import OCRDataModule, DALI_OCRDataModule
import os


class OCRTrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add arguments for time-based validation
        parser.add_argument("--validation_interval", type=float, 
                          help="Run validation every N hours during training")
        parser.add_argument("--save_dir", type=str, default="checkpoints",
                          help="Directory to save checkpoints")
        parser.link_arguments('model.batch_max_length', 'data.batch_max_length')

    def before_fit(self):
        #TODO: Fix argument parsing in lightning cli
        # Create checkpoint directory if it doesn't exist
        save_dir = self.config.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

def cli_main():
    OCRTrainingCLI(
        OCRModel, 
        DALI_OCRDataModule,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=42
    )

if __name__ == "__main__":
    cli_main()
