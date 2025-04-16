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

    def before_fit(self):
        #TODO: Fix argument parsing in lightning cli
        # Create checkpoint directory if it doesn't exist
        save_dir = self.config.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        # Add LearningRateMonitor callback
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.trainer.callbacks.append(lr_monitor)

def cli_main():
    # Get command line arguments
    # import sys
    # args = sys.argv

    # # Check if any argument contains "data.dali"
    # use_dali = any("data.dali" in arg and "True" in arg for arg in args)

    # # Select the appropriate data module
    # data_module = DALI_OCRDataModule if use_dali else OCRDataModule

    OCRTrainingCLI(
        OCRModel,
        DALI_OCRDataModule,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=42
    )

if __name__ == "__main__":
    cli_main()
