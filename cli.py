from pytorch_lightning.cli import LightningCLI
from main import OCRModel
from dataloader import OCRDataModule


class OCRTrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--train_dir", type=str, required=True, help="Path to training images"
        )
        parser.add_argument(
            "--val_dir", type=str, required=True, help="Path to validation images"
        )
        parser.add_argument(
            "--backbone", type=str, default="resnet18", help="Backbone network"
        )
        parser.add_argument(
            "--seq_module", type=str, default="bilstm", help="Sequence module"
        )
        parser.add_argument(
            "--pred_module", type=str, default="ctc", help="Prediction module"
        )
        parser.add_argument(
            "--batch_size", type=int, default=64, help="Batch size for training"
        )
        parser.add_argument(
            "--learning_rate", type=float, default=1e-3, help="Learning rate"
        )
        parser.add_argument(
            "--max_epochs", type=int, default=30, help="Maximum number of epochs"
        )


def cli_main():
    OCRTrainingCLI(OCRModel, OCRDataModule, seed_everything_default=42, run=True)
    # OCRTrainingCLI(OCRModel, OCRDataModule)


if __name__ == "__main__":
    cli_main()
