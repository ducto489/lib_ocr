from pytorch_lightning.cli import LightningCLI
from main import OCRModel
from dataloader import OCRDataModule


class OCRTrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--backbone_name', type=str, default='resnet18')
        parser.add_argument('--seq_name', type=str, default='bilstm')
        parser.add_argument('--pred_name', type=str, default='ctc')


def cli_main():
    OCRTrainingCLI(OCRModel, OCRDataModule, seed_everything_default=42)


if __name__ == '__main__':
    cli_main()
