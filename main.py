from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from loguru import logger

from models import get_module
from dataloader import OCRDataModule

## CHECK VOCAB, CONVERTER AGAIN
from utils import CTCLabelConverter_clovaai
from data.vocab import Vocab


class OCRModel(LightningModule):
    def __init__(
        self,
        backbone_name,
        seq_name,
        pred_name,
        batch_size,
        learning_rate,
        weight_decay,
        epochs,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.seq_name = seq_name
        self.pred_name = pred_name
        logger.debug(f"{self.backbone_name=}")
        logger.debug(f"{backbone_name=}")
        # self.vocab = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~ '
        self.vocab = Vocab("./training_images/labels.json").get_vocab()
        # breakpoint()
        self.converter = CTCLabelConverter_clovaai(self.vocab, device="cuda")
        self._build_model()

        # TODO: add optimizer, scheduler, loss, metrics
        self.loss = nn.CTCLoss(blank=0, zero_infinity=True)
        self.metric = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay

    def _build_model(self):
        # self.backbone, self.seq_module, self.pred_module = get_module(
        #     self.backbone_name, self.seq_name, self.pred_name)
        logger.info(f"{self.backbone_name}")
        backbone_cls, seq_module_cls, pred_module_cls = get_module(
            self.backbone_name, self.seq_name, self.pred_name
        )

        # Initialize backbone
        self.backbone = backbone_cls()

        # Initialize sequence module if provided
        self.seq_module = None
        if seq_module_cls is not None:
            self.seq_module = seq_module_cls(
                input_size=512,  # ResNet backbone output channels
                hidden_size=256,  # Hidden size for LSTM
                output_size=256,  # Output size from sequence module
            )

        # Initialize prediction module
        input_dim = (
            256 if self.seq_module else 512
        )  # Use seq_module output size if exists, else backbone output size
        self.pred_module = pred_module_cls(
            input_dim=input_dim,
            num_classes=len(self.vocab) + 1,  # Number of classes including blank token
        )

    def forward(self, x):
        x = self.backbone(x)
        if self.seq_module:
            x = self.seq_module(x)
        x = self.pred_module(x)
        return x

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat = self.forward(x)
        # loss = self.loss(y_hat, y)

        # self.log('train_loss', loss)
        # # decode y_hat -> greedy/beam search

        # # TODO: log images + predictions prob 10%
        # return loss
        images = batch["images"]
        labels = batch["labels"]

        text_encoded, text_lengths = self.converter.encode(labels, batch_max_length=50)

        # Forward pass
        logits = self(images)
        # Prepare CTC input
        log_probs = logits.log_softmax(2).permute(1, 0, 2)

        # Calculate input lengths
        preds_size = torch.IntTensor([logits.size(1)] * images.size(0))

        # Calculate loss
        loss = self.loss(log_probs, text_encoded, preds_size, text_lengths)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)

        # Log sample predictions periodically
        # if batch_idx % 100 == 0:
        #     self._log_predictions(logits, labels, prefix='train')

        return loss

    def validation_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat = self.forward(x)
        # loss = self.loss(y_hat, y)

        # self.log('val_loss', loss)
        # # decode y_hat -> greedy/beam search
        # # TODO: log images + predictions prob 10%
        # return loss

        # Calculate loss (same as training)

        images = batch["images"]
        labels = batch["labels"]

        logits = self(images)
        log_probs = logits.log_softmax(2).permute(1, 0, 2)
        # text_encoded = [
        #     torch.tensor(self.converter.encode(label)) for label in labels
        # ]

        # text_lengths = torch.tensor([len(t)
        #                              for t in text_encoded]).to(self.device)
        text_encoded, text_lengths = self.converter.encode(labels)
        max_length = max(text_lengths)
        text_padded = torch.zeros(len(text_encoded), max_length).long()
        # for i, t in enumerate(text_encoded):
        #     text_padded[i, :len(t)] = t
        text_padded = text_padded.to(self.device)

        input_lengths = torch.full(
            size=(logits.size(0),),
            fill_value=logits.size(1),
            dtype=torch.long,
            device=self.device,
        )

        loss = self.loss(log_probs, text_padded, input_lengths, text_lengths)

        self.log("val_loss", loss, prog_bar=True)

        # Log predictions for visualization
        # if batch_idx % 50 == 0:
        #     self._log_predictions(logits, labels, prefix='val')

        return loss

    def on_validation_epoch_end(self):
        # TODO: log accumulated metrics -> CER > WER > SER

        pass

    def configure_optimizers(self):
        optimizer = Adam(
            [
                {"params": self.backbone.parameters(), "lr": self.learning_rate * 0.1},
                {"params": self.seq_module.parameters(), "lr": self.learning_rate},
                {"params": self.pred_module.parameters(), "lr": self.learning_rate},
            ],
            weight_decay=self.weight_decay,
        )

        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=[self.learning_rate * 0.1, self.learning_rate, self.learning_rate],
        #     epochs=self.trainer.max_epochs,
        #     steps_per_epoch=len(self.train_dataloader),
        #     pct_start=0.1,
        #     div_factor=25,
        #     final_div_factor=1000,
        #     anneal_strategy='cos'
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step"
            # }
        }


def main():
    # Initialize wandb logger
    wandb_logger = WandbLogger(project="vietnamese-ocr", name="resnet18-bilstm-ctc")

    # Initialize model
    model = OCRModel(
        backbone_name="resnet18",
        seq_name="bilstm",
        pred_name="ctc",
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-5,
        epochs=4,
    )

    # Initialize data module
    data_module = OCRDataModule(
        train_data_path="./training_images",
        val_data_path="./training_images",
        batch_size=64,
        num_workers=4,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="ocr-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize trainer
    trainer = Trainer(
        max_epochs=30,
        accelerator="cuda",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=5.0,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
    )

    # Start training
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
