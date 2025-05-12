from pytorch_lightning import LightningModule
import torch
from torch import nn
from loguru import logger
from torchmetrics.text import CharErrorRate, WordErrorRate
import time

from models import get_module
from utils import SentenceErrorRate

from utils import CTCLabelConverter_clovaai, AttnLabelConverter
from data.vocab import Vocab

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OCRModel(LightningModule):
    def __init__(
        self,
        batch_max_length,
        dali,
        backbone_name: str = "resnet18",
        seq_name: str = "bilstm",
        pred_name: str = "ctc",
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        save_dir: str = "checkpoints",
        train_data_path: str = "./training_images/",
    ):
        super().__init__()
        self.backbone_name = backbone_name if backbone_name is not None else "resnet18"
        self.seq_name = seq_name
        self.pred_name = pred_name
        path = os.path.join(train_data_path, "tgt.csv")
        self.vocab = Vocab().get_vocab()
        if self.pred_name == "ctc":
            self.converter = CTCLabelConverter_clovaai(self.vocab, device="cuda")
        else:
            self.converter = AttnLabelConverter(self.vocab, batch_max_length=batch_max_length)
        self._build_model()

        # TODO: add optimizer, scheduler, loss, metrics
        if self.pred_name == "ctc":
            self.loss = nn.CTCLoss(blank=0, zero_infinity=True)
        else:
            self.loss = nn.CrossEntropyLoss(ignore_index=0)

        # Initialize metrics
        self.cer = CharErrorRate()
        self.wer = WordErrorRate()
        self.ser = SentenceErrorRate()
        self.val_predictions = []
        self.val_targets = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.batch_max_length = batch_max_length
        self.save_dir = save_dir
        self.dali = dali
        self.val_epoch_start_time = 0  # Because the sanity check fails
        logger.info(f"{self.dali=}")

    def _build_model(self):
        logger.info(f"{self.backbone_name}")
        backbone_cls, seq_module_cls, pred_module_cls = get_module(self.backbone_name, self.seq_name, self.pred_name)

        # Initialize backbone
        self.backbone = backbone_cls(input_channels=3, output_channels=512)

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
            256 if self.seq_module else 512  # channels from VGG output
        )  # When no seq_module, use channel size from VGG
        self.pred_module = pred_module_cls(
            input_dim=input_dim,
            num_classes=len(self.converter.character),  # len(self.vocab),  # Number of classes including blank token
        )

    def forward(self, x, text):
        x = self.backbone(x)
        if self.seq_module:
            x = self.seq_module(x)
        x = self.pred_module(x, text=text, is_train=self.training, batch_max_length=self.batch_max_length)
        return x

    def on_train_epoch_start(self):
        # Start timing for training epoch
        self.train_epoch_start_time = time.time()
        logger.debug("Starting training epoch")

    def training_step(self, batch, batch_idx):
        images = batch["data"]
        text_encoded = batch["label"]
        text_lengths = batch["length"]

        if self.pred_name == "ctc":
            # Forward pass
            logits = self(images, text=text_encoded)
            # Prepare CTC input
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            # Calculate input lengths
            preds_size = torch.LongTensor([logits.size(1)] * images.size(0))
            # Calculate loss
            loss = self.loss(log_probs, text_encoded, preds_size, text_lengths)
        else:
            preds = self(images, text=text_encoded[:, :-1]).to(device)
            target = text_encoded[:, 1:].to(device)
            loss = self.loss(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["data"]
        text_encoded = batch["label"]
        text_lengths = batch["length"]
        # text_encoded, text_lengths = self.converter.encode(labels, batch_max_length=self.batch_max_length)

        labels = self.converter.decode(text_encoded, text_lengths)

        if self.pred_name == "ctc":
            # Forward pass
            logits = self(images, text=text_encoded)
            # Prepare CTC input
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            # Calculate input lengths
            input_lengths = torch.full(
                size=(logits.size(0),),
                fill_value=logits.size(1),
                dtype=torch.long,
                device=self.device,
            )
            # Calculate loss
            loss = self.loss(log_probs, text_encoded, input_lengths, text_lengths)

            # Get predictions for metrics
            preds = log_probs.argmax(2).permute(1, 0).detach().cpu()
            pred_texts = self.converter.decode(preds)

        else:
            # Attention model validation
            preds = self(images, text=text_encoded[:, :-1]).to(device)
            target = text_encoded[:, 1:].to(device)  # Shift target by 1 since we predict next char
            loss = self.loss(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # Get predictions for metrics
            pred_size = torch.LongTensor([preds.size(1)] * preds.size(0))
            _, pred_index = preds.max(2)
            pred_texts = self.converter.decode(pred_index, pred_size)

        # Store predictions and targets for epoch end metrics
        self.val_predictions.extend(pred_texts)
        self.val_targets.extend(labels)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        images = batch["data"]

        if self.pred_name == "ctc":
            # Forward pass
            logits = self(images, text=None)
            # Prepare CTC input
            log_probs = logits.log_softmax(2).permute(1, 0, 2)

            # Get predictions for metrics
            preds = log_probs.argmax(2).permute(1, 0).detach().cpu()
            pred_texts = self.converter.decode(preds)

        else:
            # Attention model validation
            preds = self(images, text=None).to(device)
            # Get predictions for metrics
            _, pred_index = preds.max(2)
            pred_texts = self.converter.decode(pred_index, None)

        return pred_texts

    def on_train_start(self):
        # Log hyperparameters to the logger
        hyperparams = {
            "backbone_name": self.backbone_name,
            "seq_name": self.seq_name,
            "pred_name": self.pred_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_max_length": self.batch_max_length,
            "max_epochs": self.trainer.max_epochs,
            "dali": self.dali,
        }

        self.logger.log_hyperparams(hyperparams)
        logger.info(f"Logged hyperparameters: {hyperparams}")

    def on_train_epoch_end(self):
        # Calculate and log training epoch time
        train_epoch_time = time.time() - self.train_epoch_start_time
        logger.info(f"Training epoch {self.current_epoch} completed in {train_epoch_time:.2f} seconds")
        self.log("train_epoch_time", train_epoch_time)

        # Save model after each training epoch
        epoch = self.current_epoch
        save_path = f"{self.save_dir}/model_train_epoch_{epoch}.ckpt"
        self.trainer.save_checkpoint(save_path)
        logger.info(f"Saved model checkpoint after training epoch {epoch} to {save_path}")
        # Reset
        if self.dali:
            self.trainer.datamodule.train_dataloader.reset()

    def on_validation_epoch_start(self):
        # Reset stored predictions and targets
        self.val_predictions = []
        self.val_targets = []

        # Start timing for validation epoch
        self.val_epoch_start_time = time.time()
        logger.debug("Starting validation epoch")

    def on_validation_epoch_end(self):
        # Calculate and log validation epoch time
        val_epoch_time = time.time() - self.val_epoch_start_time
        logger.info(f"Validation epoch {self.current_epoch} completed in {val_epoch_time:.2f} seconds")
        self.log("val_epoch_time", val_epoch_time)

        # Calculate and log CER
        cer = self.cer(self.val_predictions, self.val_targets)
        self.log("val_cer", cer, prog_bar=True)
        logger.info(f"Validation CER: {cer:.4f}")

        # Calculate and log WER
        wer = self.wer(self.val_predictions, self.val_targets)
        self.log("val_wer", wer, prog_bar=True)
        logger.info(f"Validation WER: {wer:.4f}")

        # Calculate and log SER
        ser = self.ser(self.val_predictions, self.val_targets)
        self.log("val_ser", ser, prog_bar=True)
        logger.info(f"Validation SER: {ser:.4f}")

        # Save model after validation with metrics in filename
        epoch = self.current_epoch
        val_loss = self.trainer.callback_metrics.get("val_loss", 0)
        save_path = f"{self.save_dir}/model_val_epoch_{epoch}_loss_{val_loss:.4f}_cer_{cer:.4f}_wer_{wer:.4f}.ckpt"
        self.trainer.save_checkpoint(save_path)
        logger.info(f"Saved model checkpoint after validation epoch {epoch} to {save_path}")

        # Clear predictions and targets
        self.val_predictions = []
        self.val_targets = []
        # Reset
        if self.dali:
            self.trainer.datamodule.val_dataloader.reset()

    def evaluate(self, batch, batch_idx):
        # TODO: implement evaluation
        pass

    def configure_optimizers(self):
        # Group parameters by module for different learning rates
        param_groups = [
            {"params": self.backbone.parameters(), "lr": self.learning_rate * 0.1},
            {"params": self.pred_module.parameters(), "lr": self.learning_rate},
        ]

        # Only add seq_module parameters if it exists
        if self.seq_module:
            param_groups.insert(1, {"params": self.seq_module.parameters(), "lr": self.learning_rate})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        # Use CosineAnnealingLR for learning rate scheduling
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate / 100,  # Minimum learning rate at the end of schedule
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
