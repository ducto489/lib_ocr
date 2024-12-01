from pytorch_lightning import LightningModule
from models import get_module


class OCRModel(LightningModule):
    def __init__(self, backbone_name, seq_name, pred_name):
        super().__init__()
        self.backbone_name = backbone_name
        self.seq_name = seq_name
        self.pred_name = pred_name

        self._build_model()

        # TODO: add optimizer, scheduler, loss, metrics
        self.loss = ...
        self.metric = ...

    def _build_model(self):
        self.backbone, self.seq_module, self.pred_module = get_module(self.backbone_name, self.seq_name, self.pred_name)

    def forward(self, x):
        x = self.backbone(x)
        if self.seq_module:
            x = self.seq_module(x)
        x = self.pred_module(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log('train_loss', loss)
        # decode y_hat -> greedy/beam search
        # TODO: log images + predictions prob 10%
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log('val_loss', loss)
        # decode y_hat -> greedy/beam search
        # TODO: log images + predictions prob 10%
        return loss

    def on_validation_epoch_end(self):
        # TODO: log accumulated metrics -> CER > WER > SER

        pass





