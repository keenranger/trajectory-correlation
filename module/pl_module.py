import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM


class BinaryClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        hypothesis = self.model(x)
        loss = self.criterion(hypothesis, y)
        # loss = F.binary_cross_entropy(hypothesis, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        hypothesis = self.model(x)
        loss = self.criterion(hypothesis, y)
        acc = FM.accuracy(torch.sigmoid(hypothesis), y.int())
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        hypothesis = self.model(x)
        loss = self.criterion(hypothesis, y)
        acc = FM.accuracy(torch.sigmoid(hypothesis), y.int())
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        return optimizer
