import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM


class AnnModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(32, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


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
        # loss = F.binary_cross_entropy(hypothesis, y)
        acc = FM.accuracy(torch.sigmoid(hypothesis), y.int())
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        # optimizer = torch.optim.SGD(self.parameters(), lr=3e-3)
        return optimizer
