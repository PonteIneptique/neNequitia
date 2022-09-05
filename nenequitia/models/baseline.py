from typing import Optional, Dict, Union

import torchmetrics
import torch
import torch.nn as nn

import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import ConfusionMatrixDisplay

from nenequitia.codecs import BaselineEncoder


class BaselineModule(pl.LightningModule):
    # https://github.com/lascivaroma/seligator/blob/main/seligator/modules/seq2vec/han.py
    def __init__(self, encoder: Union[BaselineEncoder, Dict], lr: float = 5e-3, training: bool = False):
        super(BaselineModule, self).__init__()
        if isinstance(encoder, dict):
            encoder = BaselineEncoder.from_hparams(encoder)
        self.encoder = encoder
        self.inp, self.out = encoder.shape

        self.hparams["encoder"] = self.encoder.to_hparams()
        self.hparams["bins"] = self.out

        self._lr: float = lr

        self.net = nn.Sequential(nn.Linear(self.inp, self.out), nn.Softmax(dim=-1))

        self.metric_accuracy: Optional[torchmetrics.Accuracy] = None
        self.metric_precision: Optional[torchmetrics.Precision] = None
        self.metric_recall: Optional[torchmetrics.Recall] = None
        self.metric_confusion: Optional[torchmetrics.ConfusionMatrix]
        if training:
            self.metric_recall = torchmetrics.Recall(average="macro", num_classes=self.out)
            self.metric_accuracy = torchmetrics.Accuracy(average="macro", num_classes=self.out)
            self.metric_precision = torchmetrics.Precision(average="macro", num_classes=self.out)
            self.metric_confusion = torchmetrics.ConfusionMatrix(normalize="true", num_classes=self.out)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return [optimizer], {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=2,
                threshold=2e-3,
                min_lr=1e-5
            ),
            "monitor": "Dev[Loss]"
        }

    def forward(self, inputs):
        return self.net(inputs)

    def training_step(self, train_batch, batch_idx):
        lines, truthes = train_batch
        preds = self(lines)
        loss = F.cross_entropy(preds, truthes)
        self.log('Train[Loss]', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        lines, truthes = val_batch
        preds = self(lines)
        # Compute Accuracy or Precision/Recall here
        loss = F.cross_entropy(preds, truthes)

        self.log('Dev[Loss]', loss, prog_bar=True, on_epoch=True)
        preds = preds.argmax(dim=-1)
        self.log('Dev[Acc]', self.metric_accuracy(preds, truthes), prog_bar=True, on_epoch=True)
        self.log('Dev[Pre]', self.metric_precision(preds, truthes), prog_bar=True, on_epoch=True)
        self.log('Dev[Rec]', self.metric_recall(preds, truthes), prog_bar=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        lines = batch
        preds = self(lines)
        preds = preds.argmax(dim=-1)
        return [
            self.encoder.ys[pred]
            for pred in preds.cpu().tolist()
        ]

    def test_step(self, batch, batch_idx):
        lines, truthes = batch
        preds = self(lines)
        # Compute Accuracy or Precision/Recall here
        loss = F.cross_entropy(preds, truthes)

        self.log('Test[Loss]', loss, prog_bar=True, on_epoch=True)
        preds = preds.argmax(dim=-1)
        self.log('Test[Acc]', self.metric_accuracy(preds, truthes), prog_bar=True, on_epoch=True)
        self.log('Test[Pre]', self.metric_precision(preds, truthes), prog_bar=True, on_epoch=True)
        self.log('Test[Rec]', self.metric_recall(preds, truthes), prog_bar=True, on_epoch=True)
        self.metric_confusion(preds, truthes)

    def on_test_end(self):
        conf_matrix = self.metric_confusion.confmat.cpu().numpy()
        confusion = conf_matrix/conf_matrix.sum(axis=1, keepdims=True)*100
        confusion = np.round(confusion, 0)
        confusion = ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=self.encoder.ys
        )

        figure, ax = plt.subplots(figsize=(10, 10), dpi=300)
        confusion.plot(ax=ax, values_format=".0f")
        plt.savefig("confusion.png")

        return conf_matrix, self.encoder.ys
