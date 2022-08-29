from typing import Optional

import torchmetrics
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .codecs import LabelEncoder
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LstmModule(pl.LightningModule):
    def __init__(self, encoder: LabelEncoder, bins: int,
                 emb_size: int = 128,
                 hid_size: int = 128,
                 dropout: float = .1,
                 use_highway: bool = True,
                 lr: float = 5e-3,
                 training: bool = False):
        super(LstmModule, self).__init__()
        self.encoder = encoder
        self.inp = len(encoder) + 4
        self.out = bins

        self._lr: float = lr
        self._dropout: float = dropout
        self._emb_size: int = emb_size
        self._hid_size: int = hid_size
        self._use_highway: bool = use_highway

        self._emb = nn.Sequential(
            nn.Embedding(self.inp, self._emb_size),
            nn.Dropout(self._dropout)
        )
        self._lstm = nn.LSTM(self._emb_size, hidden_size=self._hid_size, bidirectional=True, batch_first=True)

        if use_highway:
            self._lin = nn.Sequential(
                nn.Dropout(self._dropout),
                nn.Linear(self._hid_size * 4, self._hid_size * 2),  # HID_SIZE*4 because CONCAT first and last
                nn.Linear(self._hid_size * 2, self.out)
            )
        else:
            self._lin = nn.Sequential(
                nn.Dropout(self._dropout),
                nn.Linear(self._hid_size * 4, self.out),
                nn.Softmax(dim=-1)
            )

        self.accuracy: Optional[torchmetrics.Accuracy] = None
        self.precision: Optional[torchmetrics.Precision] = None
        self.recall: Optional[torchmetrics.Recall] = None
        if training:
            self.recall = torchmetrics.Recall()
            self.accuracy = torchmetrics.Accuracy()
            self.precision = torchmetrics.Precision()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        (lines, lengths), truthes = train_batch
        preds = self(lines, lengths, softmax=False)
        loss = F.cross_entropy(preds, truthes)
        self.log('Train[Loss]', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):

        (lines, lengths), truthes = val_batch
        preds = self(lines, lengths, softmax=False)
        # Compute Accuracy or Precision/Recall here
        loss = F.cross_entropy(preds, truthes)
        self.log('Dev[Loss]', loss)
        preds = preds.argmax(dim=-1)
        self.log('Dev[Acc]', self.accuracy(preds, truthes))
        self.log('Dev[Pre]', self.precision(preds, truthes))
        self.log('Dev[Rec]', self.recall(preds, truthes))
        return loss

    def forward(self, matrix: torch.Tensor, lengths: torch.Tensor, softmax: bool = False):
        lengths = lengths.cpu()
        matrix = self._emb(matrix)
        matrix = pack_padded_sequence(matrix, lengths, batch_first=True, enforce_sorted=False)
        matrix, z = self._lstm(matrix)
        matrix, _ = pad_packed_sequence(matrix, batch_first=True)
        # Retrieve EOS of each line
        first = matrix[:, 0, :]
        # Retrieve BOS of each line
        last = matrix[range(matrix.shape[0]), lengths - 1, :]
        # Concat EOS + BOS Encoding
        matrix = torch.cat([first, last], dim=-1)
        matrix = self._lin(matrix)

        if softmax:
            return F.softmax(matrix, dim=-1)
        return matrix
