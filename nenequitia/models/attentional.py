from typing import Optional

import torchmetrics
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from nenequitia.codecs import LabelEncoder
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class AttentionalModule(pl.LightningModule):
    # https://github.com/lascivaroma/seligator/blob/main/seligator/modules/seq2vec/han.py
    def __init__(self, encoder: LabelEncoder, bins: int,
                 emb_size: int = 128,
                 hid_size: int = 128,
                 dropout: float = .1,
                 use_highway: bool = True,
                 lr: float = 5e-3,
                 training: bool = False):
        super(AttentionalModule, self).__init__()
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

        # Attention everyone !
        self._rnn = nn.GRU(self._emb_size, hidden_size=self._hid_size, bidirectional=True, batch_first=True)
        self._rnn_dropout = nn.Dropout(self._dropout)
        self._context = nn.Parameter(torch.Tensor(2 * hid_size, 1), requires_grad=True)
        self._rnn_dense = nn.Linear(2 * hid_size, 2 * hid_size)

        self._lin = nn.Sequential(
            nn.Dropout(self._dropout),
            nn.Linear(self._hid_size * 2, self.out)
        )

        self.metric_accuracy: Optional[torchmetrics.Accuracy] = None
        self.metric_precision: Optional[torchmetrics.Precision] = None
        self.metric_recall: Optional[torchmetrics.Recall] = None
        if training:
            self._context.data.normal_(.0, .05)
            self.metric_recall = torchmetrics.Recall()
            self.metric_accuracy = torchmetrics.Accuracy()
            self.metric_precision = torchmetrics.Precision()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        (lines, lengths), truthes = train_batch
        preds, weights = self(lines, lengths, softmax=False)
        loss = F.cross_entropy(preds, truthes)
        self.log('Train[Loss]', loss)
        return loss, weights

    def validation_step(self, val_batch, batch_idx):

        (lines, lengths), truthes = val_batch
        preds, weights = self(lines, lengths, softmax=False)
        # Compute Accuracy or Precision/Recall here
        loss = F.cross_entropy(preds, truthes)
        self.log('Dev[Loss]', loss)
        preds = preds.argmax(dim=-1)
        self.log('Dev[Acc]', self.metric_accuracy(preds, truthes))
        self.log('Dev[Pre]', self.metric_precision(preds, truthes))
        self.log('Dev[Rec]', self.metric_recall(preds, truthes))
        return loss, weights

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor, softmax: bool = False):
        # https://www.kaggle.com/code/kaushal2896/packed-padding-masking-with-attention-rnn-gru
        lengths = lengths.cpu()
        matrix = self._emb(inputs)
        matrix = pack_padded_sequence(matrix, lengths, batch_first=True, enforce_sorted=False)

        char_outputs, _ = self._rnn(matrix)
        char_outputs, _ = pad_packed_sequence(char_outputs, batch_first=True)
        char_outputs = self._rnn_dropout(char_outputs)

        # attention: (batch_size, sentence_len, 2*gru_size)
        word_attention = torch.tanh(self._rnn_dense(char_outputs))
        # weights: batch_size, sentence_len, 1
        weights = torch.matmul(word_attention, self._context)
        # weights : (batch_size, sentence_len, 1)
        weights = F.softmax(weights, dim=1)

        # Get masks
        mask = torch.ones(inputs.shape[:2]).bool().unsqueeze(dim=-1)

        # weights : (batch_size, sentence_len, 1)
        weights = torch.where(mask != 0, weights, torch.full_like(mask, 0, dtype=torch.float, device=weights.device))

        # weights : (batch_size, sentence_len, 1)
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)

        # Apply attention
        output = torch.sum((weights * char_outputs), dim=1)

        # Normalize weight shape
        weights = weights.squeeze(2)

        matrix = self._lin(output)

        if softmax:
            return F.softmax(matrix, dim=-1), weights
        return matrix, weights
