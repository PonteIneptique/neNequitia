from typing import Optional

import torchmetrics
import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nenequitia.models.base import BaseModule
from nenequitia.codecs import LabelEncoder


class RnnModule(BaseModule):
    def __init__(self, encoder: LabelEncoder,
                 cell: str = "LSTM",
                 emb_size: int = 128,
                 hid_size: int = 128,
                 dropout: float = .1,
                 use_highway: bool = True,
                 lr: float = 5e-3,
                 training: bool = False):
        super(RnnModule, self).__init__(
            encoder=encoder, lr=lr, training=training
        )

        self.hparams["dropout"]: float = dropout
        self.hparams["emb_size"]: int = emb_size
        self.hparams["hid_size"]: int = hid_size
        self.hparams["use_highway"]: bool = use_highway

        self._emb = nn.Sequential(
            nn.Embedding(
                self.inp,
                self.hparams["emb_size"]
            ),
            nn.Dropout(self.hparams["dropout"])
        )
        if cell == "GRU":
            self._rnn = nn.GRU(
                input_size=self.hparams["emb_size"], hidden_size=self.hparams["hid_size"],
                bidirectional=True, batch_first=True)
        else:
            self._rnn = nn.LSTM(
                input_size=self.hparams["emb_size"], hidden_size=self.hparams["hid_size"],
                bidirectional=True, batch_first=True)

        if use_highway:
            self._lin = nn.Sequential(
                nn.Dropout(self.hparams["dropout"]),
                # HID_SIZE*4 because CONCAT first and last
                nn.Linear(self.hparams["hid_size"] * 4, self.hparams["hid_size"] * 2),
                nn.Linear(self.hparams["hid_size"] * 2, self.out)
            )
        else:
            self._lin = nn.Sequential(
                nn.Dropout(self.hparams["dropout"]),
                nn.Linear(self.hparams["hid_size"] * 4, self.out)
            )

    def forward(self, matrix: torch.Tensor, lengths: torch.Tensor, softmax: bool = False):
        lengths = lengths.cpu()
        matrix = self._emb(matrix)
        matrix = pack_padded_sequence(matrix, lengths, batch_first=True, enforce_sorted=False)
        matrix, z = self._rnn(matrix)
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
