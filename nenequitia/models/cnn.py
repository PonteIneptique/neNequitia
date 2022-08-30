from typing import Optional, Tuple

import torchmetrics
import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nenequitia.models.base import BaseModule
from nenequitia.codecs import LabelEncoder


class TextCnnModule(BaseModule):
    # https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/TextCNN.ipynb#scrollTo=rjZMYxFoznj9
    def __init__(self, encoder: LabelEncoder,
                 emb_size: int = 200,
                 ngrams: Tuple[int, ...] = (3, 4, 5, 6),
                 ngram_proj: int = 64,
                 dropout: float = .1,
                 lr: float = 5e-3,
                 training: bool = False):
        super(TextCnnModule, self).__init__(
            encoder=encoder, lr=lr, training=training
        )

        self.hparams["emb_size"]: int = emb_size
        self.hparams["dropout"]: float = dropout
        self.hparams["ngrams"]: int = ngrams  # Kernel Size
        self.hparams["ngram_proj"]: int = ngram_proj  # Num Filters
        # self.hparams["attention_size"]: bool = attention_size

        self._emb = nn.Embedding(
            self.inp,
            self.hparams["emb_size"]
        )
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=ngram_proj,
                    kernel_size=(ngram_size, emb_size),
                    stride=(1, )
                ),
                nn.ReLU()
            )
            for ngram_size in ngrams
        ])

        self._lin = nn.Sequential(
            nn.Dropout(self.hparams["dropout"]),
            nn.Linear(len(ngrams) * ngram_proj, self.out)
        )

    def forward(self, matrix: torch.Tensor, lengths: torch.Tensor):
        # https://github.com/Doragd/Text-Classification-PyTorch/blob/master/models/TextCNN.py
        matrix = self._emb(matrix).unsqueeze(1)
        matrix = [conv(matrix).squeeze(3) for conv in self._convs]
        matrix = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in matrix]
        matrix = torch.cat(matrix, dim=1)
        matrix = self._lin(matrix)
        return F.softmax(matrix, dim=-1)


class CustomTextRCnnModule(BaseModule):
    # https://github.com/QimingPeng/Text-Classification/blob/master/model/TextRCNN.py
    def __init__(self, encoder: LabelEncoder,
                 emb_size: int = 100,
                 ngrams: Tuple[int, ...] = (2, 3, 4, 5, 6),
                 hid_size: int = 128,
                 ngram_proj: int = 100,
                 dropout: float = .1,
                 lr: float = 5e-3,
                 training: bool = False):
        super(CustomTextRCnnModule, self).__init__(
            encoder=encoder, lr=lr, training=training
        )

        self.hparams["emb_size"]: int = emb_size
        self.hparams["dropout"]: float = dropout
        self.hparams["ngrams"]: int = ngrams  # Kernel Size
        self.hparams["ngram_proj"]: int = ngram_proj  # Num Filters
        self.hparams["hid_size"]: int = hid_size
        # self.hparams["attention_size"]: bool = attention_size

        self._emb = nn.Embedding(
            self.inp,
            self.hparams["emb_size"]
        )
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=ngram_proj,
                    kernel_size=(ngram_size, self.hparams["hid_size"]*2),  # BiDirectional
                    stride=(1, )
                ),
                nn.ReLU()
            )
            for ngram_size in ngrams
        ])

        self._rnn = nn.LSTM(
            input_size=self.hparams["emb_size"], hidden_size=self.hparams["hid_size"],
            bidirectional=True, batch_first=True)

        self._lin = nn.Sequential(
            nn.Dropout(self.hparams["dropout"]),
            nn.Linear(len(ngrams) * ngram_proj, self.out)
        )

    def forward(self, matrix: torch.Tensor, lengths: torch.Tensor):
        # https://github.com/Doragd/Text-Classification-PyTorch/blob/master/models/TextCNN.py
        matrix = self._emb(matrix)

        # Apply RNN
        matrix = pack_padded_sequence(matrix, lengths.cpu(), batch_first=True, enforce_sorted=False)
        matrix, z = self._rnn(matrix)
        matrix, _ = pad_packed_sequence(matrix, batch_first=True)
        matrix = matrix.unsqueeze(1)

        # Apply CNN
        matrix = [conv(matrix).squeeze(3) for conv in self._convs]
        matrix = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in matrix]
        matrix = torch.cat(matrix, dim=1)
        matrix = self._lin(matrix)
        return F.softmax(matrix, dim=-1)