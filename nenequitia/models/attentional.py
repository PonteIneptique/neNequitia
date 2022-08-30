import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nenequitia.codecs import LabelEncoder
from nenequitia.models.base import BaseModule


class AttentionalModule(BaseModule):
    # https://github.com/lascivaroma/seligator/blob/main/seligator/modules/seq2vec/han.py
    def __init__(self, encoder: LabelEncoder, bins: int,
                 emb_size: int = 100,
                 hid_size: int = 256,
                 dropout: float = .1,
                 lr: float = 5e-3,
                 training: bool = False):
        super(AttentionalModule, self).__init__(
            encoder=encoder, bins=bins, lr=lr, training=training
        )
        self._dropout: float = dropout
        self._emb_size: int = emb_size
        self._hid_size: int = hid_size

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

        if training:
            self._context.data.normal_(.0, .05)

    def get_preds_from_forward(self, predictions) -> torch.Tensor:
        return predictions[0]

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
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
        # When it's not pad, then it's accepted in the mask
        mask = (inputs != self.encoder.pad).unsqueeze(dim=-1)

        # weights : (batch_size, sentence_len, 1)
        weights = torch.where(
            mask == 1,
            weights,
            torch.full_like(mask, 0, dtype=torch.float, device=weights.device)
        )

        # weights : (batch_size, sentence_len, 1)
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)

        # Apply attention
        output = torch.sum((weights * char_outputs), dim=1)

        # Normalize weight shape
        weights = weights.squeeze(2)

        matrix = self._lin(output)
        #matrix = F.softmax(matrix, dim=-1)

        return matrix, weights
