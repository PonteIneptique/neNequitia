import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nenequitia.codecs import LabelEncoder
from nenequitia.models.base import BaseModule


class AttentionalModule(BaseModule):
    # https://github.com/lascivaroma/seligator/blob/main/seligator/modules/seq2vec/han.py
    def __init__(self, encoder: LabelEncoder,
                 cell: str = "GRU",
                 emb_size: int = 64,
                 hid_size: int = 120,
                 dropout: float = .2,
                 lr: float = 5e-3,
                 training: bool = False):
        super(AttentionalModule, self).__init__(
            encoder=encoder, lr=lr, training=training
        )
        self.hparams["cell"]: str = cell
        self.hparams["dropout"]: float = dropout
        self.hparams["emb_size"]: int = emb_size
        self.hparams["hid_size"]: int = hid_size

        self._emb = nn.Sequential(
            nn.Embedding(self.inp, self.hparams["emb_size"]),
            nn.Dropout(self.hparams["dropout"])
        )

        # Attention everyone !
        if cell == "GRU":
            self._rnn = nn.GRU(
                input_size=self.hparams["emb_size"], hidden_size=self.hparams["hid_size"],
                bidirectional=True, batch_first=True)
        else:
            self._rnn = nn.LSTM(
                input_size=self.hparams["emb_size"], hidden_size=self.hparams["hid_size"],
                bidirectional=True, batch_first=True)
        self._rnn_dropout = nn.Dropout(self.hparams["dropout"])
        self._context = nn.Parameter(torch.Tensor(2 * hid_size, 1), requires_grad=True)
        self._rnn_dense = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)

        self._lin = nn.Sequential(
            nn.Dropout(self.hparams["dropout"]),
            nn.Linear(self.hparams["hid_size"] * 2, self.out)
        )

        if training:
            self._context.data.normal_(.0, .05)

    def get_preds_from_forward(self, predictions) -> torch.Tensor:
        return predictions[0]

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
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

        return matrix, weights
