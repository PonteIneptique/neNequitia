import torch
import torch.cuda
import torch.nn
import json
from torch.nn.utils.rnn import pad_sequence
from pandas import DataFrame
from typing import Tuple, Dict, List, Optional, Sequence, Union, IO


class LabelEncoder:
    def __init__(
        self,
        features: Optional[List[str]],
        inject_special: bool = True
    ):
        self.features = features or ()
        if self.features:
            self.features: Tuple[str, ...] = tuple(features)
        if inject_special:
            self.features = ("[PAD]", "[UNK]", "[BOS]", "[EOS]", *self.features)
        self._pad = 0
        self._unk = 1
        self._bos = 2
        self._eos = 3

    def __len__(self):
        return len(self.features)

    @classmethod
    def from_text(cls, texts: List[str]) -> "LabelEncoder":
        return cls(
            features=["[PAD]", "[UNK]", "[BOS]", "[EOS]"] + sorted(list(set("".join(texts)))),
            inject_special=False
        )

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> "LabelEncoder":
        if "transcription" not in df.columns.tolist():
            raise ValueError("Dataframe requires a `transcription` column containing the text of each line")
        return cls(
            features=["[PAD]", "[UNK]", "[BOS]", "[EOS]"] + sorted(list(set("".join(df.transcription.tolist())))),
            inject_special=False
        )

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls(
                json.load(f),
                inject_special=False
            )

    def encode_string(self, string: str) -> torch.Tensor:
        return torch.tensor(
            [self._bos, *list([self.features.index(c) if c in self.features else 0 for c in string]), self._eos]
        )

    def pad_gt(self, gt: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        strings, ys = zip(*gt)
        lengths = [string.shape[0] for string in strings]
        return (pad_sequence(strings, batch_first=True, padding_value=self._pad), torch.tensor(lengths)),\
               torch.tensor(ys)

    def to_json(self, path: Union[str, IO]):
        if isinstance(path, str):
            f = open(path)
        else:
            f = path
        json.dump(self.features, f)
        f.close()


if __name__ == "__main__":
    encoder = LabelEncoder(list("Helo"))
    assert (encoder.encode_string("Hello").tolist() == [2, 4, 5, 6, 6, 7, 3]), \
        "Encoding should be correct"

    batch = ["Helloo", "Hello"]
    assert (encoder.pad_batch(batch)[1].tolist() == [[2, 4, 5, 6, 6, 7, 7, 3], [2, 4, 5, 6, 6, 7, 3, 0]]), \
        "Padding should be correct."
    assert (encoder.pad_batch(batch)[0].tolist() == [8, 7]), \
        "Lengths should be correct."
