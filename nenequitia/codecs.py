import random

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
        features: List[str],
        ys: List[str],
        langs: Optional[List[str]] = None,
        inject_special: bool = True,
        use_langs: bool = False
    ):
        self.features: Tuple[str, ...] = tuple(features)
        self.ys: Tuple[str, ...] = tuple(ys)
        self.use_langs: bool = use_langs
        if langs:
            self.use_langs = True
            self.langs = langs
            self.features = [*[f"<{lang}>" for lang in langs], *self.features]

        if inject_special:
            self.features = ("[PAD]", "[UNK]", "[BOS]", "[EOS]", *self.features)
        self.pad = 0
        self._unk = 1
        self._bos = 2
        self._eos = 3
        self._random_unk: Optional[int] = None

    def set_random_unk(self, ratio: int):
        assert 0 <= ratio <= 99, "Ratio should be an int between 0 and 99 included"
        self._random_unk = ratio

    @property
    def shape(self):
        return len(self.features), len(self.ys)

    @classmethod
    def from_text(cls, texts: List[str], ys: List[str]) -> "LabelEncoder":
        return cls(
            features=["[PAD]", "[UNK]", "[BOS]", "[EOS]"] + sorted(list(set("".join(texts)))),
            ys=ys,
            inject_special=False
        )

    @classmethod
    def from_dataframe(cls, df: DataFrame, use_lang: bool = True) -> "LabelEncoder":
        if "transcription" not in df.columns.tolist():
            raise ValueError("Dataframe requires a `transcription` column containing the text of each line")
        if use_lang and "lang" not in df.columns.tolist():
            raise ValueError("`use_lang` required the dataframe to have a `lang` column")
        if "bin" not in df.columns.tolist():
            raise ValueError("Dataframe requires a `bin` column containing the class of each line")
        return cls(
            features=sorted(list(set("".join(df.transcription.tolist())))),
            ys=df.bin.unique().tolist(),
            langs=sorted(df.lang.unique().tolist()) if use_lang else None,
            inject_special=True
        )

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls(
                json.load(f),
                inject_special=False
            )

    def encode_string(self, string: str, lang: Optional[str] = None) -> torch.Tensor:
        def map_string(local_string):
            if self._random_unk:
                local_string = [
                    char if random.randint(0, 100) > self._random_unk else "[UNK]"
                    for char in local_string
                ]
            return list([
                self.features.index(c) if c in self.features else self._unk
                for c in local_string
            ])
        return torch.tensor(
            [
                self._bos,
                *(
                    ([] if not self.use_langs else [self.features.index(f"<{lang}>")]) +
                    map_string(string)
                ),
                self._eos
            ]
        )

    def encode_y(self, bin: str) -> int:
        return self.ys.index(bin)

    def collate_pred(self, strings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = [string.shape[0] for string in strings]
        return (
           pad_sequence(strings, batch_first=True, padding_value=self.pad),
           torch.tensor(lengths)
        )

    def collate_gt(self, gt: List[Tuple[torch.Tensor, int]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        strings, ys = zip(*gt)
        lengths = [string.shape[0] for string in strings]
        return (
            (
               pad_sequence(strings, batch_first=True, padding_value=self.pad),
               torch.tensor(lengths)
            ),
            torch.tensor(ys)
        )

    def to_json(self, path: Union[str, IO]):
        if isinstance(path, str):
            f = open(path)
        else:
            f = path
        json.dump(self.features, f)
        f.close()

    def to_hparams(self):
        return {
            "features": self.features,
            "use_lang": self.use_langs,
            "ys": self.ys
        }

    @classmethod
    def from_hparams(cls, hparams: Dict):
        return cls(
            hparams["features"],
            use_langs=hparams["use_lang"],
            ys=hparams["ys"],
            inject_special=False
        )


class BaselineEncoder:
    def __init__(
        self,
        features: List[str],
        ys: List[str],
        ngrams: Tuple[float, ...] = (3, )
    ):
        self.features: Tuple[str, ...] = tuple(features)
        self.ys: Tuple[str, ...] = tuple(ys)
        self.ngrams: Tuple[float, ...] = ngrams

    @property
    def shape(self):
        return len(self.features), len(self.ys)

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> "BaselineEncoder":
        if "transcription" not in df.columns.tolist():
            raise ValueError("Dataframe requires a `transcription` column containing the text of each line")
        if "bin" not in df.columns.tolist():
            raise ValueError("Dataframe requires a `bin` column containing the class of each line")
        feats = [col[1:] for col in df.columns if col.startswith("$")]
        return cls(
            features=feats,
            ys=df.bin.unique().tolist(),
            ngrams=tuple(sorted(list(set([len(col) for col in feats]))))
        )

    def to_hparams(self):
        return {
            "features": self.features,
            "ys": self.ys,
            "ngrams": self.ngrams
        }

    @classmethod
    def from_hparams(cls, hparams: Dict):
        return cls(
            hparams["features"],
            ys=hparams["ys"],
            ngrams=hparams["ngrams"]
        )

    def encode_string(self, string: str) -> List[float]:
        string = string.replace(" ", "_")
        return [
            string.count(feature) / len(feature)
            for feature in self.features
        ]

    def encode_y(self, bin: str) -> int:
        return self.ys.index(bin)

    def collate_gt(self, gt: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        strings, ys = zip(*gt)
        return torch.tensor(strings), torch.tensor(ys)


if __name__ == "__main__":

    from pandas import read_hdf
    features = BaselineEncoder.from_dataframe(read_hdf("../features.hdf5", key="df"))
    print(features.features)

