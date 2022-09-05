from torch.utils.data import Dataset
from nenequitia.codecs import LabelEncoder, BaselineEncoder
from pandas import DataFrame
from typing import Union


__all__ = ["DataFrameDataset"]


class DataFrameDataset(Dataset):
    def __init__(self, df: DataFrame, encoder: LabelEncoder, is_gt: bool = True):
        self.df: DataFrame = df
        self.encoder: LabelEncoder = encoder
        self._is_gt: bool = is_gt

        if self.encoder.use_langs:
            self._transcriptions = [
                self.encoder.encode_string(string, lang=lang)
                for string, lang in zip(df.transcription, df.lang)
            ]
        else:
            self._transcriptions = [
                self.encoder.encode_string(string)
                for string in df.transcription
            ]
        if self._is_gt:
            self._bins = [self.encoder.encode_y(bin) for bin in self.df.bin.tolist()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not self._is_gt:
            return self._transcriptions[idx]
        return self._transcriptions[idx], self._bins[idx]


class BaselineDataFrameDataset(Dataset):
    def __init__(self, df: DataFrame, encoder: Union[LabelEncoder, BaselineEncoder], is_gt: bool = True):
        self.df: DataFrame = df
        self.encoder: LabelEncoder = encoder
        self._is_gt: bool = is_gt

        self._transcriptions = self.df.transcription.tolist()
        if self._is_gt:
            self._bins = [self.encoder.encode_y(bin) for bin in self.df.bin.tolist()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not self._is_gt:
            return self._transcriptions[idx]
        return self.encoder.encode_string(self._transcriptions[idx]), self._bins[idx]
