from torch.utils.data import Dataset
from nenequitia.codecs import LabelEncoder
from pandas import DataFrame


__all__ = ["DataFrameDataset"]


class DataFrameDataset(Dataset):
    def __init__(self, df: DataFrame, encoder: LabelEncoder):
        self.df: DataFrame = df
        self.encoder: LabelEncoder = encoder

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
        self._bins = [self.encoder.encode_y(bin) for bin in self.df.bin.tolist()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self._transcriptions[idx], self._bins[idx]
