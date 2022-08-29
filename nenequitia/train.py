from nenequitia.models import LstmModule
from nenequitia.codecs import LabelEncoder
from typing import Optional
from pandas import DataFrame, read_hdf
from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl


class DataFrameDataset(Dataset):
    def __init__(self, df: DataFrame, encoder: LabelEncoder):
        self.df: DataFrame = df
        self.encoder: LabelEncoder = encoder

        self._transcriptions = [
            self.encoder.encode_string(string)
            for string in df.transcription
        ]
        self._bins = self.df.bin.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self._transcriptions[idx], self._bins[idx]


def train_from_hdf5_dataframe(
    train: DataFrame, dev: DataFrame, test: Optional[DataFrame] = None,
    batch_size: int = 256
):
    encoder = LabelEncoder.from_dataframe(train)

    # data
    train_loader = DataLoader(DataFrameDataset(train, encoder), batch_size=batch_size, collate_fn=encoder.pad_gt)
    val_loader = DataLoader(DataFrameDataset(dev, encoder), batch_size=batch_size, collate_fn=encoder.pad_gt)


    # model
    model = LstmModule(encoder=encoder, bins=len(train.bin.unique()), training=True)

    # training
    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    df = read_hdf("../texts.hdf5", key="df", index_col=0)
    train, dev, test = df.iloc[:1024, :], df.iloc[1024:2048, :], df.iloc[2048:3072, :]
    print(LabelEncoder.from_dataframe(train).encode_string("Hello"))
    train_from_hdf5_dataframe(train, dev)