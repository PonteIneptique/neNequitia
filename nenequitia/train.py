from typing import Optional, Dict
from pandas import DataFrame, read_hdf
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl


from nenequitia.models import LstmModule, AttentionalModule
from nenequitia.codecs import LabelEncoder
from nenequitia.datasets import DataFrameDataset
from nenequitia.contrib import get_manuscripts_and_lang_kfolds


def train_from_hdf5_dataframe(
    train: DataFrame,
    dev: DataFrame,
    test: Optional[DataFrame] = None,
    batch_size: int = 256,
    lr=1e-4,
    hparams: Optional[Dict] = None
):
    encoder = LabelEncoder.from_dataframe(train)
    if not hparams:
        hparams = {}

    # data
    train_loader = DataLoader(
        DataFrameDataset(train, encoder), batch_size=batch_size, collate_fn=encoder.pad_gt,
        num_workers=4,
        shuffle=True
    )
    val_loader = DataLoader(
        DataFrameDataset(dev, encoder), batch_size=batch_size, collate_fn=encoder.pad_gt,
        num_workers=4
    )

    # model
    model = AttentionalModule(encoder=encoder, training=True, lr=lr, **hparams)

    # training
    checkpoint_callback = ModelCheckpoint(
        monitor="Dev[Rec]",
        filename="sample-{epoch:02d}",
        save_top_k=3,
        mode="max",
        verbose=True
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="Dev[Rec]", mode="max", min_delta=5e-3, patience=5, verbose=True)
        ],
        max_epochs=300
    )
    model.save_hyperparameters()
    trainer.fit(model, train_loader, val_loader)
    if test:
        test_loader = DataLoader(
            DataFrameDataset(test, encoder), batch_size=batch_size, collate_fn=encoder.pad_gt,
            num_workers=4
        )

    return trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    df = read_hdf("texts.hdf5", key="df", index_col=0)

    df["bin"] = ""
    df.loc[df.CER < 10, "bin"] = "Good"
    df.loc[df.CER.between(10, 20, inclusive="left"), "bin"] = "Acceptable"
    df.loc[df.CER.between(20, 50, inclusive="left"), "bin"] = "Bad"
    df.loc[df.CER >= 50, "bin"] = "Very bad"

    for (lr, dropout) in [(1e-3, .1), (1e-3, .2), (1e-4, .1)]:
        for i in range(1):
            train, dev, test = get_manuscripts_and_lang_kfolds(
                df,
                k=i, per_k=2,
                force_test=["SBB_PK_Hdschr25"]
            )
            model = train_from_hdf5_dataframe(train, dev, test=test, lr=lr, hparams={
                "dropout": dropout
            })
