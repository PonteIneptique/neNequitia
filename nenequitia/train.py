from typing import Optional, Dict
from pandas import DataFrame, read_hdf
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers.csv_logs import CSVLogger
import pytorch_lightning as pl


from nenequitia.models import RnnModule, AttentionalModule, TextCnnModule, CustomTextRCnnModule
from nenequitia.codecs import LabelEncoder
from nenequitia.datasets import DataFrameDataset
from nenequitia.contrib import get_manuscripts_and_lang_kfolds


def train_from_hdf5_dataframe(
    train: DataFrame,
    dev: DataFrame,
    test: Optional[DataFrame] = None,
    module = AttentionalModule,
    batch_size: int = 256,
    lr = 1e-4,
    logger_kwargs: Optional[Dict] = None,
    hparams: Optional[Dict] = None
):
    encoder = LabelEncoder.from_dataframe(train, use_lang=False)
    if not hparams:
        hparams = {}

    encoder.set_random_unk(2)

    # data
    train_loader = DataLoader(
        DataFrameDataset(train, encoder), batch_size=batch_size, collate_fn=encoder.collate_gt,
        num_workers=4,
        shuffle=True
    )
    encoder.set_random_unk(0) # Disable for dev
    val_loader = DataLoader(
        DataFrameDataset(dev, encoder), batch_size=batch_size, collate_fn=encoder.collate_gt,
        num_workers=4
    )

    # model
    model = module(encoder=encoder, training=True, lr=lr, **hparams)

    # training
    checkpoint_callback = ModelCheckpoint(
        monitor="Dev[Rec]",
        filename="sample-{epoch:02d}",
        save_top_k=1,
        mode="max",
        verbose=True
    )
    logger = CSVLogger(**logger_kwargs)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(monitor="Dev[Rec]", mode="max", min_delta=2e-3, patience=10, verbose=True)
        ],
        max_epochs=100,
        logger=logger
    )
    model.save_hyperparameters()
    trainer.fit(model, train_loader, val_loader)
    if test is not None:
        test_loader = DataLoader(
            DataFrameDataset(test, encoder), batch_size=batch_size, collate_fn=encoder.collate_gt,
            num_workers=4
        )
        return model, trainer.test(dataloaders=test_loader)
    return model


if __name__ == "__main__":
    df = read_hdf("texts.hdf5", key="df", index_col=0)

    df["bin"] = ""
    df.loc[df.CER < 10, "bin"] = "Good"
    df.loc[df.CER.between(10, 25, inclusive="left"), "bin"] = "Acceptable"
    df.loc[df.CER.between(25, 50, inclusive="left"), "bin"] = "Bad"
    df.loc[df.CER >= 50, "bin"] = "Very bad"

    for (lr, dropout) in [(5e-4, .1)]:
        for module, kwargs in [
            (RnnModule, {"dropout": dropout, "emb_size": 200, "hid_size": 256}),
            (AttentionalModule, {"dropout": dropout, "emb_size": 200, "hid_size": 256, "cell": "LSTM"}),
            (TextCnnModule, {"dropout": dropout, "emb_size": 100, "ngrams": (2, 3, 4, 5, 6)}),
            #(CustomTextRCnnModule, {"dropout": dropout, "emb_size": 100, "ngrams": (2, 3, 4, 5, 6)})
        ]:
            for i in range(5):
                train, dev, test = get_manuscripts_and_lang_kfolds(
                    df,
                    k=i, per_k=2,
                    force_test=["SBB_PK_Hdschr25"]
                )
                print("Train dataset description")
                print(train.groupby("bin").size())
                print("Dev dataset description")
                print(dev.groupby("bin").size())
                print("Test dataset description")
                print(test.groupby("bin").size())
                model = train_from_hdf5_dataframe(
                    train, dev, test=test,
                    module=module, lr=lr, hparams=kwargs,
                    batch_size=128,
                    logger_kwargs=dict(
                        save_dir="explogs-nolang",
                        name=f"{module.__name__}",
                        version=str(i)
                    )
                )
