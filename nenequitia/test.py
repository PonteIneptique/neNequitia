from typing import Optional
from pandas import DataFrame, read_hdf
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl


from nenequitia.models import RnnModule, AttentionalModule
from nenequitia.codecs import LabelEncoder
from nenequitia.datasets import DataFrameDataset


def test_from_hdf5_dataframe(
        train: DataFrame,
        ckpt_path: str,
        test: Optional[DataFrame] = None,
        batch_size: int = 256
):
    encoder = LabelEncoder.from_dataframe(train)

    # data
    test_loader = DataLoader(
        DataFrameDataset(test, encoder), batch_size=batch_size,
        collate_fn=encoder.pad_gt,
        num_workers=4
    )

    # model
    model = AttentionalModule(encoder=encoder, training=True)

    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)
    return model


if __name__ == "__main__":
    from nenequitia.contrib import get_manuscripts_and_lang_kfolds
    df = read_hdf("../texts.hdf5", key="df", index_col=0)

    df.loc[df.CER < 10, "bin"] = "Good"
    df.loc[df.CER.between(10, 20, inclusive="left"), "bin"] = "Acceptable"
    df.loc[df.CER.between(20, 50, inclusive="left"), "bin"] = "Bad"
    df.loc[df.CER >= 50, "bin"] = "Very bad"
    print(df.bin.unique())
    print(df.lang.unique())

    train, dev, test = get_manuscripts_and_lang_kfolds(
        df,
        k=0,
        per_k=2,
        force_test=["SBB_PK_Hdschr25"]
    )
    model = test_from_hdf5_dataframe(
        train=train,
        test=test,
        ckpt_path="/home/thibault/dev/Medieval-Model/lightning_logs/version_5/checkpoints/sample-epoch=01.ckpt"
    )


