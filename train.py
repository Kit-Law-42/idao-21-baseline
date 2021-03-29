import configparser
import pathlib as path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from idao.data_module import IDAODataModule
from idao.model import ResNetModel, SimpleConv

seed_everything(666)


# def trainer(mode: ["classification", "regression"], cfg):
def trainer(mode, cfg):
    # init model
    #model = SimpleConv(mode=mode)
    model = ResNetModel(mode=mode)
    if mode == "classification":
        epochs = cfg["TRAINING"]["ClassificationEpochs"]
    else:
        epochs = cfg["TRAINING"]["RegressionEpochs"]
    # Initialize a trainer
    # Require at least 1 GPU for training.
    trainer = pl.Trainer(
        gpus=int(cfg["TRAINING"]["NumGPUs"]),
        tpus=int(cfg["TRAINING"]["NumTPUs"]),
        max_epochs=int(epochs),
        progress_bar_refresh_rate=20,
        weights_save_path=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]).joinpath(
            mode
        ),
        default_root_dir=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]),
        weights_summary='full'
    )

    # Train the model ⚡
    # AttributeError: Can't pickle local object 'IDAODataModule.prepare_data.<locals>.<lambda>'
    # Cannot run on windows
    # https://discuss.pytorch.org/t/cant-pickle-local-object-dataloader-init-locals-lambda/31857/13
    trainer.fit(model, dataset_dm)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    #config.read("./config.ini")
    config.read("drive/MyDrive/Colab Notebooks/Olympiad2021/config.ini")

    PATH = path.Path(config["DATA"]["DatasetPath"])

    # see data_module.py
    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=int(config["TRAINING"]["BatchSize"]), cfg=config
    )
    dataset_dm.prepare_data()
    dataset_dm.setup()

    for mode in ["classification", "regression"]:
        trainer(mode, cfg=config)
