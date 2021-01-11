# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : FullxtbTrain.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import pytorch_lightning as pl
import torch
from ase.units import eV

from lightMolNet import Properties
from lightMolNet.Struct.Atomistic.atomwise import Atomwise
from lightMolNet.Struct.nn import SchNet
from lightMolNet.data.atomsref import get_refatoms
from lightMolNet.datasets.LitDataSet.xtbxyzdataset import XtbXyzDataSet
from lightMolNet.logger import DebugLogger
from lightMolNet.net import LitNet

logger = DebugLogger(__name__)

Batch_Size = 32
USE_GPU = 1

refat_xTB = {Properties.UNIT: {Properties.energy_U0: eV},
             # "H": {Properties.energy_U0: -10.707211383396714},
             "C": {Properties.energy_U0: -48.847445262804705},
             # "N": {Properties.energy_U0: -71.00681805517411},
             # "O": {Properties.energy_U0: -102.57117256025786},
             # "F": {Properties.energy_U0: -125.69864294466228}
             }

atomrefs = get_refatoms(refat_xTB, Properties.energy_U0, z_max=18)


def cli_main(ckpt_path=None, schnetold=False):
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_0_loss_MAE',
        filename='FullNet-{epoch:02d}-{val_loss:.4f}',
        save_top_k=2,
        save_last=True
    )
    statistics = False
    if statistics is False:
        logger.warning("Use explicit statistic values.")
    dataset = XtbXyzDataSet(dbpath="fullerxtbdata20to88.db",
                            xyzfiledir=r"D:\CODE\#DATASETS\FullDB\xTBBack",
                            atomref=atomrefs,
                            batch_size=Batch_Size,
                            pin_memory=True,
                            # proceed=True,
                            statistics=statistics,
                            )
    dataset.prepare_data()
    dataset.setup(data_partial=[10, 1, 99989])
    scheduler = {"_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                 "patience": 50,
                 "factor": 0.8,
                 "min_lr": 1e-7,
                 "eps": 1e-8,
                 "cooldown": 25
                 }
    model = LitNet(learning_rate=1e-5,
                   datamodule=dataset,
                   representNet=[SchNet],
                   outputNet=[Atomwise],
                   outputPro=[Properties.energy_U0],
                   batch_size=Batch_Size,
                   scheduler=scheduler
                   # means=0,
                   # stddevs=1,
                   )
    if ckpt_path is not None:
        from collections import OrderedDict
        state_dict = torch.load(ckpt_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if "represent" in k:
                if "embedding" in k:
                    name = ".".join(["represent.0.embeddings", *k.split(".")[2:]])
                else:
                    name = ".".join(["represent.0", *k.split(".")[1:]])
                if k == "representation.embedding.weight":
                    v = v[:18, :]
            elif "outputU0" in k:
                if 'outputU0.atomref.weight' in k:
                    name = "output.0.atomref.weight"
                    v = torch.Tensor(atomrefs["energy_U0"])
                elif 'standardize.mean' in k:
                    name = "output.0.standardize.mean"
                    v = v * 0
                elif 'standardize.stddev' in k:
                    name = "output.0.standardize.stddev"
                    v = v / v
                elif 'out_net' in k:
                    name = ".".join(["output.0.out_net", *k.split(".")[2:]])

            new_state_dict[name] = v
        if schnetold:
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict["state_dict"])

        # model.freeze()
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=USE_GPU,
        auto_lr_find=True,
        benchmark=True,
        max_epochs=10000,
        # auto_scale_batch_size='binsearch'
    )

    ### train
    trainer.fit(model)

    ### scale_batch
    # trainer.tune(model)

    ### lr_finder
    # lr_finder = trainer.tuner.lr_find(model, min_lr=1e-6, max_lr=0.5e-2)
    # fig = lr_finder.plot(suggest=True, show=True)
    # print(lr_finder.suggestion())

    # result = trainer.test(model, verbose=True)
    # print(result)


if __name__ == '__main__':
    ckpt_path = r"E:\#Projects\#Research\0109-xtbfuller-SchNet-baseline-1\output\checkpoints\FullNet-epoch=22-val_loss=0.0000.ckpt"
    cli_main(ckpt_path, schnetold=False)
