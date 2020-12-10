# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : FullerlogTrain.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import pytorch_lightning as pl
import torch
from ase.units import Hartree

from lightMolNet import Properties
from lightMolNet.Struct.Atomistic.Atomwise import Atomwise
from lightMolNet.Struct.nn import SchNet
from lightMolNet.data.atomsref import get_refatoms
from lightMolNet.datasets.LitDataSet.G16DataSet import G16DataSet
from lightMolNet.net import LitNet

Batch_Size = 32
USE_GPU = 1

refat_b3lypgd3 = {Properties.UNIT: {Properties.energy_U0: Hartree},
                  # "H": {Properties.energy_U0: -0.500273},
                  "C": {Properties.energy_U0: -37.843662},
                  # "N": {Properties.energy_U0: -54.583861},
                  # "O": {Properties.energy_U0: -75.064579},
                  # "F": {Properties.energy_U0: -99.718730}
                  }

atomrefs = get_refatoms(refat_b3lypgd3, Properties.energy_U0)


def cli_main():
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_MAE',
        filename='FullNet-{epoch:02d}-{val_loss:.4f}',
        save_top_k=2,
        save_last=True
    )

    dataset = G16DataSet(dbpath="fullerxtb.db",
                         logfiledir=r"D:\CODE\PycharmProjects\lightMolNet\examples\logdata",
                         atomref=atomrefs,
                         batch_size=Batch_Size,
                         pin_memory=True)
    dataset.prepare_data()
    dataset.setup(data_partial=None)
    scheduler = {"_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
                 "patience": 50,
                 "factor": 0.8,
                 "min_lr": 1e-7,
                 "eps": 1e-7
                 }
    model = LitNet(learning_rate=1e-4,
                   datamodule=dataset,
                   representNet=[SchNet],
                   outputNet=[Atomwise],
                   outputPro=[Properties.energy_U0],
                   batch_size=Batch_Size,
                   scheduler=scheduler)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=USE_GPU,
        auto_lr_find=True,
        benchmark=True,
        max_epochs=10000,
        # auto_scale_batch_size='binsearch'
    )

    ### train
    # trainer.fit(model)

    ### scale_batch
    # trainer.tune(model)

    ### lr_finder
    lr_finder = trainer.tuner.lr_find(model, min_lr=1e-7, max_lr=0.5e-2)
    fig = lr_finder.plot(suggest=True, show=True)
    print(lr_finder.suggestion())

    # result = trainer.test(model,verbose=True)
    # print(result)


if __name__ == '__main__':
    cli_main()
