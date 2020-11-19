# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : create_db_from_files.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #
import numpy as np
import pytorch_lightning as pl
import torch
from ase.data import atomic_numbers
from lightMolNet.data.dataloader import _collate_aseatoms
from lightMolNet.data.partitioning import random_split_partial
from lightMolNet.datasets.g16datadb import G16datadb
from torch.utils.data import DataLoader

refat = {"C": {"U0": -37.843662}}

atomrefarray = np.zeros([100, 1])

for key, v in refat.items():
    atomrefarray[atomic_numbers[key]] = refat[key]["U0"]


# from C20-C84, B3LYP/6-31G GD3


class G16dataset(pl.LightningDataModule):
    def __init__(
            self,
            dbpath="fullerene.db",
            logfiledir="logdata",
            refatom=None,
            proceed=True,
            batch_size=10
    ):
        if refatom is None:
            refatom = refat
        self.batch_size = batch_size
        self.dbpath = dbpath
        self.logfiledir = logfiledir
        self.refatom = refatom
        self.proceed = proceed
        super().__init__()

    def prepare_data(self, stage=None):
        self.dataset = G16datadb(dbpath=self.dbpath,
                                 logfiledir=self.logfiledir,
                                 refatom=self.refatom,
                                 proceed=self.proceed
                                 )

    def setup(self, stage=None):
        self.train, self.val, self.test = \
            random_split_partial(self.dataset, [60, 20, 20])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=_collate_aseatoms, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=_collate_aseatoms)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=_collate_aseatoms)


from lightMolNet.Struct.Atomistic.Atomwise import Atomwise
from lightMolNet.Struct.nn import LitMolNet

import torch.nn.functional as F


class litNet(pl.LightningModule):

    def __init__(self, learning_rate=1e-4, datamodule=None, batch_size=None):
        super(litNet, self).__init__()
        self.represent = LitMolNet(
            n_atom_embeddings=128,
            n_filters=128,
            n_interactions=3,
            cutoff=5.0,
            n_gaussians=25,
            max_Z=18
        )
        self.outputU0 = Atomwise(
            n_in=128,
            n_out=1,
            atomref=atomrefarray,
            property="energy_U0"
        )
        self.datamodule = datamodule
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=25,
            factor=0.8,
            min_lr=0,
            eps=1e-15
        )

    def forward(self, inputs):
        inputs["representation"] = self.represent(inputs)
        outs = {}
        outs.update(self.outputU0(inputs))
        return outs

    def training_step(self, batch, batch_idx):
        self.log("lr", self.optimizer.state_dict()['param_groups'][0]['lr'], on_epoch=True)
        batch["representation"] = self.represent(batch)
        outs = {}
        outs.update(self.outputU0(batch))
        loss = F.mse_loss(outs["energy_U0"], batch["energy_U0"])
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch["representation"] = self.represent(batch)
        outs = {}
        outs.update(self.outputU0(batch))
        loss = F.mse_loss(outs["energy_U0"], batch["energy_U0"])
        # Logging to TensorBoard by default
        self.log('val_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        batch["representation"] = self.represent(batch)
        outs = {}
        outs.update(self.outputU0(batch))
        loss = F.mse_loss(outs["energy_U0"], batch["energy_U0"])
        # Logging to TensorBoard by default
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return {"optimizer": self.optimizer,
                'lr_scheduler': self.scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
                }


def cli_main():
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='FullNet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        save_last=True
    )

    dataset = G16dataset(batch_size=64)
    # dataset.prepare_data()
    # dataset.setup()
    # model = litNet()
    model = litNet(learning_rate=0.001, datamodule=dataset, batch_size=10)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=1,
        # auto_lr_find=True,
        # benchmark=True,
        max_epochs=10000,
        # auto_scale_batch_size='binsearch'
    )
    model.load_from_checkpoint("lightning_logs/version_1/checkpoints/FullNet-epoch=1192-val_loss=25273.80.ckpt")
    # model.freeze()
    trainer.fit(model)
    # lr_finder = trainer.tuner.lr_find(model,min_lr=1e-7,max_lr=1e-2)
    # fig = lr_finder.plot(suggest=True,show=True)
    # print(lr_finder.suggestion())

    # result = trainer.test(model,verbose=True)
    # print(result)


if __name__ == '__main__':
    cli_main()
