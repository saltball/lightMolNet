# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : __init__.py.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import lightMolNet.Module.functional as F
import numpy as np
import pytorch_lightning as pl
import torch
from lightMolNet import Properties
from lightMolNet.Struct.Atomistic.Atomwise import Atomwise
from lightMolNet.Struct.nn import SchNet
from torch.nn import ModuleList


class LitNet(pl.LightningModule):

    def __init__(
            self,
            learning_rate=1e-4,
            datamodule=None,
            representNet=None,
            outputNet=None,
            outputPro=None,
            batch_size=None,
            atomref=None,
            means=None,
            stddevs=None,
            scheduler=None
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        super().__init__()
        self.save_hyperparameters("batch_size", "learning_rate", "means", "stddevs")
        self.datamodule = datamodule
        if stddevs is None:
            if self.datamodule is not None:
                if self.datamodule.stddevs is not None:
                    self.stddevs = self.datamodule.stddevs
            else:
                self.stddevs = stddevs
        else:
            self.stddevs = stddevs
        if self.datamodule is not None and self.stddevs is None:
            raise ValueError("Please specify `stddev`.")

        if means is None:
            if self.datamodule is not None:
                if self.datamodule.means is not None:
                    self.means = self.datamodule.means
            else:
                self.means = means
        else:
            self.means = means
        if self.datamodule is not None and self.means is None:
            raise ValueError("Please specify `mean`.")

        if self.datamodule is not None and atomref is None:
            self.atomref = self.datamodule.atomref
        else:
            self.atomref = atomref

        if self.datamodule is not None and self.atomref is None:
            raise ValueError("Please specify or check your `atomref` input.")

        self.represent = [

        ]

        self.output = [

        ]
        if outputNet is None:
            outputNet = [Atomwise]
            outputPro = [Properties.energy_U0]
        if self.atomref is None:
            self.atomref = {i: None for i in outputPro}
        if self.stddevs is None:
            self.stddevs = {i: None for i in outputPro}
        if self.means is None:
            self.means = {i: None for i in outputPro}
        if representNet is None:
            representNet = [SchNet]
        for net in representNet:
            self.represent.append(
                net(
                    n_atom_embeddings=128,
                    n_filters=128,
                    n_interactions=6,
                    cutoff=10,
                    n_gaussians=50,
                    max_Z=18
                )
            )

        self.represent = ModuleList(self.represent)
        for index, net in enumerate(outputNet):
            self.output.append(
                net(
                    n_in=128,
                    n_out=1,
                    atomref=self.atomref[outputPro[index]],
                    property=outputPro[index],
                    mean=self.means[outputPro[index]],
                    stddev=self.stddevs[outputPro[index]]
                )
            )
        self.output = ModuleList(self.output)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if scheduler is not None:
            self.scheduler = scheduler["_scheduler"](
                self.optimizer,
                patience=scheduler["patience"],
                factor=scheduler["factor"],
                min_lr=scheduler["min_lr"],
                eps=scheduler["eps"]
            )

    def forward(self, inputs):
        for net in self.represent:
            inputs["representation"] = net(inputs)
        outs = {}
        for net in self.output:
            outs.update(net(inputs))
        return outs

    def training_step(self, batch, batch_idx):
        self.log("lr", self.optimizer.state_dict()['param_groups'][0]['lr'], on_epoch=True)
        outs = self.forward(batch)
        loss = F.mae_loss_for_train(outs["energy_U0"], batch["energy_U0"])
        # Logging to TensorBoard by default
        self.log('train_loss_MAE', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self.forward(batch)
        loss = F.mae_loss_for_metric(outs["energy_U0"], batch["energy_U0"])
        # Logging to TensorBoard by default
        self.log('val_loss_MAE', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        outs = self.forward(batch)
        idx = batch["_idx"]
        loss = F.mae_loss_for_metric(outs["energy_U0"], batch["energy_U0"])
        # Logging to TensorBoard by default
        self.log('test_loss_MAE', loss, on_step=True)
        return {"pred": outs["energy_U0"].cpu(),
                "ref": batch["energy_U0"].cpu(),
                "idx": idx.cpu()}

    def test_epoch_end(self, outputs):
        # np.save("outputs",outputs)
        pred = []
        refs = []
        idx = []
        for items in outputs:
            for item1, item2, item3 in zip(items["pred"], items["ref"], items["idx"]):
                pred.append(item1)
                refs.append(item2)
                idx.append(item3)
        import matplotlib.pyplot as plt
        plt.scatter(refs, pred, marker="x", s=10)
        plt.xlabel("${ E_\mathrm{reference}}$/(eV)", size=20)
        plt.ylabel("${ E_\mathrm{prediction}}$/(eV)", size=20)
        print(F.mae_loss_for_metric(torch.Tensor(pred), torch.Tensor(refs)))
        np.savez("testresult", pred=pred, refs=refs, idx=idx)
        plt.tight_layout()
        plt.show()

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return {"optimizer": self.optimizer,
                'lr_scheduler': self.scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
                }
