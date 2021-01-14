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
from lightMolNet import Properties, AtomWiseInputPropertiesList, InputPropertiesList
from lightMolNet.Struct.Atomistic.atomwise import Atomwise
from lightMolNet.Struct.nn.schnet import SchNet
from lightMolNet.logger import InfoLogger
from torch.nn import ModuleList

logger = InfoLogger(__name__)


class represent2out(torch.nn.Module):

    def __init__(
            self,
            Nrepresent=1,
            Nout=1,
            trainable=False,
            mode="sum"
    ):
        super(represent2out, self).__init__()
        self.Nrepresent = Nrepresent
        self.Nout = Nout
        self.trainable = trainable
        self.mode = mode
        assert Nrepresent == 1, "Multiple representation and multiple representation output Support will be added in the future."
        if self.trainable:
            self.trans = torch.nn.Linear(Nrepresent, Nout)
        else:
            pass

    def forward(self, represent):
        atomwiseinput = [None for i in range(len(AtomWiseInputPropertiesList.input_list))]
        atomwiseinput[AtomWiseInputPropertiesList.representation_value] = represent[0]
        # TODO:Add multiple representation and multiple representation output in the future.
        return atomwiseinput


class LitNet(pl.LightningModule):

    def __init__(
            self,
            learning_rate=1e-4,
            datamodule=None,
            representNet=None,
            n_atom_embeddings=128,
            n_filters=128,
            n_interactions=6,
            cutoff=10,
            n_gaussians=50,
            max_Z=18,
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
                    stddevs = self.datamodule.stddevs
        self.stddevs = stddevs
        if self.datamodule is not None and self.stddevs is None:
            logger.warning("No stddev specified, using stddev={Properties.energy_U0: 1}.")
            self.stddevs = {Properties.energy_U0: 1}
            # raise ValueError("Please specify `stddev`.")

        if means is None:
            if self.datamodule is not None:
                if self.datamodule.means is not None:
                    means = self.datamodule.means
        self.means = means
        if self.datamodule is not None and self.means is None:
            logger.warning("No means specified, using means={Properties.energy_U0: 0}.")
            self.means = {Properties.energy_U0: 0}
            # raise ValueError("Please specify `mean`.")

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
            self.outputNet = [Atomwise]
            self.outputPro = {Properties.energy_U0: 0}
        else:
            self.outputNet = outputNet
            self.outputPro = outputPro
        if self.atomref is None:
            self.atomref = {i: None for i in self.outputPro}
        if self.stddevs is None:
            self.stddevs = {i: None for i in self.outputPro}
        if self.means is None:
            self.means = {i: None for i in self.outputPro}
        if representNet is None:
            representNet = [SchNet]
        else:
            self.representNet = representNet

        self.represent2out = represent2out(len(self.representNet), len(self.outputNet))

        self.n_atom_embeddings = n_atom_embeddings
        self.n_filters = n_filters
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.n_gaussians = n_gaussians
        self.max_Z = max_Z

        for net in representNet:
            self.represent.append(
                net(
                    n_atom_embeddings=self.n_atom_embeddings,
                    n_filters=self.n_filters,
                    n_interactions=self.n_interactions,
                    cutoff=self.cutoff,
                    n_gaussians=self.n_gaussians,
                    max_Z=self.max_Z
                )
            )

        self.represent = ModuleList(self.represent)
        for index, net in enumerate(outputNet):
            self.output.append(
                net(
                    n_in=self.n_atom_embeddings,
                    n_out=1,
                    atomref=self.atomref[self.outputPro[index]],
                    property=self.outputPro[index],
                    mean=self.means[self.outputPro[index]],
                    stddev=self.stddevs[self.outputPro[index]]
                )
            )
        self.output = ModuleList(self.output)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if scheduler is not None:
            schedulerpara = {k: v for k, v in scheduler.items() if k != "_scheduler"}
            self.scheduler = scheduler["_scheduler"](
                self.optimizer,
                **schedulerpara
            )

    def forward(
            self,
            inputs
    ):
        """

        Parameters
        ----------
        inputs:list
            check order of `lightMolNet.InputPropertiesList`

        Returns
        -------

        """
        # representation parts
        representation = []
        for net in self.represent:
            representation.append(net(inputs))

        # representation2out parts
        represent2out = self.represent2out(representation)
        for index, item in enumerate(AtomWiseInputPropertiesList.input_list):
            if item != AtomWiseInputPropertiesList.input_list[AtomWiseInputPropertiesList.representation_value]:
                represent2out[index] = inputs[InputPropertiesList.input_list.index(item)]

        # output parts
        outs = {}

        for net in self.output:
            outs.update(net(represent2out))
        return outs

    def training_step(self, batch, batch_idx):
        batch, y = batch
        Nouts = len(self.outputPro)

        self.log("lr", self.optimizer.state_dict()['param_groups'][0]['lr'], on_epoch=True)

        outs = self.forward(batch)
        loss = torch.zeros([Nouts])
        for outPro in range(Nouts):
            loss[outPro] = F.mae_loss_for_train(outs[self.outputPro[outPro]], y[outPro])
            # Logging to TensorBoard by default
            self.log('train_{}_loss_MAE'.format(outPro), loss[outPro], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return torch.mean(loss)

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        Nouts = len(self.outputPro)

        outs = self.forward(batch)
        loss = torch.zeros([Nouts])
        for outPro in range(Nouts):
            loss[outPro] = F.mae_loss_for_train(outs[self.outputPro[outPro]], y[outPro])
            # Logging to TensorBoard by default
            self.log('val_{}_loss_MAE'.format(outPro), loss[outPro], on_step=True)
        return torch.mean(loss)

    def test_step(self, batch, batch_idx):
        batch, y = batch
        Nouts = len(self.outputPro)

        outs = self.forward(batch)
        loss = torch.zeros([Nouts])
        test_result = {}
        for outPro in range(Nouts):
            test_result.update({f"pred{outPro}": None,
                                f"ref{outPro}": None})
            loss[outPro] = F.mae_loss_for_train(outs[self.outputPro[outPro]], y[outPro])
            # Logging to TensorBoard by default
            self.log('test_{}_loss_MAE'.format(outPro), loss[outPro], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            test_result.update({f"pred{outPro}": outs[self.outputPro[outPro]].cpu(),
                                f"ref{outPro}": y[outPro].cpu()})
        return test_result

    def test_epoch_end(self, outputs):
        # np.save("outputs",outputs)
        Nouts = len(self.outputPro)
        pred = []
        refs = []
        for outPro in range(Nouts):
            pred.append([])
            refs.append([])
            for items in outputs:
                for item1, item2 in zip(items[f"pred{outPro}"], items[f"ref{outPro}"]):
                    pred[outPro].append(item1)
                    refs[outPro].append(item2)
            import matplotlib.pyplot as plt
            plt.scatter(refs, pred, marker="x", s=10)
            plt.xlabel("${ E_\mathrm{reference}}$/(eV)", size=20)
            plt.ylabel("${ E_\mathrm{prediction}}$/(eV)", size=20)
            print(F.mae_loss_for_metric(torch.Tensor(pred), torch.Tensor(refs)))
            plt.tight_layout()
            plt.show()
        np.savez("testresult", pred=pred, refs=refs)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        if getattr(self, "scheduler", None) is not None:
            return {"optimizer": self.optimizer,
                    'lr_scheduler': self.scheduler,
                    'monitor': 'val_0_loss_MAE',
                    'interval': 'epoch'
                    }
        else:
            return {"optimizer": self.optimizer,
                    'monitor': 'val_0_loss_MAE',
                    'interval': 'epoch'
                    }

    def todict(self):
        state_dict = self.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.detach().cpu().numpy()
        return state_dict
