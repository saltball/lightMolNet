# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : __init__.py.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import ModuleList

import lightMolNet.Module.functional as F
from lightMolNet import Properties, AtomWiseInputPropertiesList, InputPropertiesList
from lightMolNet.Struct.Atomistic.atomwise import Atomwise
from lightMolNet.Struct.nn.schnet import SchNet
from lightMolNet.logger import InfoLogger

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


class LitNetParent(pl.LightningModule):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 datamodule,
                 scheduler,
                 optimizer_kwargs={},
                 optimizer=torch.optim.Adam,
                 representNet=None,
                 stddevs=None,
                 means=None,
                 atomref=None,
                 cal_distance=True,
                 *args, **kwargs):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cal_distance=cal_distance
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("batch_size", "learning_rate")
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler

        self.represent = [

        ]

        self.output = [

        ]
        if self.datamodule is not None and atomref is None:
            self.atomref = self.datamodule.atomref
        else:
            self.atomref = atomref

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

        if self.datamodule is not None and self.atomref is None:
            raise ValueError("Please specify or check your `atomref` input.")

        self.init(representNet=representNet, *args, **kwargs)

        self.optimizer = self.optimizer(self.parameters(), lr=self.hparams.learning_rate, **self.optimizer_kwargs)
        if self.scheduler is not None:
            schedulerpara = {k: v for k, v in self.scheduler.items() if k != "_scheduler"}
            self.scheduler = self.scheduler["_scheduler"](
                self.optimizer,
                **schedulerpara
            )

    def init(self, *args, **kwargs):
        raise NotImplementedError("Don't override `__init__()` method.")

    def training_step(self, batch, batch_idx):
        batch, y = batch
        Nouts = len(self.outputPro)

        self.log("lr", self.optimizer.state_dict()['param_groups'][0]['lr'], on_epoch=True)

        outs = self.forward(batch)
        loss = torch.zeros([Nouts], device=self.device)
        for outPro in range(Nouts):
            loss[outPro] = F.mae_loss_for_train(outs[self.outputPro[outPro]], y[outPro])
            # Logging to TensorBoard by default
            self.log('train_{}_loss_MAE'.format(outPro), loss[outPro], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return torch.mean(loss)

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        Nouts = len(self.outputPro)

        outs = self.forward(batch)
        loss = torch.zeros([Nouts], device=self.device)
        for outPro in range(Nouts):
            loss[outPro] = F.mae_loss_for_metric(outs[self.outputPro[outPro]], y[outPro])
            # Logging to TensorBoard by default
            self.log('val_{}_loss_MAE'.format(outPro), loss[outPro], on_step=True, logger=True)
        return torch.mean(loss).detach()

    def test_step(self, batch, batch_idx):
        batch, y = batch
        Nouts = len(self.outputPro)

        outs = self.forward(batch)
        loss = torch.zeros([Nouts], device=self.device)
        test_result = {}
        for outPro in range(Nouts):
            test_result.update({f"pred{outPro}": None,
                                f"ref{outPro}": None})
            loss[outPro] = F.mae_loss_for_metric(outs[self.outputPro[outPro]], y[outPro])
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


class LitNet(LitNetParent):
    def init(
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
            scheduler=None,
            **kwargs
    ):
        """


        Parameters
        ----------
        learning_rate
        datamodule
        representNet
        n_atom_embeddings
        n_filters
        n_interactions
        cutoff
        n_gaussians
        max_Z
        outputNet
        outputPro
        batch_size
        atomref
        means
        stddevs
        scheduler

        """

        self.representNet = representNet
        if representNet is None:
            self.representNet = [SchNet]
        else:
            self.representNet = representNet

        self.outputNet = outputNet
        if self.outputNet is None:
            self.outputNet = [Atomwise]
            self.outputPro = [Properties.energy_U0]
        else:
            self.outputNet = outputNet
            self.outputPro = outputPro

        if self.atomref is None:
            logger.warning("`atomref1 is initial as 0.")
            self.atomref = {i: 0 for i in self.outputPro}
        if self.stddevs is None:
            logger.warning("`stddevs` is initial as 1.")
            self.stddevs = {i: 1 for i in self.outputPro}
        if self.means is None:
            logger.warning("`means` is initial as 0.")
            self.means = {i: 0 for i in self.outputPro}

        self.represent2out = represent2out(len(self.representNet), len(self.outputNet))

        self.n_atom_embeddings = n_atom_embeddings
        self.n_filters = n_filters
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.n_gaussians = n_gaussians
        self.max_Z = max_Z

        for net in self.representNet:
            self.represent.append(
                net(
                    n_atom_embeddings=self.n_atom_embeddings,
                    n_filters=self.n_filters,
                    n_interactions=self.n_interactions,
                    cutoff=self.cutoff,
                    n_gaussians=self.n_gaussians,
                    max_Z=self.max_Z,
                    cal_distance=self.cal_distance,
                    **kwargs
                )
            )

        self.represent = ModuleList(self.represent)

        for index, net in enumerate(self.outputNet):
            self.output.append(
                net(
                    n_in=self.n_atom_embeddings,
                    n_out=1,
                    atomref=self.atomref[self.outputPro[index]],
                    property=self.outputPro[index],
                    mean=self.means[self.outputPro[index]],
                    stddev=self.stddevs[self.outputPro[index]],
                    **kwargs
                )
            )
        self.output = ModuleList(self.output)

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

    def _freeze_represent(self,rep_idx=None):
        if rep_idx:
            net = self.represent[rep_idx]
            for param in net.parameters():
                param.requires_grad = False
        else:
            for net in self.represent:
                for param in net.parameters():
                    param.requires_grad = False

    def _freeze_output(self,out_idx=None):
        if out_idx:
            for param in net.parameters():
                param.requires_grad = False
        else:
            for net in self.output:
                for param in net.parameters():
                    param.requires_grad = False